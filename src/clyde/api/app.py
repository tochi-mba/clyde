"""The three routes, and the startup that decides whether they can work.

Startup does the three things that are true of the machine rather than of a request: find the
binary, make the sandbox, and ask the installed CLI which tools it would load. None of them
fail startup. A service that refuses to boot because `claude` is missing gives an operator a
crash loop; one that boots and says so on `/ready` gives them a sentence.
"""

from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any

from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse

from clyde import VERSION
from clyde.cli import argv as argv_mod
from clyde.cli.locate import ClaudeNotFoundError, find
from clyde.cli.run import ClaudeFailedError, ClaudeTimeoutError, run
from clyde.cli.sandbox import Sandbox
from clyde.core.config import Settings, load_settings
from clyde.core.logs import configure as configure_logs
from clyde.openai import translate
from clyde.openai.errors import Problem, from_error
from clyde.openai.service import PROBE_PROMPT, Runtime, tools_from_init

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

log = logging.getLogger("clyde")

PROBE_TIMEOUT = 120.0
"""The startup probe still runs a turn, so it gets a real ceiling -- but a shorter one than a
request, because a machine that cannot answer `ok` in two minutes is not going to serve."""


async def learn_tools(
    binary: str,
    sandbox: Sandbox,
    timeout: float = PROBE_TIMEOUT,  # noqa: ASYNC109 - the probe kills its own child
) -> tuple[str, ...]:
    """Ask the installed CLI what it loads, so the disallow list matches the binary.

    A failure here is not fatal: the service starts with an empty list, which costs tokens
    rather than correctness, and `/ready` reports it.
    """
    process = await asyncio.create_subprocess_exec(
        *argv_mod.probe(binary),
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.DEVNULL,
        cwd=str(sandbox.root),
    )
    try:
        out, _ = await asyncio.wait_for(
            process.communicate(PROBE_PROMPT.encode("utf-8")), timeout=timeout
        )
    except TimeoutError:
        process.kill()
        await process.wait()
        return ()
    return tools_from_init(out.decode("utf-8", "replace"))


async def build_runtime(settings: Settings) -> Runtime:
    """Everything decided once. Never raises: a broken machine becomes `problem`."""
    sandbox = Sandbox.create()
    limit = asyncio.Semaphore(settings.max_concurrent)
    try:
        binary = find(settings.claude_binary)
    except ClaudeNotFoundError as error:
        return Runtime(
            binary="",
            sandbox=sandbox,
            limit=limit,
            timeout=settings.timeout_seconds,
            default_model=settings.default_model,
            stdin_limit=settings.stdin_limit_bytes,
            problem=str(error),
        )
    return Runtime(
        binary=binary,
        sandbox=sandbox,
        limit=limit,
        timeout=settings.timeout_seconds,
        default_model=settings.default_model,
        stdin_limit=settings.stdin_limit_bytes,
        disallowed=await learn_tools(binary, sandbox),
    )


def create_app(settings: Settings | None = None) -> FastAPI:
    """The app. `settings` is injected in tests; production reads the environment."""
    config = settings or load_settings()
    configure_logs(config.log_level, config.log_format)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.runtime = await build_runtime(config)
        yield

    app = FastAPI(title="clyde", version=VERSION, lifespan=lifespan)

    def runtime() -> Runtime:
        value: Runtime = app.state.runtime
        return value

    @app.get("/healthy")
    async def healthy() -> dict[str, str]:
        """The process is up. Says nothing about whether it can answer."""
        return {"status": "ok", "version": VERSION}

    @app.get("/ready")
    async def ready() -> JSONResponse:
        """Whether a call would work, and why not when it would not."""
        live = runtime()
        body = {
            "status": "ok" if live.ready else "degraded",
            "version": VERSION,
            "checks": {
                "claude": {
                    "status": "ok" if live.ready else "degraded",
                    "detail": {"binary": live.binary, "reason": live.problem or None},
                },
                "tools_disallowed": {
                    "status": "ok" if live.disallowed else "degraded",
                    "detail": {"count": len(live.disallowed)},
                },
            },
        }
        return JSONResponse(body, status_code=200 if live.ready else 503)

    @app.get("/v1/models")
    async def models() -> dict[str, Any]:
        """The aliases the CLI accepts, in OpenAI's listing shape."""
        live = runtime()
        return {
            "object": "list",
            "data": [
                {"id": name, "object": "model", "created": 0, "owned_by": "anthropic"}
                for name in live.models
            ],
        }

    def refuse(body: object, live: Runtime) -> Problem | None:
        """Why this request cannot be served, before anything is spawned.

        Each of these is a 400 the caller can fix by asking differently, except the last,
        which is this machine's fault and says so with a `Retry-After`.
        """
        if not isinstance(body, dict):
            return Problem(400, "the body must be an object", code="invalid_request_error")
        if body.get("tools"):
            return Problem(
                400,
                "clyde offers no tools; ask for structured output with `response_format` instead",
                code="tools_unsupported",
            )
        if not live.ready:
            return Problem(503, live.problem, code="claude_not_found", retry_after=30)
        return None

    def as_response(problem: Problem) -> JSONResponse:
        # Logged as well as returned. A refusal that exists only in the caller's response body
        # is invisible to whoever runs this: three separate failures here were diagnosed by
        # guessing, because the access log said `500` and nothing else. The message is already
        # capped and stripped of prompt text by `run.first_line`, which is what makes it safe
        # to write down.
        log.warning("refused: %s %s: %s", problem.status, problem.code, problem.message)
        headers = {"Retry-After": str(problem.retry_after)} if problem.retry_after else None
        return JSONResponse(problem.body(), status_code=problem.status, headers=headers)

    @app.post("/v1/chat/completions")
    async def completions(request: Request) -> Response:
        live = runtime()
        try:
            body = await request.json()
        except ValueError:
            # Unparseable input is the caller's to fix. Letting it escape turned a bad body
            # into a 500 that read `clyde failed: JSONDecodeError`, which blames the service.
            return as_response(
                Problem(400, "the body is not valid JSON", code="invalid_request_error")
            )
        refusal = refuse(body, live)
        if refusal is not None:
            return as_response(refusal)
        try:
            outcome = await live.complete(body)
        except (ClaudeFailedError, ClaudeTimeoutError, ValueError, OSError) as error:
            return as_response(from_error(error))
        if outcome.is_error:
            return as_response(from_error(ClaudeFailedError(outcome.result or outcome.why)))
        model = str(body.get("model") or live.default_model)
        if body.get("stream"):
            # The reply already exists in full -- `claude -p` returns it at once -- so this is
            # the same answer delivered down the streaming shape rather than progressive
            # generation. A caller that asks for `stream: true` is asking for the protocol,
            # and some have no other path: Lucy attaches a chunk projector to every turn, so
            # refusing this refused Lucy entirely.
            chunks = translate.stream_chunks(outcome, model=model)
            return StreamingResponse(
                iter(chunks),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"},
            )
        return JSONResponse(translate.to_completion(outcome, model=model))

    return app


__all__ = ["PROBE_TIMEOUT", "build_runtime", "create_app", "learn_tools", "run"]

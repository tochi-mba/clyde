"""The three routes, against a runtime whose `complete` is scripted.

No process is spawned. What is being tested is the HTTP surface: which refusals happen before
anything runs, which failures become which status, and that a reply comes back in the shape a
chat-completions client expects.
"""

from __future__ import annotations

import asyncio
from typing import Any

import httpx
import pytest
from asgi_lifespan import LifespanManager

from clyde.api.app import create_app
from clyde.cli.run import ClaudeFailedError, ClaudeTimeoutError, Outcome
from clyde.cli.sandbox import Sandbox
from clyde.core.config import Settings
from clyde.openai.service import UNPROVEN, Loaded, Runtime


def runtime(tmp_path: Any, **kwargs: Any) -> Runtime:
    base: dict[str, Any] = {
        "binary": "/usr/local/bin/claude",
        "sandbox": Sandbox.create(tmp_path),
        "limit": asyncio.Semaphore(1),
        "timeout": 30.0,
        "default_model": "sonnet",
        "stdin_limit": 1_000_000,
        "loaded": Loaded(),
    }
    base.update(kwargs)
    return Runtime(**base)


async def client(live: Runtime, monkeypatch: pytest.MonkeyPatch) -> tuple[httpx.AsyncClient, Any]:
    """An app whose startup is replaced wholesale.

    `build_runtime` locates a binary and then *runs* it to see what it loads. Letting the real
    one fire in a unit test would spawn a `claude` process per test -- slow, dependent on a
    login, and billed to somebody. The probe has its own tests; here it is stubbed out.
    """

    async def fake_build(settings: Settings) -> Runtime:
        return live

    monkeypatch.setattr("clyde.api.app.build_runtime", fake_build)
    app = create_app(Settings())
    manager = LifespanManager(app, startup_timeout=10)
    await manager.__aenter__()
    transport = httpx.ASGITransport(app=app)
    return httpx.AsyncClient(transport=transport, base_url="http://clyde"), manager


async def call(
    live: Runtime, monkeypatch: pytest.MonkeyPatch, path: str = "/v1/chat/completions", **body: Any
) -> httpx.Response:
    http, manager = await client(live, monkeypatch)
    try:
        if path == "/v1/chat/completions":
            payload = body or {"messages": [{"role": "user", "content": "hi"}]}
            return await http.post(path, json=payload)
        return await http.get(path)
    finally:
        await http.aclose()
        await manager.__aexit__(None, None, None)


# --- probes ------------------------------------------------------------------------------------


async def test_healthy_says_nothing_about_readiness(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    response = await call(runtime(tmp_path), monkeypatch, path="/healthy")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


async def test_ready_is_ok_when_the_binary_was_found_and_nothing_loaded(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    response = await call(runtime(tmp_path), monkeypatch, path="/ready")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["checks"]["claude"]["status"] == "ok"
    assert body["checks"]["lockdown"] == {
        "status": "ok",
        "detail": {"tools": [], "mcp_servers": [], "reason": None},
    }


async def test_ready_is_degraded_and_says_why(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """503 with a sentence beats a crash loop when `claude` is simply not installed."""
    response = await call(
        runtime(tmp_path, binary="", problem="could not find `claude`", loaded=None),
        monkeypatch,
        path="/ready",
    )
    assert response.status_code == 503
    assert "could not find" in response.json()["checks"]["claude"]["detail"]["reason"]


async def test_ready_names_whatever_the_probe_saw_load(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The check that used to read `tools_disallowed: {count: 157}` while TodoWrite loaded. It
    now reports what a probe under the calls' own flags saw, and it is not `ok` unless that
    was nothing."""
    loaded = Loaded(tools=("TodoWrite",), mcp_servers=("claude.ai Gmail",))
    response = await call(runtime(tmp_path, loaded=loaded), monkeypatch, path="/ready")
    assert response.status_code == 503
    body = response.json()
    assert body["status"] == "degraded"
    assert body["checks"]["claude"]["status"] == "ok", "the binary is not what is wrong"
    lockdown = body["checks"]["lockdown"]
    assert lockdown["status"] == "degraded"
    assert lockdown["detail"]["tools"] == ["TodoWrite"]
    assert lockdown["detail"]["mcp_servers"] == ["claude.ai Gmail"]
    assert "tools: TodoWrite; MCP servers: claude.ai Gmail" in lockdown["detail"]["reason"]


async def test_ready_says_when_the_probe_could_not_be_read(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`null` rather than `[]`: an empty list would claim a proof nothing produced."""
    response = await call(runtime(tmp_path, loaded=None), monkeypatch, path="/ready")
    assert response.status_code == 503
    assert response.json()["checks"]["lockdown"] == {
        "status": "degraded",
        "detail": {"tools": None, "mcp_servers": None, "reason": UNPROVEN},
    }


async def test_models_lists_the_aliases(tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    body = (await call(runtime(tmp_path), monkeypatch, path="/v1/models")).json()
    assert body["object"] == "list"
    assert {row["id"] for row in body["data"]} == {"sonnet", "opus", "haiku", "fable"}


# --- refusals, before anything is spawned --------------------------------------------------------


async def test_streaming_is_served_as_server_sent_events(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Refusing `stream: true` refused Lucy entirely: it attaches a chunk projector to every
    turn, so it never asks for the blocking path. Found by running it, not by reading it."""
    live = runtime(tmp_path)

    async def complete(body: dict[str, Any]) -> Outcome:
        return Outcome(result="hello", session_id="abc")

    monkeypatch.setattr(live, "complete", complete)
    response = await call(live, monkeypatch, messages=[], stream=True)
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    body = response.text
    assert '"content":"hello"' in body
    assert body.rstrip().endswith("data: [DONE]")


async def test_tools_are_refused_rather_than_dropped(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Silently ignoring them would let a caller think its tools were offered."""
    response = await call(runtime(tmp_path), monkeypatch, messages=[], tools=[{"type": "function"}])
    assert response.json()["error"]["code"] == "tools_unsupported"


async def test_a_non_object_body_is_refused(tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    http, manager = await client(runtime(tmp_path), monkeypatch)
    try:
        response = await http.post("/v1/chat/completions", json=["not", "an", "object"])
    finally:
        await http.aclose()
        await manager.__aexit__(None, None, None)
    assert response.status_code == 400


async def test_a_call_with_no_binary_is_503_with_retry_after(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With no binary there was no probe either; the binary is the sentence worth reading."""
    response = await call(
        runtime(tmp_path, binary="", problem="no claude here", loaded=None),
        monkeypatch,
        messages=[],
    )
    assert response.status_code == 503
    assert response.headers["Retry-After"] == "30"
    assert response.json()["error"]["code"] == "claude_not_found"


@pytest.mark.parametrize(
    ("loaded", "reason"),
    [(Loaded(tools=("TodoWrite",)), "tools: TodoWrite"), (None, UNPROVEN)],
    ids=["a tool loaded", "the probe could not be read"],
)
async def test_a_call_is_refused_unless_nothing_was_shown_to_load(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch, loaded: Loaded | None, reason: str
) -> None:
    """A service that knows a tool is loaded and serves anyway is the failure this replaces,
    so nothing is spawned. 503 because it is this machine and not the request: a caller falls
    back to another provider rather than asking again differently."""
    live = runtime(tmp_path, loaded=loaded)
    spawned: list[dict[str, Any]] = []

    async def complete(body: dict[str, Any]) -> Outcome:
        spawned.append(body)
        return Outcome(result="should never be seen")

    monkeypatch.setattr(live, "complete", complete)
    response = await call(live, monkeypatch)
    assert response.status_code == 503
    assert response.headers["Retry-After"] == "30"
    assert response.json()["error"]["code"] == "not_locked_down"
    assert reason in response.json()["error"]["message"]
    assert spawned == []


# --- the happy path and the failures ------------------------------------------------------


async def test_a_reply_comes_back_in_chat_completions_shape(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    live = runtime(tmp_path)

    async def complete(body: dict[str, Any]) -> Outcome:
        return Outcome(result="hello", session_id="abc", usage={"output_tokens": 2})

    monkeypatch.setattr(live, "complete", complete)
    body = (await call(live, monkeypatch)).json()
    assert body["choices"][0]["message"] == {"role": "assistant", "content": "hello"}
    assert body["choices"][0]["finish_reason"] == "stop"
    assert body["usage"]["completion_tokens"] == 2
    assert body["id"] == "chatcmpl-abc"


async def test_the_cli_reporting_its_own_error_becomes_a_failure(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Exit 0 with `is_error: true` is the case that would otherwise look like a reply."""
    live = runtime(tmp_path)

    async def complete(body: dict[str, Any]) -> Outcome:
        return Outcome(result="something went wrong", is_error=True)

    monkeypatch.setattr(live, "complete", complete)
    response = await call(live, monkeypatch)
    assert response.status_code == 502
    assert response.json()["error"]["message"] == "something went wrong"


async def test_an_error_with_no_text_of_its_own_says_why_the_run_stopped(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The CLI's error results carry no `result`, and every one of them used to reach the
    caller as "claude reported an error" -- a usage limit, a turn limit and a tool the model
    reached for alike."""
    live = runtime(tmp_path)

    async def complete(body: dict[str, Any]) -> Outcome:
        return Outcome(
            result="",
            is_error=True,
            subtype="error_max_turns",
            num_turns=2,
            terminal_reason="max_turns",
        )

    monkeypatch.setattr(live, "complete", complete)
    response = await call(live, monkeypatch)
    assert response.status_code == 502
    assert response.json()["error"] == {
        "message": "claude stopped with error_max_turns after 2 turns (max_turns)",
        "type": "claude_failed",
        "code": "claude_failed",
    }


@pytest.mark.parametrize("stream", [False, True], ids=["blocking", "streaming"])
@pytest.mark.parametrize("result", ["", "\n  \n"], ids=["empty", "only whitespace"])
async def test_an_empty_reply_is_a_failure_the_caller_can_see(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch, stream: bool, result: str
) -> None:
    """It used to be a 200 with `"content": ""`, and the caller ended its turn a success having
    said nothing to anybody. Streaming and blocking are refused alike: Lucy only ever streams,
    so a check on one path alone would have missed the caller it was for."""
    live = runtime(tmp_path)

    async def complete(body: dict[str, Any]) -> Outcome:
        return Outcome(result=result, stop_reason="tool_use", usage={"output_tokens": 266})

    monkeypatch.setattr(live, "complete", complete)
    response = await call(live, monkeypatch, messages=[], stream=stream)
    assert response.status_code == 502
    assert response.headers["content-type"].startswith("application/json")
    assert response.json()["error"] == {
        "message": (
            "the model finished after 1 turn without producing any text (stop_reason: tool_use)"
        ),
        "type": "claude_failed",
        "code": "claude_failed",
    }


async def test_a_logged_out_cli_is_403_and_not_retried(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    live = runtime(tmp_path)

    async def complete(body: dict[str, Any]) -> Outcome:
        return Outcome(result="Not logged in. Please run /login", is_error=True)

    monkeypatch.setattr(live, "complete", complete)
    response = await call(live, monkeypatch)
    assert response.status_code == 403
    assert "Retry-After" not in response.headers


@pytest.mark.parametrize(
    ("error", "status"),
    [
        (ClaudeTimeoutError("slow"), 504),
        (ClaudeFailedError("exited 2"), 502),
        (ValueError("too big"), 400),
    ],
)
async def test_each_failure_becomes_its_own_status(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch, error: Exception, status: int
) -> None:
    live = runtime(tmp_path)

    async def complete(body: dict[str, Any]) -> Outcome:
        raise error

    monkeypatch.setattr(live, "complete", complete)
    assert (await call(live, monkeypatch)).status_code == status


async def test_a_body_that_is_not_json_is_the_callers_fault(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """It used to escape the route and surface as `clyde failed: JSONDecodeError` -- a 500,
    which tells a caller to retry something only they can fix."""
    http, manager = await client(runtime(tmp_path), monkeypatch)
    try:
        response = await http.post(
            "/v1/chat/completions",
            content=b"{not json",
            headers={"content-type": "application/json"},
        )
    finally:
        await http.aclose()
        await manager.__aexit__(None, None, None)
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "invalid_request_error"

"""The one module that starts a process, and everything that can go wrong when it does.

An import contract names this file: nothing else in `clyde` may import `subprocess`. Spawning
is the genuinely dangerous thing this service does, and keeping it in one place means the
review question is "what does run.py do" rather than "where does this end up running code".

The CLI reports failure three ways and they mean different things:

* it never starts, because the binary is missing
* it starts, exits 0, and says `is_error: true` in its own JSON
* it starts and exits non-zero with something on stderr

Collapsing those is how a harness becomes unusable at 2 a.m., so :class:`Outcome` keeps them
apart and `openai.errors` decides what each becomes on the wire.
"""

from __future__ import annotations

import asyncio
import json
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

SIGTERM_EXIT = 143
"""The CLI's documented exit code when SIGTERM stopped it. The turn is unfinished, not failed,
and a caller must not retry it as though the model had refused."""

STDERR_KEPT = 300
"""How much of stderr may reach a response body. The first line is enough to act on; the rest
routinely carries file paths and prompt fragments, and a response body is the wrong place for
either. `test_run.py` asserts a planted secret-shaped string never survives this."""


class ClaudeTimeoutError(RuntimeError):
    """The call outlived its ceiling and the process was killed."""


class ClaudeFailedError(RuntimeError):
    """The process exited non-zero, or said nothing this module can parse."""

    def __init__(self, message: str, *, exit_code: int = 0) -> None:
        super().__init__(message)
        self.exit_code = exit_code


@dataclass(frozen=True, slots=True)
class Outcome:
    """One finished run, parsed but not yet translated into anybody's dialect."""

    result: str
    """The reply text. When a schema was asked for, the CLI already puts the JSON here as a
    string, so it needs no re-serialising before a caller parses it back out."""

    is_error: bool = False
    subtype: str = "success"
    """How the run ended, in the CLI's own words. `success` or an `error_*` reason."""

    stop_reason: str = "end_turn"
    num_turns: int = 1
    session_id: str = ""
    cost_usd: float = 0.0
    usage: dict[str, Any] = field(default_factory=dict)
    model: str = ""

    @property
    def hit_turn_limit(self) -> bool:
        """The run stopped because it ran out of turns, rather than because it finished.

        Read from `subtype`, and deliberately **not** from `num_turns`. A structured-output
        call reports `num_turns: 2` and `stop_reason: tool_use` on a completely successful
        run, because the CLI emits the JSON through an internal tool. Treating that as
        truncation would tell every caller that every plan-shaped turn ran out of room, and
        a caller that believes it will offer to resume a turn that already finished.
        """
        return self.subtype.startswith("error_max_turns")


LEADING_TOOL_CALL = re.compile(
    r"\A\s*<(?:\w+:)?(invoke|function_calls)\b[^>]*>.*?</(?:\w+:)?\1\s*>",
    re.DOTALL | re.IGNORECASE,
)
"""A tool-call block at the very front of a reply, which is where a leaked one appears.

Claude Code is trained to emit `<invoke name="...">` when it decides to call something, and it
does so with every tool disallowed, because a prompt full of named operations reads exactly
like a set of tools. With no tool to match, the CLI hands the block through as part of the
reply. Observed, twice:

    <invoke name="none">
    </invoke>
    {"steps":[{"id":"ls","op":"workspace.list","input":{"path":"."}}]}

That is a perfectly good structured reply with four lines in front of it, and a caller doing
`json.loads` on the whole thing reads it as prose. Asking the model not to, in the prompt, did
not stop it.

Anchored at the start on purpose, and no further. A block in the middle of a reply is the model
writing *about* the syntax -- somebody asking clyde to explain Claude Code would get exactly
that -- and deleting it would be this harness editing an answer it was not asked to edit.
"""


def without_leading_tool_calls(text: str) -> str:
    """`text` with any leaked tool-call blocks taken off the front.

    Returns the original whenever stripping would leave nothing: an empty reply is a worse
    answer than a strange one, and it is indistinguishable downstream from the model saying
    nothing at all.
    """
    stripped = text
    while match := LEADING_TOOL_CALL.match(stripped):
        stripped = stripped[match.end() :]
    stripped = stripped.strip()
    return stripped or text


def parse(stdout: str) -> Outcome:
    """The CLI's `--output-format json` object, as an :class:`Outcome`.

    Anything unparseable is a failure rather than an empty reply: a caller that cannot tell
    "the model said nothing" from "the harness could not read the answer" will treat the
    second as the first exactly once, in production.
    """
    try:
        payload = json.loads(stdout)
    except json.JSONDecodeError as error:
        msg = f"claude did not return JSON: {stdout[:STDERR_KEPT]!r}"
        raise ClaudeFailedError(msg) from error
    if not isinstance(payload, dict):
        msg = f"claude returned {type(payload).__name__}, not an object"
        raise ClaudeFailedError(msg)

    models = payload.get("modelUsage")
    named = next(iter(models), "") if isinstance(models, dict) else ""
    return Outcome(
        result=without_leading_tool_calls(str(payload.get("result") or "")),
        is_error=bool(payload.get("is_error")),
        subtype=str(payload.get("subtype") or "success"),
        stop_reason=str(payload.get("stop_reason") or "end_turn"),
        num_turns=int(payload.get("num_turns") or 1),
        session_id=str(payload.get("session_id") or ""),
        cost_usd=float(payload.get("total_cost_usd") or 0.0),
        usage=usage if isinstance(usage := payload.get("usage"), dict) else {},
        model=str(payload.get("model") or named),
    )


def first_line(text: str) -> str:
    """The actionable part of stderr, capped. Never the whole stream."""
    for line in text.splitlines():
        stripped = line.strip()
        if stripped:
            return stripped[:STDERR_KEPT]
    return ""


async def run(
    argv: Sequence[str],
    *,
    stdin: str,
    cwd: Path,
    timeout: float,  # noqa: ASYNC109 - the ceiling is this service's policy, and the
    # timeout path must kill the child; a caller-side `asyncio.timeout` cannot reap it.
) -> Outcome:
    """Spawn, feed stdin, wait, and reap. Never leaves a process behind.

    The kill path matters as much as the happy one: a timeout that returns without killing
    the child leaves a Node process holding a model call, and a service that does that a few
    hundred times has quietly become the problem.
    """
    process = await asyncio.create_subprocess_exec(
        *argv,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        cwd=str(cwd),
    )
    try:
        out, err = await asyncio.wait_for(
            process.communicate(stdin.encode("utf-8")), timeout=timeout
        )
    except TimeoutError as error:
        process.kill()
        await process.wait()
        msg = f"claude did not finish within {timeout:g}s"
        raise ClaudeTimeoutError(msg) from error

    code = process.returncode or 0
    if code == SIGTERM_EXIT:
        msg = "claude was terminated before it finished"
        raise ClaudeFailedError(msg, exit_code=code)
    if code != 0:
        detail = first_line(err.decode("utf-8", "replace")) or "no detail on stderr"
        msg = f"claude exited {code}: {detail}"
        raise ClaudeFailedError(msg, exit_code=code)
    return parse(out.decode("utf-8", "replace"))


__all__ = [
    "LEADING_TOOL_CALL",
    "SIGTERM_EXIT",
    "STDERR_KEPT",
    "ClaudeFailedError",
    "ClaudeTimeoutError",
    "Outcome",
    "first_line",
    "parse",
    "run",
    "without_leading_tool_calls",
]

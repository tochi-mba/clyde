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
import contextlib
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

    denied: tuple[str, ...] = ()
    """Tools the model reached for and was refused, by name, from `permission_denials`.

    A name here is the model trying to *do* something itself -- write the file it was asked
    to write -- instead of answering. While the tools were disallowed by name, that was the
    commonest way a weak model's run ended in `error_during_execution`. With none loaded at
    all (`argv.LOCKDOWN`) this should stay empty, so a name in it now also says that
    something loaded which the startup probe did not see. Either way it is the one fact that
    says what to change.
    """

    terminal_reason: str = ""
    """The CLI's own word for why the run stopped (`completed`, `max_turns`, ...)."""

    @property
    def answered(self) -> bool:
        """Whether there is a reply here to hand a caller: no error, and some text in it.

        An empty `result` on a run the CLI calls a success is not an answer, for the reason
        :func:`parse` gives -- a caller cannot tell "the model said nothing" from "the harness
        lost the answer". Measured: a Haiku run whose only output was a TodoWrite call reached
        its caller as HTTP 200 with `"content": ""` and 266 completion tokens, and the caller
        ended its turn a success, having said nothing to the person it was talking to.
        """
        return not self.is_error and bool(self.result.strip())

    @property
    def why(self) -> str:
        """Why a run did not answer, in one sentence built from what the CLI reported.

        Meaningful only when :attr:`answered` is false. The CLI's error results carry no
        `result` text, and "claude reported an error" was all a caller ever heard -- for a
        model at its usage limit, a model that tried to use a tool, and a run that ran out of
        turns alike. Measured after, on a real Haiku run that ran out of turns: `claude stopped
        with error_max_turns after 2 turns (max_turns)`. A run that succeeded with nothing to
        say gets the same treatment, with `stop_reason` in place of an error, because that is
        what tells a tool call apart from a model that simply stopped.
        """
        turns = f"{self.num_turns} turn" + ("" if self.num_turns == 1 else "s")
        if self.is_error:
            sentence = f"claude stopped with {self.subtype} after {turns}"
            # `completed` is how every run ends that was not cut short, so it tells the reader
            # of a failure nothing; anything else is the CLI naming what cut it short.
            if self.terminal_reason and self.terminal_reason != "completed":
                sentence += f" ({self.terminal_reason})"
        else:
            sentence = (
                f"the model finished after {turns} without producing any text "
                f"(stop_reason: {self.stop_reason})"
            )
        if self.denied:
            sentence += (
                f"; it tried to use {', '.join(self.denied)}, which this service does not "
                "allow -- the model reached for a tool instead of answering"
            )
        return sentence

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


TOOL_CALL_TAG = re.compile(
    r"\A\s*</?(?:\w+:)?(?:invoke|function_calls)\b[^>]*>\s*\Z",
    re.IGNORECASE,
)
"""A line that is nothing but a tool-call tag, opening or closing.

Claude Code is trained to emit `<invoke name="...">` when it decides to call something, and it
does so with no tool to call, because a prompt full of named operations reads exactly like a
set of tools. With no tool to match, the CLI hands the tags through as part of the reply, and
they have arrived wrapped around the payload as well as in front of it:

    <invoke name="none">          <invoke>
    </invoke>                     {"steps":[...]}
    {"steps":[...]}              </invoke>

Both are a good structured reply with tags around it, and a caller doing `json.loads` on the
whole thing reads either as prose. Matching the *tag* rather than the block is what handles
both: an earlier version matched `<invoke>...</invoke>` and deleted everything between, which
was right for the left-hand shape and threw the reply away in the right-hand one.

The line has to be nothing but the tag. `Here is the syntax: <invoke name="x"></invoke>` is the
model writing *about* it -- which is what somebody asking clyde to explain Claude Code would
get -- and editing that would be this harness rewriting an answer it was not asked to rewrite.
"""


def without_tool_call_tags(text: str) -> str:
    """`text` with leaked tool-call tags taken off its edges.

    Only the edges: a tag line in the middle is holding two halves of something apart, and
    nothing here knows what. Returns the original whenever stripping would leave nothing --
    an empty reply is a worse answer than a strange one, and downstream it cannot be told
    apart from the model saying nothing at all.
    """

    def spare(line: str) -> bool:
        """A tag on its own, or the blank line that so often comes with one."""
        return not line.strip() or bool(TOOL_CALL_TAG.match(line))

    lines = text.splitlines()
    while lines and spare(lines[0]):
        lines.pop(0)
    while lines and spare(lines[-1]):
        lines.pop()
    stripped = "\n".join(lines).strip()
    return stripped or text


ANY_TOOL_CALL = re.compile(r"</?(?:\w+:)?(?:invoke|function_calls|parameter)\b", re.IGNORECASE)
"""Tool-call syntax anywhere in a reply, in any of the shapes it has arrived in."""


def is_mangled_call(text: str) -> bool:
    """Whether a reply is a tool call the CLI could not make.

    There are no tools in this conversation -- `argv.LOCKDOWN` loads none, and the service
    refuses to run until a probe under the same flags has shown it -- so a reply carrying
    `<invoke>` is not an answer. It is the model reaching for a channel that is not there,
    and whatever it meant to say is in a shape no caller can read. `without_tool_call_tags`
    recovers the two shapes where the tags are only wrapping; this catches the rest, where
    the syntax is tangled through the text:

        <parameter name="op">notes.search</parameter>
        </invoke>
        ```
        Wait, correcting format: here is the JSON object.
        {"steps":[...]}

    Measured at roughly one turn in three against a real caller, so it is worth a retry
    rather than a shrug. The cost of a false positive -- somebody asking clyde to explain
    Claude Code's own syntax -- is one wasted call before the same answer is returned anyway.
    """
    return bool(ANY_TOOL_CALL.search(text))


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
        result=without_tool_call_tags(str(payload.get("result") or "")),
        is_error=bool(payload.get("is_error")),
        subtype=str(payload.get("subtype") or "success"),
        stop_reason=str(payload.get("stop_reason") or "end_turn"),
        num_turns=int(payload.get("num_turns") or 1),
        session_id=str(payload.get("session_id") or ""),
        cost_usd=float(payload.get("total_cost_usd") or 0.0),
        usage=usage if isinstance(usage := payload.get("usage"), dict) else {},
        model=str(payload.get("model") or named),
        denied=_denied(payload.get("permission_denials")),
        terminal_reason=str(payload.get("terminal_reason") or ""),
    )


def _denied(value: object) -> tuple[str, ...]:
    """Each refused tool's name, once, in the order the model reached for them."""
    if not isinstance(value, list):
        return ()
    names: list[str] = []
    for entry in value:
        name = entry.get("tool_name") if isinstance(entry, dict) else None
        if isinstance(name, str) and name and name not in names:
            names.append(name)
    return tuple(names)


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
    stdout = out.decode("utf-8", "replace")
    if code != 0:
        # A non-zero exit does not always mean there is nothing to read. The CLI has exited 1
        # while writing a complete `--output-format json` object to stdout, and that object
        # says what happened -- `is_error` and `subtype` in its own words -- where the exit
        # code only says "something". When it parses, it wins.
        with contextlib.suppress(ClaudeFailedError):
            return parse(stdout)
        # Otherwise the best sentence available, wherever it is. stderr first because that is
        # where the CLI puts what it means to say; stdout second because it is sometimes the
        # only place anything appears, and `no detail on stderr` is a sentence about clyde
        # rather than about what went wrong.
        detail = (
            first_line(err.decode("utf-8", "replace"))
            or first_line(stdout)
            or "nothing on stderr or stdout"
        )
        msg = f"claude exited {code}: {detail}"
        raise ClaudeFailedError(msg, exit_code=code)
    return parse(stdout)


__all__ = [
    "ANY_TOOL_CALL",
    "SIGTERM_EXIT",
    "STDERR_KEPT",
    "TOOL_CALL_TAG",
    "ClaudeFailedError",
    "ClaudeTimeoutError",
    "Outcome",
    "first_line",
    "is_mangled_call",
    "parse",
    "run",
    "without_tool_call_tags",
]

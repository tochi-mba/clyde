"""What holds the pieces together for the life of the process.

Three things are decided once at startup rather than per request, because all three are
answers about *this machine* and none of them change between calls:

**Where `claude` is.** Looked up once; a missing binary is reported by `/ready` rather than
discovered on the first real request.

**That a call loads nothing.** Every call runs under `argv.LOCKDOWN`, which removes every
built-in tool and every MCP server by construction. At startup the CLI is run once under the
same flags and its `system/init` is read back; unless that shows nothing loaded, every call is
refused (:attr:`Runtime.unsafe`). This replaced a list of tool names to disallow, learned from
a probe run *without* those flags -- and on 2026-09-24 that list let TodoWrite into real calls
while `/ready` vouched for 157 names. `cli/argv.py` has the measurements.

**One sandbox.** An empty directory, verified before every call.

Concurrency is a semaphore, not a queue with a worker pool. Each call is a whole Node runtime;
the useful number is small, and making it configurable-but-low is more honest than pretending
this scales horizontally.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from clyde.cli import argv as argv_mod
from clyde.cli import locate
from clyde.cli.run import Outcome, is_mangled_call, run
from clyde.openai import translate

if TYPE_CHECKING:
    import asyncio

    from clyde.cli.sandbox import Sandbox

log = logging.getLogger("clyde")

PROBE_PROMPT = "ok"
"""The probe still runs a turn, so it is the shortest prompt that can end one."""

SHOWN = 5
"""How many names a sentence lists before it counts the rest. A CLI that ignored `--tools ""`
would load every built-in, and a refusal naming each of them on every call would bury the log
line after it. `/ready` lists them all."""

UNPROVEN = (
    "the startup probe did not report what a call loads, so nothing shows that calls here run "
    "without tools; refusing them until a restart probes again"
)
"""What every call is refused with when the probe never ran or could not be read."""


@dataclass(frozen=True, slots=True)
class Loaded:
    """What a run's `system/init` said it had loaded for the model to call.

    Built-in tools and MCP servers, by name: the two things `argv.LOCKDOWN` removes that a
    model can call. Skills and slash commands are removed too, and are not counted here: a
    model reaches a skill only through the Skill tool, which is a built-in and so already in
    `tools`, and a slash command is something a prompt invokes, not a model.
    """

    tools: tuple[str, ...] = ()
    mcp_servers: tuple[str, ...] = ()

    @property
    def anything(self) -> bool:
        return bool(self.tools or self.mcp_servers)

    def describe(self) -> str:
        """What loaded, as a phrase: `tools: TodoWrite; MCP servers: claude.ai Gmail`."""

        def listed(names: tuple[str, ...]) -> str:
            rest = len(names) - SHOWN
            shown = ", ".join(names[:SHOWN])
            return f"{shown} and {rest} more" if rest > 0 else shown

        kinds = (("tools", self.tools), ("MCP servers", self.mcp_servers))
        return "; ".join(f"{kind}: {listed(names)}" for kind, names in kinds if names)


def tools_from_init(stream: str) -> Loaded | None:
    """What the `system/init` event of a stream-json run said loaded.

    `None` when there is no init event to read, and deliberately not an empty :class:`Loaded`:
    "the probe said nothing loaded" and "the probe said nothing" are different answers, and
    the second is not evidence of anything. The disallow list this replaced collapsed them --
    an unreadable probe meant nothing disallowed, and every call was served anyway.

    Both lists have to be there. An init event with `tools` and no `mcp_servers` says nothing
    about MCP servers, which is not the same as saying there are none.
    """
    for line in stream.splitlines():
        if not line.strip():
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            return None
        if not (
            isinstance(event, dict)
            and event.get("type") == "system"
            and event.get("subtype") == "init"
        ):
            continue
        tools, servers = event.get("tools"), event.get("mcp_servers")
        if isinstance(tools, list) and isinstance(servers, list):
            return Loaded(tools=_names(tools), mcp_servers=_names(servers))
        return None
    return None


def _names(entries: list[Any]) -> tuple[str, ...]:
    """Each entry's name: a tool is a string, an MCP server an object with a `name`.

    Nothing is dropped. An entry with no readable name is still something that loaded, and
    leaving it out would turn "something loaded" into "nothing did".
    """
    return tuple(
        str(entry.get("name") or entry) if isinstance(entry, dict) else str(entry)
        for entry in entries
    )


@dataclass
class Runtime:
    """The live service. Built by :func:`start`, held on the app."""

    binary: str
    sandbox: Sandbox
    limit: asyncio.Semaphore
    timeout: float
    default_model: str
    stdin_limit: int
    loaded: Loaded | None = None
    """What the startup probe saw load, under the flags every call runs with. `None` until a
    probe has been read, which is the safe default: it refuses."""

    problem: str = ""
    """Why this is not usable, when it is not. Empty means ready."""

    models: tuple[str, ...] = field(default=("sonnet", "opus", "haiku", "fable"), init=False)
    """The aliases the CLI accepts. Listed for `GET /v1/models`, which is how a client
    discovers what to ask for and how a catalogue row probes this service."""

    @property
    def unsafe(self) -> str:
        """Why a call here is not known to run with nothing loaded. Empty means it is.

        Anything but empty refuses every call. The disallow list only ever reported its
        failures -- an empty list was said to cost "tokens rather than correctness" -- and it
        then failed in a way it could not report at all: a call ran with TodoWrite loaded while
        `/ready` vouched for 157 names. A service that knows a tool is loaded and serves anyway
        is that failure again with better logging.

        A probe that could not be read refuses too. It is no evidence of a tool, but it is no
        evidence of none either, and "no tools" is what every reply from here rests on. What
        that costs is a restart after a probe that failed for a passing reason, and a 503 until
        then -- which a caller reads as this provider being unavailable, and falls back from.
        """
        if self.loaded is None:
            return UNPROVEN
        if self.loaded.anything:
            return (
                "the startup probe loaded something under the flags every call runs with "
                f"({self.loaded.describe()}), so every call is refused rather than handed to a "
                "model that can reach it"
            )
        return ""

    @property
    def ready(self) -> bool:
        return not self.problem and not self.unsafe

    async def complete(self, body: dict[str, Any]) -> Outcome:
        """One chat-completions body, one process, one outcome."""
        call = translate.to_call(body, default_model=self.default_model)
        size = len(call.prompt.encode("utf-8"))
        if size > self.stdin_limit:
            msg = f"the conversation is {size} bytes, over the {self.stdin_limit}-byte limit"
            raise ValueError(msg)
        # Anything too long for a command line moves, and `fit` decides where. The sandbox
        # supplies the one thing `fit` cannot do for itself, which is write a file.
        fitted = argv_mod.fit(call, binary=self.binary, spill=self.sandbox.spill)
        if fitted is not call:
            # Worth a line every time: a moved schema changes what the model is able to
            # return, and the only symptom downstream is a reply that is shaped wrong.
            log.info(
                "fitted: %d -> %d chars (schema %s, system %s)",
                argv_mod.length(argv_mod.build(call, binary=self.binary)),
                argv_mod.length(argv_mod.build(fitted, binary=self.binary)),
                "moved" if fitted.json_schema is None and call.json_schema else "kept",
                "spilled" if fitted.system_file else ("folded" if call.system else "kept"),
            )
        self.sandbox.verify()
        argv = argv_mod.build(fitted, binary=self.binary)
        try:
            async with self.limit:
                outcome = await run(
                    argv, stdin=fitted.prompt, cwd=self.sandbox.root, timeout=self.timeout
                )
                if not is_mangled_call(outcome.result):
                    return outcome
                # The model reached for a tool channel that is not there. Nothing about the
                # request caused it -- the same request answers cleanly most of the time --
                # so the useful response is to ask again rather than to hand a caller a reply
                # in a shape it cannot read. Once, and only once: a second failure is a fact
                # about this request, and hiding it behind a third attempt would only make it
                # slower to find.
                log.info("retrying: the reply was a tool call with no tool to make it")
                return await run(
                    argv, stdin=fitted.prompt, cwd=self.sandbox.root, timeout=self.timeout
                )
        finally:
            # A spilled prompt lives exactly as long as the process reading it. Leaving them
            # behind would accumulate one copy of the caller's system prompt per request.
            # Done inline rather than on a thread: this is one unlink of one small local file,
            # and a thread per request would cost more than the microsecond it blocks for.
            if fitted.system_file:
                Path(fitted.system_file).unlink(missing_ok=True)  # noqa: ASYNC240


def problem_for(error: Exception) -> str:
    """One sentence a person can act on, for `/ready` and for the first failed call."""
    return str(error)


__all__ = [
    "PROBE_PROMPT",
    "SHOWN",
    "UNPROVEN",
    "Loaded",
    "Runtime",
    "locate",
    "problem_for",
    "tools_from_init",
]

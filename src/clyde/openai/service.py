"""What holds the pieces together for the life of the process.

Three things are decided once at startup rather than per request, because all three are
answers about *this machine* and none of them change between calls:

**Where `claude` is.** Looked up once; a missing binary is reported by `/ready` rather than
discovered on the first real request.

**Which tools to disallow.** Read from the CLI's own `system/init` rather than hard-coded. A
hand-written list silently rots the moment Claude Code ships a new tool, and the cost of
missing one was measured at $0.012 against $0.294 for the same prompt. Learning the list from
the binary that is actually installed is the only version of this that stays true.

**One sandbox.** An empty directory, verified before every call.

Concurrency is a semaphore, not a queue with a worker pool. Each call is a whole Node runtime;
the useful number is small, and making it configurable-but-low is more honest than pretending
this scales horizontally.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from clyde.cli import argv as argv_mod
from clyde.cli import locate
from clyde.cli.run import Outcome, run
from clyde.openai import translate

if TYPE_CHECKING:
    import asyncio

    from clyde.cli.sandbox import Sandbox

PROBE_PROMPT = "ok"
"""The probe still runs a turn, so it is the shortest prompt that can end one."""


def tools_from_init(stream: str) -> tuple[str, ...]:
    """Every tool name `system/init` reported, from the first line of a stream-json run.

    Returns empty on anything unexpected rather than raising: an unreadable probe means the
    harness disallows nothing by name, which is visible in `/ready` and in the first call's
    token count -- where a crash at startup would just mean no service at all.
    """
    for line in stream.splitlines():
        if not line.strip():
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            return ()
        if isinstance(event, dict) and event.get("type") == "system":
            names = event.get("tools")
            if isinstance(names, list):
                return tuple(str(name) for name in names)
        return ()
    return ()


@dataclass
class Runtime:
    """The live service. Built by :func:`start`, held on the app."""

    binary: str
    sandbox: Sandbox
    limit: asyncio.Semaphore
    timeout: float
    default_model: str
    stdin_limit: int
    disallowed: tuple[str, ...] = ()
    problem: str = ""
    """Why this is not usable, when it is not. Empty means ready."""

    models: tuple[str, ...] = field(default=("sonnet", "opus", "haiku", "fable"), init=False)
    """The aliases the CLI accepts. Listed for `GET /v1/models`, which is how a client
    discovers what to ask for and how a catalogue row probes this service."""

    @property
    def ready(self) -> bool:
        return not self.problem

    async def complete(self, body: dict[str, Any]) -> Outcome:
        """One chat-completions body, one process, one outcome."""
        call = translate.to_call(body, disallowed=self.disallowed, default_model=self.default_model)
        size = len(call.prompt.encode("utf-8"))
        if size > self.stdin_limit:
            msg = f"the conversation is {size} bytes, over the {self.stdin_limit}-byte limit"
            raise ValueError(msg)
        self.sandbox.verify()
        async with self.limit:
            return await run(
                argv_mod.build(call, binary=self.binary),
                stdin=call.prompt,
                cwd=self.sandbox.root,
                timeout=self.timeout,
            )


def problem_for(error: Exception) -> str:
    """One sentence a person can act on, for `/ready` and for the first failed call."""
    return str(error)


__all__ = ["PROBE_PROMPT", "Runtime", "locate", "problem_for", "tools_from_init"]

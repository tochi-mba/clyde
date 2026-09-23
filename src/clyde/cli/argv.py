"""One call, turned into one argument list. Pure: no I/O, no environment, no shell.

Everything about *what this process actually runs* is here, in one function, against one table
of tests. That is deliberate. The flags below are the entire security posture of this service,
and a posture spread across three modules is one nobody can check.

The measurements that chose these flags, taken before any of this existed:

    claude -p                                   34,933 input tokens   $0.0758   ~140 tools
    + --system-prompt                           18,983 input tokens   $0.1529   ~140 tools
    + --strict-mcp-config + full disallow list      701 input tokens   $0.0066      0 tools

The middle row is the trap. `--system-prompt` replaces Claude Code's prompt, which looks like
it should be enough, and it halves the context -- but the remaining 19k is the *account's* MCP
connectors: Gmail, Google Drive, Linear, Calendar. They are loaded from the account rather than
from any file here, `--disallowedTools` cannot name them, and a sterile working directory does
not touch them. They are simply offered to the model, next to a prompt full of somebody else's
untrusted tool results.

`--mcp-config '{"mcpServers":{}}'` with `--strict-mcp-config` is what removes them. It is not
an optimisation and it is not optional, which is why :func:`build` takes no flag to skip it and
`test_argv.py` asserts it is present in every argv this module can produce.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

EMPTY_MCP = json.dumps({"mcpServers": {}}, separators=(",", ":"))
"""No MCP servers, stated positively. With `--strict-mcp-config` this replaces the account's
own set rather than adding to it."""

SETTINGS_BLOB = json.dumps(
    {"includeCoAuthoredBy": False, "cleanupPeriodDays": 1},
    separators=(",", ":"),
)
"""Pinned rather than inherited from `~/.claude`, so two machines behave the same."""

PERMISSION_MODE = "dontAsk"
"""Anything that would prompt is denied. Nobody is at the keyboard."""

MAX_TURNS = 1
"""One request, one reply. This is a model, not an agent: the caller runs the loop."""


@dataclass(frozen=True, slots=True)
class Call:
    """One request, in the only terms this module understands.

    Deliberately not an OpenAI request: the `cli` layer does not know that dialect exists, and
    an import contract enforces it. Translating is `openai.translate`'s job.
    """

    prompt: str
    """What goes in on stdin. The conversation, already rendered."""

    system: str = ""
    model: str = ""
    json_schema: Mapping[str, object] | None = None
    disallowed_tools: Sequence[str] = field(default_factory=tuple)
    """Every tool name to remove. Read from `system/init` at startup rather than hard-coded:
    a hand-written list silently rots when Claude Code adds a tool, and the cost of missing
    one is measured above."""


def build(call: Call, *, binary: str) -> list[str]:
    """The argv for one call. `prompt` is **not** in it -- that goes on stdin.

    stdin rather than an argument because a real conversation is tens of kilobytes, Windows
    has a command-line length limit a long turn will cross, and the CLI documents stdin as
    supported in `-p` mode.
    """
    argv = [
        binary,
        "-p",
        "--output-format",
        "json",
        "--max-turns",
        str(MAX_TURNS),
        "--permission-mode",
        PERMISSION_MODE,
        "--permission-prompts",
        "none",
        # The pair that removes the account's connectors. See the module docstring.
        "--mcp-config",
        EMPTY_MCP,
        "--strict-mcp-config",
        "--settings",
        SETTINGS_BLOB,
    ]
    if call.system:
        argv += ["--system-prompt", call.system]
    if call.model:
        argv += ["--model", call.model]
    if call.disallowed_tools:
        argv += ["--disallowedTools", ",".join(call.disallowed_tools)]
    if call.json_schema is not None:
        argv += ["--json-schema", json.dumps(call.json_schema, separators=(",", ":"))]
    return argv


def probe(binary: str) -> list[str]:
    """The argv that asks Claude Code what it loaded, without asking a model anything.

    `system/init` is the first line of a `stream-json` run and reports `tools` and
    `mcp_servers`. That list is what `Call.disallowed_tools` is filled from, so the harness
    learns the tool names from the CLI it is actually running rather than from a constant
    somebody updates by hand.
    """
    return [
        binary,
        "-p",
        "--output-format",
        "stream-json",
        "--verbose",
        "--max-turns",
        str(MAX_TURNS),
        "--permission-mode",
        PERMISSION_MODE,
        "--permission-prompts",
        "none",
        "--settings",
        SETTINGS_BLOB,
    ]


__all__ = ["EMPTY_MCP", "MAX_TURNS", "PERMISSION_MODE", "SETTINGS_BLOB", "Call", "build", "probe"]

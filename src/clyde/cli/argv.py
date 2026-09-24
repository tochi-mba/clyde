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

**2026-09-24: the disallow list leaked.** The last row's "0 tools" came from a list of names: a
startup probe read them out of `system/init` and every call passed them back as
`--disallowedTools`. That stopped being true without anything here changing. The probe ran
with ToolSearch available, which defers some tools out of the init listing; calls disallowed
ToolSearch, so the deferred tools came back. Measured against Claude Code 2.1.280, a real
call's init line read `"tools":["TodoWrite"]` while `/ready` reported 157 names disallowed.
Haiku, asked to build a website, called TodoWrite -- which spends the only turn -- and the run
ended `error_max_turns`, or with an empty reply.

A deny-list can only name what somebody saw, so it is gone rather than kept as a second layer:
it is also what let `/ready` vouch for a lockdown that was not there. `--tools ""` removes the
whole built-in set by construction, and `--disable-slash-commands` removes the dozens of skills
and slash commands the same init line was offering. Replayed with both, the two requests that
had failed each answered in one turn with a plan, and a `--json-schema` call still returned its
structured output -- two turns and `stop_reason: tool_use`, as before, because that channel is
the CLI's own and not one of the tools this removes. The MCP pair stays: `--tools` governs only
the built-in set. All four flags are :data:`LOCKDOWN`, and the startup probe runs under it too,
so what `/ready` reports is what a call loads.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

EMPTY_MCP = json.dumps({"mcpServers": {}}, separators=(",", ":"))
"""No MCP servers, stated positively. With `--strict-mcp-config` this replaces the account's
own set rather than adding to it."""

NO_TOOLS = ""
"""The value of `--tools` that loads none of the built-in set.

`claude --help`: 'Use "" to disable all tools'. An allow-list of nothing, where what it
replaced was a deny-list of whatever a startup probe had seen. A list of names to refuse cannot
refuse a name it never saw; an empty list of names to offer offers nothing, whatever a Claude
Code release adds next.
"""

LOCKDOWN = (
    "--tools",
    NO_TOOLS,
    "--disable-slash-commands",
    # The pair that removes the account's connectors. See the module docstring.
    "--mcp-config",
    EMPTY_MCP,
    "--strict-mcp-config",
)
"""Everything that takes away what a model could reach, as it appears in every argv.

One tuple for :func:`build` and :func:`probe` rather than the same flags written twice, because
the probe exists to show what a call loads, and a probe run under different flags shows
something else. That is exactly how the disallow list came to leak.

The order is not cosmetic. `--tools` and `--mcp-config` each take a list, and a list runs until
the next flag, so each value here is followed by a flag. Anything placed straight after the
empty value would be read as the name of a tool to load.
"""

SETTINGS_BLOB = json.dumps(
    {"includeCoAuthoredBy": False, "cleanupPeriodDays": 1},
    separators=(",", ":"),
)
"""Pinned rather than inherited from `~/.claude`, so two machines behave the same."""

PERMISSION_MODE = "dontAsk"
"""Anything that would prompt is denied. Nobody is at the keyboard."""

MAX_TURNS = 1
"""One request, one reply. This is a model, not an agent: the caller runs the loop."""

STRUCTURED_MAX_TURNS = 4
"""The turns a call with `--json-schema` may spend handing back its structured answer.

That answer travels through the CLI's own structured-output tool, which costs a turn of its
own, and a model that answers in words first is asked again. Measured 2026-09-24 on haiku,
with a request small enough to keep its schema on the command line: allowed one turn, it
ended `error_max_turns after 2 turns` three times in three -- and the caller heard only
"claude answered 502"; allowed three, it answered in three every time. One more is room for
an answer that fails the schema once. Under :data:`LOCKDOWN` there is no other tool to spend
a turn on, so these turns can only be spent reaching the answer: it is still one request and
one reply, and still not an agent.
"""

ARGV_CEILING = 28_000
"""How long a command line :func:`fit` will allow before it moves arguments onto stdin.

Windows caps a command line at 32767 characters, and `CreateProcess` reports the overflow as
`FileNotFoundError` -- errno 2, the same error as a missing binary, saying nothing about
length. That is how this was found: Lucy's plan schema is tens of kilobytes, `--json-schema`
takes a JSON string and no file (checked against the installed CLI), and every plan-shaped
turn died as `clyde failed: FileNotFoundError`.

The ceiling is applied on every platform, not just Windows, so that the degradation is the
same everywhere and a Linux test proves what Windows does. The margin under 32767 covers the
quoting the OS adds around each argument.
"""

SCHEMA_IN_WORDS = (
    "\n\nWhile you still need something you do not have, reply with one JSON object and nothing"
    " else -- no prose around it, no code fence -- conforming to this JSON Schema. When you"
    " have everything you need and the only thing left is to answer, write the answer as"
    " ordinary prose instead, with no JSON in it at all. That is how a reply ends."
    "\n\nWhatever that schema describes, it is not a set of tools you can call. You have no"
    " tools in this conversation. Never emit tool-call syntax -- no <invoke> or"
    " <function_calls> blocks -- and nothing at all outside the JSON object. Naming"
    " something in the JSON is the only way it runs.\n"
)
"""How a schema is asked for when it is too large to pass as a flag.

The last two sentences are the whole file's most load-bearing text, and they are here rather
than in the caller's prompt because they are about *this harness* rather than about that
caller. :data:`LOCKDOWN` makes "You have no tools in this conversation" true by construction,
so a caller reaching Claude Code through a schema like this is using JSON as its tool-call
channel and prose as its answer channel. Callers that reach a model through a vendor SDK never
need saying this: `tool_use` and `text` are different reply shapes there, and the provider
reports which one it sent.

Measured. Without them, the first real caller ever pointed at clyde re-emitted the same plan
eight rounds running, then twelve, and never answered -- because "reply with JSON only" is an
instruction a good model follows, forever. A schema in words is a request rather than a
guarantee, and this is the part of the request that says when to stop making it.

The fence is asked about twice for a reason: told once, the model wrapped its JSON in ```json
anyway, and a caller doing `json.loads` on the whole reply read that as prose.

The paragraph about tool syntax is here for the same reason and is just as specific to this
harness. Claude Code is trained to emit `<invoke name="...">` when it decides to call
something, and it does so even with no tool to call, because a schema full of named
operations reads exactly like a set of tools. Observed: a reply of

    <invoke name="none">
    </invoke>
    {"steps":[{"id":"ls","op":"workspace.list","input":{"path":"."}}]}

which is a perfectly good plan with four lines in front of it, so it parsed as nothing, and a
caller that ends its loop on "not JSON" ended -- showing the person the wire format it had
just failed to read.
"""


@dataclass(frozen=True, slots=True)
class Call:
    """One request, in the only terms this module understands.

    Deliberately not an OpenAI request: the `cli` layer does not know that dialect exists, and
    an import contract enforces it. Translating is `openai.translate`'s job.
    """

    prompt: str
    """What goes in on stdin. The conversation, already rendered."""

    system: str = ""
    system_file: str = ""
    """Where the system prompt was spilled to, when it was too long for the command line.
    Set by :func:`fit`, never by a caller. When it is set, `system` is empty."""

    model: str = ""
    json_schema: Mapping[str, object] | None = None


def length(argv: Sequence[str]) -> int:
    """The command line as the OS will see it: the arguments plus the spaces between them."""
    return sum(len(argument) for argument in argv) + len(argv) - 1


def fit(
    call: Call,
    *,
    binary: str,
    ceiling: int = ARGV_CEILING,
    spill: Callable[[str], str] | None = None,
) -> Call:
    """The same call, rearranged until its argv fits.

    Lucy sends a 22,232-character system prompt and a 30,984-character plan schema. Both on
    one command line is 55,716 characters, and nothing on Windows will run that. So something
    has to move, and what it costs depends entirely on where it moves to.

    **The system prompt goes to a file first**, because `--system-prompt-file` costs nothing
    at all: the model sees the same words in the same role. `spill` writes it and returns the
    path. This is first because it is free, and because it is usually enough on its own.

    **The schema goes into the prompt second.** A schema in words is a request where a schema
    on the command line is a guarantee, but the words still reach the model.

    **The system prompt is folded into the conversation last**, and only when there was no
    `spill` to write it to. This is the expensive one, and it was measured: the turn that
    kept its system prompt planned, ran its step and stopped after two rounds, while the turn
    that folded the same words into stdin re-planned twelve times and died on the iteration
    cap. The role separation is doing real work.

    Returning a `Call` rather than an argv keeps this honest: whatever leaves the command
    line has to arrive somewhere, and this says where.
    """
    if length(build(call, binary=binary)) <= ceiling:
        return call
    if call.system and spill is not None:
        call = replace(call, system="", system_file=spill(call.system))
        if length(build(call, binary=binary)) <= ceiling:
            return call
    if call.json_schema is not None:
        moved = json.dumps(call.json_schema, separators=(",", ":"))
        call = replace(call, prompt=call.prompt + SCHEMA_IN_WORDS + moved, json_schema=None)
        if length(build(call, binary=binary)) <= ceiling:
            return call
    if call.system:
        call = replace(call, prompt=call.system + "\n\n" + call.prompt, system="")
    return call


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
        str(STRUCTURED_MAX_TURNS if call.json_schema is not None else MAX_TURNS),
        "--permission-mode",
        PERMISSION_MODE,
        "--permission-prompts",
        "none",
        *LOCKDOWN,
        "--settings",
        SETTINGS_BLOB,
    ]
    if call.system_file:
        argv += ["--system-prompt-file", call.system_file]
    elif call.system:
        argv += ["--system-prompt", call.system]
    if call.model:
        argv += ["--model", call.model]
    if call.json_schema is not None:
        argv += ["--json-schema", json.dumps(call.json_schema, separators=(",", ":"))]
    return argv


def probe(binary: str) -> list[str]:
    """The argv that asks Claude Code what a call would load, under the flags a call runs with.

    `system/init` is the first event of a `stream-json` run and reports `tools` and
    `mcp_servers`. Under :data:`LOCKDOWN` both are empty -- measured: `"tools":[]` and
    `"mcp_servers":[]` -- and the service checks that they are rather than assuming it.

    It used to run without the lockdown, to learn names for a disallow list, and so it
    reported what *it* loaded rather than what a call did. ToolSearch made those two different
    sets; see the module docstring. A probe is only evidence about the flags it ran with.
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
        *LOCKDOWN,
        "--settings",
        SETTINGS_BLOB,
    ]


__all__ = [
    "ARGV_CEILING",
    "EMPTY_MCP",
    "LOCKDOWN",
    "MAX_TURNS",
    "NO_TOOLS",
    "PERMISSION_MODE",
    "SCHEMA_IN_WORDS",
    "SETTINGS_BLOB",
    "STRUCTURED_MAX_TURNS",
    "Call",
    "build",
    "fit",
    "length",
    "probe",
]

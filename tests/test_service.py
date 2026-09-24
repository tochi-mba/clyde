"""Startup: finding the binary, proving nothing loads, and one call end to end.

`loaded_tools` is the interesting one. It runs the CLI once under the flags every call runs
with and reads back what loaded, so `/ready` reports what a call actually gets rather than what
a differently-configured probe saw -- which is the gap TodoWrite came through. A probe that
could not be read is `None` rather than an empty answer, and a runtime refuses on either. All
of it is tested here with a stand-in `claude` rather than a real one.
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import Any

import pytest

from clyde.api.app import build_runtime, loaded_tools
from clyde.cli import argv as argv_mod
from clyde.cli.locate import ClaudeNotFoundError
from clyde.cli.run import ClaudeFailedError, Outcome
from clyde.cli.sandbox import ContaminatedSandboxError, Sandbox
from clyde.core.config import Settings
from clyde.openai.service import UNPROVEN, Loaded, Runtime, problem_for, tools_from_init

LOCKED_INIT: dict[str, Any] = {
    "type": "system",
    "subtype": "init",
    "tools": [],
    "mcp_servers": [],
    "slash_commands": [],
    "skills": [],
}
"""The first line a probe under the lockdown printed on a real run against 2.1.280, trimmed to
what loaded -- `slash_commands` and `skills` included, which `--disable-slash-commands` emptied."""

LEAKY_INIT: dict[str, Any] = {
    "type": "system",
    "subtype": "init",
    "tools": ["TodoWrite", "Bash"],
    "mcp_servers": [{"name": "claude.ai Gmail", "status": "connected"}],
}
"""What the lockdown exists to prevent: a built-in that got through, and an account connector."""


def stream(*events: dict[str, Any]) -> str:
    """Events as a stream-json run writes them: one object per line."""
    return "".join(json.dumps(event) + "\n" for event in events)


def fake_claude(tmp_path: Path, stdout: str, *, code: int = 0) -> Path:
    script = tmp_path / "fake.py"
    script.write_text(
        f"import sys\nsys.stdin.read()\nsys.stdout.write({stdout!r})\nsys.exit({code})\n",
        encoding="utf-8",
    )
    return script


def runtime(tmp_path: Path, **kwargs: Any) -> Runtime:
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


# --- reading system/init -------------------------------------------------------------------


def test_a_locked_down_init_event_reads_as_nothing_loaded() -> None:
    """The measured line, and the one answer that lets the service serve."""
    loaded = tools_from_init(stream(LOCKED_INIT, {"type": "assistant"}))
    assert loaded is not None
    assert loaded == Loaded()
    assert not loaded.anything


def test_tools_and_mcp_servers_are_read_by_name() -> None:
    assert tools_from_init(stream(LEAKY_INIT)) == Loaded(
        tools=("TodoWrite", "Bash"), mcp_servers=("claude.ai Gmail",)
    )


def test_a_leading_blank_line_is_skipped() -> None:
    assert tools_from_init("\n\n" + stream(LOCKED_INIT)) == Loaded()


def test_events_before_the_init_event_are_passed_over() -> None:
    """Only the init event says what loaded. Reading whatever came first as the answer would
    turn some other event into "nothing loaded" -- or, now, into a refusal to serve."""
    loaded = tools_from_init(stream({"type": "system", "subtype": "hook"}, LEAKY_INIT))
    assert loaded is not None
    assert loaded.anything


def test_an_mcp_server_with_no_name_still_counts() -> None:
    """Dropping it would turn "something loaded" into "nothing did"."""
    loaded = tools_from_init(stream({**LOCKED_INIT, "mcp_servers": [{"status": "connected"}]}))
    assert loaded == Loaded(mcp_servers=("{'status': 'connected'}",))


@pytest.mark.parametrize(
    "text",
    [
        "",
        "not json at all",
        stream({"type": "assistant"}),
        stream({**LOCKED_INIT, "tools": "not a list"}),
        stream({key: value for key, value in LOCKED_INIT.items() if key != "mcp_servers"}),
        stream({**LOCKED_INIT, "mcp_servers": None}),
    ],
    ids=[
        "empty",
        "unparseable",
        "no init event",
        "tools is not a list",
        "no mcp_servers",
        "mcp_servers is not a list",
    ],
)
def test_anything_unexpected_is_no_answer_rather_than_an_empty_one(text: str) -> None:
    """An empty answer would say "nothing loaded", which nothing here has shown. The disallow
    list this replaced read every one of these as an empty list, and served every call."""
    assert tools_from_init(text) is None


# --- what loaded, as a sentence --------------------------------------------------------------


@pytest.mark.parametrize(
    ("loaded", "phrase"),
    [
        (Loaded(tools=("TodoWrite",)), "tools: TodoWrite"),
        (Loaded(mcp_servers=("claude.ai Gmail",)), "MCP servers: claude.ai Gmail"),
        (
            Loaded(tools=("TodoWrite", "Bash"), mcp_servers=("claude.ai Gmail",)),
            "tools: TodoWrite, Bash; MCP servers: claude.ai Gmail",
        ),
        (
            Loaded(tools=tuple(f"Tool{n}" for n in range(8))),
            "tools: Tool0, Tool1, Tool2, Tool3, Tool4 and 3 more",
        ),
    ],
    ids=["a tool", "a server", "both", "more than a sentence should list"],
)
def test_what_loaded_is_named(loaded: Loaded, phrase: str) -> None:
    assert loaded.anything
    assert loaded.describe() == phrase


# --- the startup probe ---------------------------------------------------------------------


async def test_the_probe_reads_a_real_looking_stream(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sandbox = Sandbox.create(tmp_path)
    script = fake_claude(tmp_path, stream(LEAKY_INIT))
    monkeypatch.setattr(
        "clyde.api.app.argv_mod.probe", lambda _binary: [sys.executable, str(script)]
    )
    assert await loaded_tools("ignored", sandbox, timeout=30) == Loaded(
        tools=("TodoWrite", "Bash"), mcp_servers=("claude.ai Gmail",)
    )


async def test_the_probe_that_runs_is_the_locked_down_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A stand-in that loads TodoWrite unless it is handed `--tools ""`. It proves the flag
    reaches the process -- the empty value included, which on Windows has to survive being
    quoted onto a command line -- and not only that `probe` returns it."""
    script = tmp_path / "honest.py"
    script.write_text(
        "import json, sys\n"
        "sys.stdin.read()\n"
        "args = sys.argv[1:]\n"
        "bare = '--tools' in args and args[args.index('--tools') + 1] == ''\n"
        "tools = [] if bare else ['TodoWrite']\n"
        "print(json.dumps({'type': 'system', 'subtype': 'init', 'tools': tools,"
        " 'mcp_servers': []}))\n",
        encoding="utf-8",
    )
    real = argv_mod.probe
    monkeypatch.setattr(
        "clyde.api.app.argv_mod.probe",
        lambda binary: [sys.executable, str(script), *real(binary)[1:]],
    )
    assert await loaded_tools("ignored", Sandbox.create(tmp_path), timeout=30) == Loaded()


async def test_a_probe_that_hangs_is_killed_and_proves_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sandbox = Sandbox.create(tmp_path)
    slow = tmp_path / "slow.py"
    slow.write_text("import time\ntime.sleep(30)\n", encoding="utf-8")
    monkeypatch.setattr("clyde.api.app.argv_mod.probe", lambda _binary: [sys.executable, str(slow)])
    assert await loaded_tools("ignored", sandbox, timeout=0.3) is None


# --- building the runtime -------------------------------------------------------------------


async def test_a_missing_binary_becomes_a_problem_rather_than_a_crash(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A service that refuses to boot gives an operator a crash loop; one that boots and says
    so on `/ready` gives them a sentence."""

    def missing(configured: str) -> str:
        msg = "could not find `claude`"
        raise ClaudeNotFoundError(msg)

    monkeypatch.setattr("clyde.api.app.find", missing)
    live = await build_runtime(Settings())
    assert not live.ready
    assert "could not find" in live.problem
    assert live.binary == ""
    assert live.loaded is None


async def test_a_found_binary_is_probed_for_what_it_loads(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("clyde.api.app.find", lambda _configured: "/usr/local/bin/claude")

    async def nothing(binary: str, sandbox: Sandbox) -> Loaded | None:
        return Loaded()

    monkeypatch.setattr("clyde.api.app.loaded_tools", nothing)
    live = await build_runtime(Settings())
    assert live.ready
    assert live.loaded == Loaded()


# --- refusing to serve what is not locked down -------------------------------------------------


def test_a_runtime_whose_probe_saw_nothing_load_is_ready(tmp_path: Path) -> None:
    live = runtime(tmp_path)
    assert live.unsafe == ""
    assert live.ready


def test_a_runtime_whose_probe_saw_a_tool_load_is_not_ready(tmp_path: Path) -> None:
    """The situation this exists to end: a service that knows a tool is loaded, serving."""
    live = runtime(tmp_path, loaded=Loaded(tools=("TodoWrite",)))
    assert "(tools: TodoWrite)" in live.unsafe
    assert not live.ready


def test_a_runtime_with_no_probe_to_read_is_not_ready(tmp_path: Path) -> None:
    """No proof of a tool is not proof of none."""
    live = runtime(tmp_path, loaded=None)
    assert live.unsafe == UNPROVEN
    assert not live.ready


def test_a_runtime_built_without_a_probe_refuses(tmp_path: Path) -> None:
    """The default is the unproven state, so a runtime built without a probe refuses rather
    than serves."""
    live = Runtime(
        binary="/usr/local/bin/claude",
        sandbox=Sandbox.create(tmp_path),
        limit=asyncio.Semaphore(1),
        timeout=30.0,
        default_model="sonnet",
        stdin_limit=1_000_000,
    )
    assert live.loaded is None
    assert not live.ready


# --- one call through the runtime -------------------------------------------------------------


def read(argv: Any) -> str:
    """The spilled system prompt, read back from where the argv points."""
    return Path(argv[argv.index("--system-prompt-file") + 1]).read_text(encoding="utf-8")


def exists(path: str) -> bool:
    """Synchronous on purpose: an async test asserting about a file is not doing I/O the
    event loop cares about, and the linter is right to ask rather than to guess."""
    return Path(path).exists()


async def test_a_call_runs_in_the_sandbox_with_the_built_argv(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    live = runtime(tmp_path)
    seen: dict[str, Any] = {}

    async def fake_run(
        argv: Any,
        *,
        stdin: str,
        cwd: Any,
        timeout: float,  # noqa: ASYNC109 - mirrors the signature it stands in for
    ) -> Outcome:
        seen.update(argv=list(argv), stdin=stdin, cwd=str(cwd))
        return Outcome(result="ok")

    monkeypatch.setattr("clyde.openai.service.run", fake_run)
    outcome = await live.complete({"messages": [{"role": "user", "content": "hi"}]})
    assert outcome.result == "ok"
    assert seen["stdin"] == "hi"
    assert seen["cwd"] == str(live.sandbox.root)
    assert "--strict-mcp-config" in seen["argv"]
    assert seen["argv"][seen["argv"].index("--tools") + 1] == ""
    assert "--disallowedTools" not in seen["argv"]


async def test_an_oversized_conversation_is_refused_here(tmp_path: Path) -> None:
    """Refused with a sentence rather than by the subprocess with an opaque error."""
    live = runtime(tmp_path, stdin_limit=10)
    with pytest.raises(ValueError, match="over the 10-byte limit"):
        await live.complete({"messages": [{"role": "user", "content": "x" * 100}]})


async def test_a_contaminated_sandbox_stops_the_call(tmp_path: Path) -> None:
    live = runtime(tmp_path)
    (live.sandbox.root / "CLAUDE.md").write_text("inject me", encoding="utf-8")
    with pytest.raises(ContaminatedSandboxError):
        await live.complete({"messages": [{"role": "user", "content": "hi"}]})


def test_problem_for_is_the_sentence_and_nothing_else() -> None:
    assert problem_for(ValueError("just this")) == "just this"


def test_the_model_list_is_the_cli_aliases(tmp_path: Path) -> None:
    assert runtime(tmp_path).models == ("sonnet", "opus", "haiku", "fable")


async def test_an_oversized_call_is_fitted_and_the_move_is_written_down(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """The only symptom of a moved schema downstream is a reply shaped wrong, so the move
    itself has to be visible. Three failures here were diagnosed by guessing before the
    service logged anything at all."""
    live = runtime(tmp_path)
    seen: dict[str, Any] = {}

    async def fake_run(
        argv: Any,
        *,
        stdin: str,
        cwd: Any,
        timeout: float,  # noqa: ASYNC109 - mirrors the signature it stands in for
    ) -> Outcome:
        seen.update(argv=list(argv), stdin=stdin, written=read(argv))
        return Outcome(result="ok")

    monkeypatch.setattr("clyde.openai.service.run", fake_run)
    with caplog.at_level("INFO", logger="clyde"):
        await live.complete(
            {
                "messages": [
                    {"role": "system", "content": "s" * 40_000},
                    {"role": "user", "content": "hi"},
                ]
            }
        )

    assert "--system-prompt" not in seen["argv"]
    assert seen["written"] == "s" * 40_000
    assert seen["stdin"] == "hi", "the conversation stays the conversation"
    assert "system spilled" in caplog.text
    assert "schema kept" in caplog.text


async def test_a_spilled_system_prompt_does_not_outlive_the_call(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One copy of the caller's system prompt per request would otherwise pile up in temp."""
    live = runtime(tmp_path)
    seen: dict[str, Any] = {}

    async def fake_run(
        argv: Any,
        *,
        stdin: str,
        cwd: Any,
        timeout: float,  # noqa: ASYNC109 - mirrors the signature it stands in for
    ) -> Outcome:
        seen["path"] = argv[argv.index("--system-prompt-file") + 1]
        assert exists(seen["path"]), "it must exist while the process reads it"
        msg = "boom"
        raise ClaudeFailedError(msg)

    monkeypatch.setattr("clyde.openai.service.run", fake_run)
    with pytest.raises(ClaudeFailedError):
        await live.complete(
            {
                "messages": [
                    {"role": "system", "content": "s" * 40_000},
                    {"role": "user", "content": "hi"},
                ]
            }
        )
    assert not exists(seen["path"])


async def test_a_call_that_already_fits_is_not_logged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """A line per call would bury the one that matters."""
    live = runtime(tmp_path)

    async def fake_run(
        argv: Any,
        *,
        stdin: str,
        cwd: Any,
        timeout: float,  # noqa: ASYNC109 - mirrors the signature it stands in for
    ) -> Outcome:
        return Outcome(result="ok")

    monkeypatch.setattr("clyde.openai.service.run", fake_run)
    with caplog.at_level("INFO", logger="clyde"):
        await live.complete({"messages": [{"role": "user", "content": "hi"}]})
    assert "fitted" not in caplog.text


# --- a reply that is a tool call with no tool to make it ---------------------------------------
#
# Measured at roughly one turn in three against a real caller: the model reaches for a channel
# that is not there, and what it meant to say arrives in a shape no caller can read. Two of the
# three observed shapes are only tags wrapping the payload and are recovered by
# `without_tool_call_tags`; this is the third, where the syntax is tangled through the text.

TANGLED = (
    '<parameter name="op">notes.search</parameter>\n'
    "</invoke>\n"
    "```\n"
    "Wait, correcting format: here is the JSON object.\n"
    '{"steps":[{"id":"mem","op":"notes.search"}]}'
)


def replies(*texts: str) -> tuple[Any, list[int]]:
    """A stand-in for `run` that hands back each text in turn, and counts the calls."""
    calls: list[int] = []

    async def fake_run(
        argv: Any,
        *,
        stdin: str,
        cwd: Any,
        timeout: float,  # noqa: ASYNC109 - mirrors the signature it stands in for
    ) -> Outcome:
        calls.append(1)
        return Outcome(result=texts[min(len(calls) - 1, len(texts) - 1)])

    return fake_run, calls


async def test_a_mangled_call_is_asked_again(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    live = runtime(tmp_path)
    fake_run, calls = replies(TANGLED, '{"steps":[{"id":"mem","op":"notes.search"}]}')
    monkeypatch.setattr("clyde.openai.service.run", fake_run)
    with caplog.at_level("INFO", logger="clyde"):
        outcome = await live.complete({"messages": [{"role": "user", "content": "hi"}]})
    assert len(calls) == 2
    assert outcome.result == '{"steps":[{"id":"mem","op":"notes.search"}]}'
    assert "retrying" in caplog.text


async def test_a_clean_reply_is_not_asked_again(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    live = runtime(tmp_path)
    fake_run, calls = replies("a perfectly ordinary answer")
    monkeypatch.setattr("clyde.openai.service.run", fake_run)
    outcome = await live.complete({"messages": [{"role": "user", "content": "hi"}]})
    assert len(calls) == 1
    assert outcome.result == "a perfectly ordinary answer"


async def test_it_is_asked_again_once_and_not_twice(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A second failure is a fact about this request, and hiding it behind a third attempt
    would only make it slower to find."""
    live = runtime(tmp_path)
    fake_run, calls = replies(TANGLED, TANGLED)
    monkeypatch.setattr("clyde.openai.service.run", fake_run)
    outcome = await live.complete({"messages": [{"role": "user", "content": "hi"}]})
    assert len(calls) == 2
    assert "correcting format" in outcome.result, "the second reply is returned as it came"

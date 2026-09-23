"""Startup: finding the binary, learning the tool list, and one call end to end.

`learn_tools` is the interesting one. It exists so the disallow list comes from the CLI that
is actually installed rather than from a constant somebody updates by hand -- and it is
allowed to fail, because an empty list costs tokens rather than correctness. Both halves are
tested here, with a stand-in `claude` rather than a real one.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest

from clyde.api.app import build_runtime, learn_tools
from clyde.cli.locate import ClaudeNotFoundError
from clyde.cli.run import ClaudeFailedError, Outcome
from clyde.cli.sandbox import ContaminatedSandboxError, Sandbox
from clyde.core.config import Settings
from clyde.openai.service import Runtime, problem_for, tools_from_init

INIT_LINE: dict[str, Any] = {
    "type": "system",
    "subtype": "init",
    "tools": ["Bash", "Read", "mcp__claude_ai_Gmail__send_message"],
    "mcp_servers": [{"name": "claude.ai Gmail", "status": "connected"}],
}


def fake_claude(tmp_path: Path, stdout: str, *, code: int = 0) -> Path:
    script = tmp_path / "fake.py"
    script.write_text(
        f"import sys\nsys.stdin.read()\nsys.stdout.write({stdout!r})\nsys.exit({code})\n",
        encoding="utf-8",
    )
    return script


# --- reading system/init -------------------------------------------------------------------


def test_the_tool_names_are_read_from_the_first_event() -> None:
    names = tools_from_init(json.dumps(INIT_LINE) + "\n" + json.dumps({"type": "assistant"}))
    assert names == ("Bash", "Read", "mcp__claude_ai_Gmail__send_message")


def test_a_leading_blank_line_is_skipped() -> None:
    assert tools_from_init("\n\n" + json.dumps(INIT_LINE)) == tuple(INIT_LINE["tools"])


@pytest.mark.parametrize(
    "stream",
    [
        "",
        "not json at all",
        json.dumps({"type": "assistant"}),
        json.dumps({"type": "system", "tools": "not a list"}),
    ],
    ids=["empty", "unparseable", "wrong event first", "tools is not a list"],
)
def test_anything_unexpected_reads_as_no_tools(stream: str) -> None:
    """Empty means "disallow nothing by name", which is visible in `/ready` and in the first
    call's token count. Raising here would mean no service at all."""
    assert tools_from_init(stream) == ()


# --- the startup probe ---------------------------------------------------------------------


async def test_the_probe_reads_a_real_looking_stream(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sandbox = Sandbox.create(tmp_path)
    script = fake_claude(tmp_path, json.dumps(INIT_LINE) + "\n")
    monkeypatch.setattr(
        "clyde.api.app.argv_mod.probe", lambda _binary: [sys.executable, str(script)]
    )
    assert await learn_tools("ignored", sandbox, timeout=30) == tuple(INIT_LINE["tools"])


async def test_a_probe_that_hangs_is_killed_and_yields_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sandbox = Sandbox.create(tmp_path)
    slow = tmp_path / "slow.py"
    slow.write_text("import time\ntime.sleep(30)\n", encoding="utf-8")
    monkeypatch.setattr("clyde.api.app.argv_mod.probe", lambda _binary: [sys.executable, str(slow)])
    assert await learn_tools("ignored", sandbox, timeout=0.3) == ()


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


async def test_a_found_binary_is_probed_for_its_tools(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("clyde.api.app.find", lambda _configured: "/usr/local/bin/claude")

    async def learned(binary: str, sandbox: Sandbox) -> tuple[str, ...]:
        return ("Bash", "Read")

    monkeypatch.setattr("clyde.api.app.learn_tools", learned)
    live = await build_runtime(Settings())
    assert live.ready
    assert live.disallowed == ("Bash", "Read")


# --- one call through the runtime -------------------------------------------------------------


def read(argv: Any) -> str:
    """The spilled system prompt, read back from where the argv points."""
    return Path(argv[argv.index("--system-prompt-file") + 1]).read_text(encoding="utf-8")


def exists(path: str) -> bool:
    """Synchronous on purpose: an async test asserting about a file is not doing I/O the
    event loop cares about, and the linter is right to ask rather than to guess."""
    return Path(path).exists()


def runtime(tmp_path: Path, **kwargs: Any) -> Runtime:
    import asyncio

    base: dict[str, Any] = {
        "binary": "/usr/local/bin/claude",
        "sandbox": Sandbox.create(tmp_path),
        "limit": asyncio.Semaphore(1),
        "timeout": 30.0,
        "default_model": "sonnet",
        "stdin_limit": 1_000_000,
        "disallowed": ("Bash",),
    }
    base.update(kwargs)
    return Runtime(**base)


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
    assert "Bash" in seen["argv"][seen["argv"].index("--disallowedTools") + 1]


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

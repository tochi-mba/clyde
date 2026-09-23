"""Spawning, parsing, and every way a run can end badly.

A real `claude` is never spawned here. A tiny Python script stands in for it, which is enough
because this module only cares about exit codes, stdout and stderr -- and a fake lets a test
produce a timeout or a SIGTERM exit on demand, which a real one does not.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from clyde.cli.run import (
    SIGTERM_EXIT,
    STDERR_KEPT,
    ClaudeFailedError,
    ClaudeTimeoutError,
    Outcome,
    first_line,
    parse,
    run,
)

CLI_JSON = {
    "result": "ok",
    "is_error": False,
    "stop_reason": "end_turn",
    "num_turns": 1,
    "subtype": "success",
    "session_id": "c298581a",
    "total_cost_usd": 0.0066,
    "usage": {
        "input_tokens": 2,
        "cache_read_input_tokens": 0,
        "cache_creation_input_tokens": 699,
        "output_tokens": 4,
    },
    "modelUsage": {"claude-opus-5-5": {}},
}
"""Shaped on a real run, so a field rename upstream shows up here rather than in production."""


def fake(tmp_path: Path, body: str) -> list[str]:
    """A stand-in `claude`: a Python script run by this interpreter."""
    script = tmp_path / "fake_claude.py"
    script.write_text(body, encoding="utf-8")
    return [sys.executable, str(script)]


def prints(payload: str, *, code: int = 0, stderr: str = "") -> str:
    """A stand-in that writes exactly what it is given, then exits how it is told."""
    return (
        "import sys\n"
        "sys.stdin.read()\n"
        f"sys.stdout.write({payload!r})\n"
        f"sys.stderr.write({stderr!r})\n"
        f"sys.exit({code})\n"
    )


# --- parsing ---------------------------------------------------------------------------------


def test_a_real_looking_payload_parses() -> None:
    outcome = parse(json.dumps(CLI_JSON))
    assert outcome.result == "ok"
    assert outcome.is_error is False
    assert outcome.session_id == "c298581a"
    assert outcome.cost_usd == pytest.approx(0.0066)
    assert outcome.model == "claude-opus-5-5"
    assert outcome.usage["cache_creation_input_tokens"] == 699


def test_an_explicit_model_beats_the_usage_map() -> None:
    outcome = parse(json.dumps({**CLI_JSON, "model": "claude-sonnet-5"}))
    assert outcome.model == "claude-sonnet-5"


def test_a_missing_field_takes_its_default_rather_than_crashing() -> None:
    outcome = parse("{}")
    assert (outcome.result, outcome.stop_reason, outcome.num_turns) == ("", "end_turn", 1)
    assert outcome.usage == {}


def test_a_usage_field_that_is_not_an_object_is_dropped() -> None:
    assert parse(json.dumps({"usage": "nonsense"})).usage == {}


def test_unparseable_output_is_a_failure_not_an_empty_reply() -> None:
    """A caller that cannot tell these apart will treat one as the other, once, in production."""
    with pytest.raises(ClaudeFailedError, match="did not return JSON"):
        parse("Killed\n")


def test_json_that_is_not_an_object_is_a_failure() -> None:
    with pytest.raises(ClaudeFailedError, match="returned list"):
        parse("[1, 2]")


@pytest.mark.parametrize(
    ("payload", "expected"),
    [
        ({"subtype": "error_max_turns"}, True),
        ({}, False),
        ({"subtype": "success"}, False),
        # Measured against the real CLI: a `--json-schema` call reports two turns and
        # `tool_use` on a completely successful run, because the JSON is emitted through an
        # internal tool. Reading either as truncation tells a caller every plan ran out of
        # room, and a caller that believes it offers to resume a finished turn.
        ({"num_turns": 2, "stop_reason": "tool_use", "subtype": "success"}, False),
    ],
    ids=["ran out of turns", "nothing said", "said success", "structured output"],
)
def test_only_running_out_of_turns_counts_as_the_limit(
    payload: dict[str, object], expected: bool
) -> None:
    assert parse(json.dumps(payload)).hit_turn_limit is expected


# --- stderr ----------------------------------------------------------------------------------


def test_only_the_first_line_of_stderr_survives() -> None:
    assert first_line("first thing\nsecond thing\n") == "first thing"


def test_leading_blank_lines_are_skipped() -> None:
    assert first_line("\n\n  the real message  \n") == "the real message"


def test_empty_stderr_is_empty() -> None:
    assert first_line("") == ""


def test_stderr_is_capped() -> None:
    assert len(first_line("x" * 5000)) == STDERR_KEPT


# --- spawning --------------------------------------------------------------------------------


async def test_a_successful_run_returns_the_outcome(tmp_path: Path) -> None:
    argv = fake(tmp_path, prints(json.dumps(CLI_JSON)))
    outcome = await run(argv, stdin="hello", cwd=tmp_path, timeout=30)
    assert outcome.result == "ok"


async def test_stdin_reaches_the_process(tmp_path: Path) -> None:
    script = tmp_path / "echo.py"
    script.write_text(
        "import sys, json\n"
        "data = sys.stdin.read()\n"
        "sys.stdout.write(json.dumps({'result': data}))\n",
        encoding="utf-8",
    )
    outcome = await run(
        [sys.executable, str(script)], stdin="the conversation", cwd=tmp_path, timeout=30
    )
    assert outcome.result == "the conversation"


async def test_a_non_zero_exit_names_the_first_stderr_line(tmp_path: Path) -> None:
    argv = fake(tmp_path, prints("", code=2, stderr="something broke\nstack frame\n"))
    with pytest.raises(ClaudeFailedError, match="exited 2: something broke") as raised:
        await run(argv, stdin="", cwd=tmp_path, timeout=30)
    assert raised.value.exit_code == 2


async def test_a_secret_on_stderr_does_not_reach_the_message(tmp_path: Path) -> None:
    """stderr routinely carries paths and prompt fragments; only the first line escapes, and
    a test pins that rather than trusting it."""
    secret = "sk-ant-THIS-MUST-NOT-ESCAPE"
    argv = fake(tmp_path, prints("", code=2, stderr=f"a harmless first line\n{secret}\n"))
    with pytest.raises(ClaudeFailedError) as raised:
        await run(argv, stdin="", cwd=tmp_path, timeout=30)
    assert secret not in str(raised.value)


async def test_a_non_zero_exit_with_silent_stderr_still_says_something(tmp_path: Path) -> None:
    argv = fake(tmp_path, prints("", code=3))
    with pytest.raises(ClaudeFailedError, match="no detail on stderr"):
        await run(argv, stdin="", cwd=tmp_path, timeout=30)


async def test_sigterm_is_its_own_failure(tmp_path: Path) -> None:
    """Exit 143 means somebody stopped it, which is not the same as the model failing."""
    argv = fake(tmp_path, prints("", code=SIGTERM_EXIT))
    with pytest.raises(ClaudeFailedError, match="terminated before it finished"):
        await run(argv, stdin="", cwd=tmp_path, timeout=30)


async def test_a_timeout_kills_the_child_and_raises(tmp_path: Path) -> None:
    """The kill matters: returning without it leaves a Node process holding a model call."""
    script = tmp_path / "slow.py"
    script.write_text("import time\ntime.sleep(30)\n", encoding="utf-8")
    with pytest.raises(ClaudeTimeoutError, match="did not finish within"):
        await run([sys.executable, str(script)], stdin="", cwd=tmp_path, timeout=0.3)


def test_the_outcome_defaults_are_the_quiet_ones() -> None:
    outcome = Outcome(result="hi")
    assert (outcome.is_error, outcome.num_turns, outcome.cost_usd) == (False, 1, 0.0)

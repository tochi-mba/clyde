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
    without_tool_call_tags,
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


# --- saying why a run stopped ----------------------------------------------------------------
#
# The CLI's error results carry no `result` text. Before `Outcome.why`, a model at its usage
# limit, a run that ran out of turns and a model that reached for a tool all reached a caller
# as the same four words: "claude reported an error".

STOPPED_JSON = {
    "type": "result",
    "subtype": "error_max_turns",
    "is_error": True,
    "num_turns": 2,
    "stop_reason": "tool_use",
    "session_id": "5b0c7e21",
    "total_cost_usd": 0.0121,
    "usage": {"input_tokens": 9, "output_tokens": 266},
    "modelUsage": {"claude-haiku-4-5-20251001": {}},
    "permission_denials": [
        {
            "tool_name": "Write",
            "tool_use_id": "toolu_01",
            "tool_input": {"file_path": "index.html", "content": "<!doctype html>"},
        }
    ],
    "terminal_reason": "max_turns",
}
"""A failed run in the shape the CLI writes one: no `result` at all, `terminal_reason` beside
`subtype`, and each refused tool call in `permission_denials` with the input it was given."""


def test_a_failed_run_keeps_what_the_model_reached_for_and_why_it_stopped() -> None:
    outcome = parse(json.dumps(STOPPED_JSON))
    assert outcome.result == ""
    assert outcome.denied == ("Write",)
    assert outcome.terminal_reason == "max_turns"


def test_a_run_that_says_nothing_about_either_keeps_nothing() -> None:
    outcome = parse(json.dumps(CLI_JSON))
    assert (outcome.denied, outcome.terminal_reason) == ((), "")


@pytest.mark.parametrize(
    ("denials", "expected"),
    [
        ({"tool_name": "Write"}, ()),
        (["Write", 3, None], ()),
        ([{"tool_use_id": "toolu_01"}], ()),
        ([{"tool_name": ""}], ()),
        ([{"tool_name": 7}], ()),
        (
            [{"tool_name": "Write"}, {"tool_name": "Bash"}, {"tool_name": "Write"}],
            ("Write", "Bash"),
        ),
    ],
    ids=[
        "not a list",
        "entries that are not objects",
        "an entry with no name",
        "a blank name",
        "a name that is not a string",
        "a tool refused twice",
    ],
)
def test_each_refused_tool_is_named_once_in_the_order_it_was_reached_for(
    denials: object, expected: tuple[str, ...]
) -> None:
    """Anything that is not a tool's name is dropped rather than guessed at: a sentence naming
    `3` or `None` as the tool a model reached for would send somebody looking for it."""
    assert parse(json.dumps({"permission_denials": denials})).denied == expected


def test_why_is_the_sentence_a_real_run_that_ran_out_of_turns_produced() -> None:
    """Word for word what a real Haiku run reported once this existed."""
    outcome = Outcome(
        result="",
        is_error=True,
        subtype="error_max_turns",
        num_turns=2,
        terminal_reason="max_turns",
    )
    assert outcome.why == "claude stopped with error_max_turns after 2 turns (max_turns)"


def test_a_single_turn_with_no_terminal_reason_is_said_plainly() -> None:
    outcome = Outcome(result="", is_error=True, subtype="error_during_execution")
    assert outcome.why == "claude stopped with error_during_execution after 1 turn"


def test_a_run_that_completed_does_not_say_so() -> None:
    """`completed` is how every run ends that was not cut short, so it explains no failure."""
    outcome = Outcome(
        result="", is_error=True, subtype="error_during_execution", terminal_reason="completed"
    )
    assert outcome.why == "claude stopped with error_during_execution after 1 turn"


def test_why_names_every_tool_the_model_reached_for() -> None:
    """The one fact that says what to change: the model tried to *do* the thing rather than
    answer."""
    denials = [{"tool_name": "Write"}, {"tool_name": "Bash"}]
    outcome = parse(json.dumps({**STOPPED_JSON, "permission_denials": denials}))
    assert outcome.why == (
        "claude stopped with error_max_turns after 2 turns (max_turns); it tried to use Write, "
        "Bash, which this service does not allow -- the model reached for a tool instead of "
        "answering"
    )


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


async def test_a_non_zero_exit_with_nothing_anywhere_still_says_something(tmp_path: Path) -> None:
    argv = fake(tmp_path, prints("", code=3))
    with pytest.raises(ClaudeFailedError, match="nothing on stderr or stdout"):
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


# --- leaked tool-call syntax ------------------------------------------------------------------
#
# Claude Code emits `<invoke name="...">` when it decides to call something, and does so with
# no tool to call, because a prompt full of named operations reads exactly like a set of tools.
# With no tool to match, the CLI hands the tags through as part of the reply. Both shapes below
# came off a real run, a few turns apart.

IN_FRONT = '<invoke name="none">\n</invoke>\n{"steps":[{"id":"ls","op":"workspace.list"}]}'
WRAPPED = '<invoke>\n{"steps":[{"id":"mem","op":"notes.search"}]}\n</invoke>'


def test_tags_in_front_of_the_reply_come_off() -> None:
    assert without_tool_call_tags(IN_FRONT) == '{"steps":[{"id":"ls","op":"workspace.list"}]}'


def test_tags_wrapped_around_the_reply_come_off_without_taking_it_with_them() -> None:
    """The shape that broke the first version of this. Matching `<invoke>...</invoke>` as a
    block and deleting it was right for `IN_FRONT` and threw the whole reply away here."""
    assert without_tool_call_tags(WRAPPED) == '{"steps":[{"id":"mem","op":"notes.search"}]}'


def test_a_nested_wrapper_comes_off() -> None:
    text = '<function_calls>\n<invoke name="x">\n</invoke>\n</function_calls>\nhello'
    assert without_tool_call_tags(text) == "hello"


def test_blank_lines_between_the_tags_and_the_reply_do_not_stop_it() -> None:
    assert without_tool_call_tags('<invoke>\n\n{"a": 1}\n\n</invoke>\n') == '{"a": 1}'


def test_a_namespaced_tag_comes_off() -> None:
    assert without_tool_call_tags("<ns:invoke>\nafter\n</ns:invoke>") == "after"


def test_a_tag_inside_a_sentence_is_the_model_writing_about_it() -> None:
    """Somebody asking clyde to explain Claude Code gets exactly this, and editing it would be
    this harness rewriting an answer it was not asked to rewrite."""
    text = 'Here is the syntax: <invoke name="x"></invoke> use it like that.'
    assert without_tool_call_tags(text) == text


def test_a_tag_in_the_middle_is_left_where_it_is() -> None:
    """Only the edges. A tag line in the middle is holding two halves of something apart, and
    nothing here knows what."""
    assert without_tool_call_tags("one\n<invoke>\ntwo") == "one\n<invoke>\ntwo"


def test_a_reply_that_is_only_tags_is_kept_as_it_is() -> None:
    """An empty reply is a worse answer than a strange one, and downstream it cannot be told
    apart from the model saying nothing at all."""
    assert without_tool_call_tags("<invoke>\n</invoke>") == "<invoke>\n</invoke>"


def test_ordinary_prose_is_untouched() -> None:
    assert without_tool_call_tags("plain prose answer") == "plain prose answer"


def test_parse_strips_before_the_outcome_is_built() -> None:
    """Every caller reads `Outcome.result`, so the leak is undone once rather than in each of
    the two dialect paths that would otherwise each have to remember."""
    outcome = parse(json.dumps({"result": WRAPPED, "subtype": "success"}))
    assert outcome.result == '{"steps":[{"id":"mem","op":"notes.search"}]}'


async def test_a_failure_with_a_silent_stderr_reads_stdout(tmp_path: Path) -> None:
    """The CLI does not always put the reason on stderr. One real failure arrived with stderr
    empty and everything worth reading on stdout, and clyde reported `no detail on stderr` --
    a sentence about clyde rather than about what went wrong."""
    argv = fake(tmp_path, prints("the real reason\nmore\n", code=1))
    with pytest.raises(ClaudeFailedError, match="the real reason"):
        await run(argv, stdin="", cwd=tmp_path, timeout=30)


async def test_stderr_still_wins_when_there_is_something_on_it(tmp_path: Path) -> None:
    argv = fake(tmp_path, prints("ignore me\n", code=1, stderr="the actual error\n"))
    with pytest.raises(ClaudeFailedError, match="the actual error"):
        await run(argv, stdin="", cwd=tmp_path, timeout=30)


async def test_a_parseable_stdout_wins_over_a_non_zero_exit(tmp_path: Path) -> None:
    """The CLI has exited 1 while writing a complete `--output-format json` object. That object
    says what happened in its own words; the exit code only says "something". Discarding it
    turned a reportable outcome into `claude exited 1: no detail on stderr`."""
    payload = json.dumps(
        {"result": "here is the answer", "subtype": "success", "stop_reason": "stop_sequence"}
    )
    outcome = await run(fake(tmp_path, prints(payload, code=1)), stdin="", cwd=tmp_path, timeout=30)
    assert outcome.result == "here is the answer"
    assert outcome.is_error is False


async def test_an_error_the_cli_declares_survives_the_same_path(tmp_path: Path) -> None:
    """`is_error` is the CLI's own verdict and reaches the caller as one, rather than being
    flattened into the exit code."""
    payload = json.dumps({"result": "the model refused", "is_error": True, "subtype": "error_x"})
    outcome = await run(fake(tmp_path, prints(payload, code=1)), stdin="", cwd=tmp_path, timeout=30)
    assert outcome.is_error is True
    assert outcome.subtype == "error_x"


async def test_an_unparseable_stdout_falls_back_to_the_sentence(tmp_path: Path) -> None:
    argv = fake(tmp_path, prints("not json at all", code=1, stderr="the actual error"))
    with pytest.raises(ClaudeFailedError, match="the actual error"):
        await run(argv, stdin="", cwd=tmp_path, timeout=30)

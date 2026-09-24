"""The argv is the security posture, so it is asserted flag by flag.

The tests that matter most are the lockdown ones, and they assert the literal flags rather than
`LOCKDOWN`, so that deleting a flag from the constant fails them instead of passing with it.
Without the MCP pair the account's own connectors load: measured at 34,933 input tokens against
701, and about 140 tools offered against none. Without `--tools ""` the built-in set comes
back, and a model that calls one -- TodoWrite, measured -- spends its only turn and answers
nothing. No later test would catch either, because the service would still mostly answer.
"""

from __future__ import annotations

import json
from collections.abc import Callable

import pytest

from clyde.cli.argv import (
    ARGV_CEILING,
    EMPTY_MCP,
    MAX_TURNS,
    NO_TOOLS,
    PERMISSION_MODE,
    SCHEMA_IN_WORDS,
    STRUCTURED_MAX_TURNS,
    Call,
    build,
    fit,
    length,
    probe,
)

BINARY = "/usr/local/bin/claude"


def pair(argv: list[str], flag: str) -> str:
    """The value that follows a flag, so a test reads as a claim rather than an index."""
    return argv[argv.index(flag) + 1]


EVERY_SHAPE = [
    Call(prompt="hi"),
    Call(prompt="hi", system="s", model="opus"),
    Call(prompt="hi", system_file="/var/spool/clyde-test/spilled-1.txt"),
    Call(prompt="hi", json_schema={"type": "object"}),
    Call(prompt="hi", system="s", model="opus", json_schema={}),
]
"""One call down each path through `build`: bare, with a system prompt and a model, with a
spilled system prompt, with a schema, and with everything at once."""

EVERY_ARGV = pytest.mark.parametrize(
    "argv",
    [build(call, binary=BINARY) for call in EVERY_SHAPE] + [probe(BINARY)],
    ids=["bare", "system and model", "spilled system", "schema", "everything", "probe"],
)
"""Every argv this module can produce, the probe included: what the probe proves is only worth
anything if it ran under the same lockdown as the calls."""


@EVERY_ARGV
def test_the_mcp_lockdown_is_in_every_argv(argv: list[str]) -> None:
    """The pair that removes the account's connectors. Absent, everything else still works."""
    assert "--strict-mcp-config" in argv
    assert pair(argv, "--mcp-config") == EMPTY_MCP
    assert json.loads(EMPTY_MCP) == {"mcpServers": {}}


@EVERY_ARGV
def test_every_built_in_tool_is_removed_from_every_argv(argv: list[str]) -> None:
    """`--tools ""` is an allow-list of nothing. The deny-list it replaced could only name what a
    probe had seen, and TodoWrite got past it into real calls."""
    assert pair(argv, "--tools") == ""
    assert "--disable-slash-commands" in argv


@EVERY_ARGV
def test_no_argv_names_a_tool_to_deny(argv: list[str]) -> None:
    """Gone rather than kept as a second layer: a deny-list is what let `/ready` vouch for 157
    names disallowed while TodoWrite loaded."""
    assert "--disallowedTools" not in argv
    assert "--disallowed-tools" not in argv


@EVERY_ARGV
def test_each_list_valued_flag_is_followed_by_another_flag(argv: list[str]) -> None:
    """`--tools` and `--mcp-config` each take a list, which runs until the next flag. Anything
    that came straight after the empty value would be read as the name of a tool to load."""
    for flag in ("--tools", "--mcp-config"):
        assert argv[argv.index(flag) + 2].startswith("--")


def test_no_tools_is_the_empty_string_the_cli_documents() -> None:
    """`claude --help`: 'Use "" to disable all tools'. `default` would load every one."""
    assert NO_TOOLS == ""


def test_the_prompt_never_reaches_the_argument_list() -> None:
    """It goes on stdin: a real conversation crosses the Windows command-line limit."""
    argv = build(Call(prompt="a distinctive sentence"), binary=BINARY)
    assert "a distinctive sentence" not in " ".join(argv)


def test_the_fixed_flags_are_all_present() -> None:
    argv = build(Call(prompt="hi"), binary=BINARY)
    assert argv[0] == BINARY
    assert "-p" in argv
    assert pair(argv, "--output-format") == "json"
    assert pair(argv, "--max-turns") == str(MAX_TURNS)
    assert pair(argv, "--permission-mode") == PERMISSION_MODE
    assert pair(argv, "--permission-prompts") == "none"
    assert "--settings" in argv


def test_one_turn_only() -> None:
    """More than one turn would make this an agent; the caller owns the loop."""
    assert MAX_TURNS == 1


def test_a_structured_call_has_the_turns_its_answer_takes() -> None:
    """The bug, named: with one turn allowed, haiku answered a `--json-schema` call in words,
    was asked again, and the run ended `error_max_turns after 2 turns` -- every time."""
    argv = build(Call(prompt="hi", json_schema={"type": "object"}), binary=BINARY)
    assert pair(argv, "--max-turns") == str(STRUCTURED_MAX_TURNS)
    assert STRUCTURED_MAX_TURNS >= 3, "measured: haiku needs three"


def test_a_schema_moved_into_the_prompt_takes_the_one_turn_again() -> None:
    """In words, the schema is a request the model answers in its reply: no extra round."""
    schema = {"type": "object", "properties": {f"k{n}": {"type": "string"} for n in range(9)}}
    fitted = fit(Call(prompt="hi", json_schema=schema), binary=BINARY, ceiling=200)
    assert fitted.json_schema is None
    assert pair(build(fitted, binary=BINARY), "--max-turns") == str(MAX_TURNS)


def test_the_system_prompt_is_passed_verbatim() -> None:
    """Replacing rather than appending is what makes the caller's prompt the only prompt."""
    argv = build(Call(prompt="hi", system="You are terse."), binary=BINARY)
    assert pair(argv, "--system-prompt") == "You are terse."


def test_no_system_prompt_flag_when_there_is_no_system_message() -> None:
    assert "--system-prompt" not in build(Call(prompt="hi"), binary=BINARY)


def test_the_model_is_passed_when_named_and_omitted_when_not() -> None:
    assert pair(build(Call(prompt="hi", model="opus"), binary=BINARY), "--model") == "opus"
    assert "--model" not in build(Call(prompt="hi"), binary=BINARY)


def test_a_json_schema_is_passed_as_compact_json() -> None:
    schema = {"type": "object", "properties": {"a": {"type": "string"}}}
    argv = build(Call(prompt="hi", json_schema=schema), binary=BINARY)
    assert json.loads(pair(argv, "--json-schema")) == schema
    assert " " not in pair(argv, "--json-schema")


def test_no_schema_flag_when_no_schema_was_asked_for() -> None:
    assert "--json-schema" not in build(Call(prompt="hi"), binary=BINARY)


def test_an_empty_schema_is_still_a_schema() -> None:
    """`{}` is falsy and means "any object"; `None` means "no structured output"."""
    assert "--json-schema" in build(Call(prompt="hi", json_schema={}), binary=BINARY)


@pytest.mark.parametrize("flag", ["--bare", "--resume", "--continue", "--allowedTools"])
def test_flags_that_must_never_appear(flag: str) -> None:
    """`--bare` would drop subscription auth; the session flags would give the CLI a second
    conversation state the caller cannot see."""
    argv = build(Call(prompt="hi", system="s", model="opus", json_schema={}), binary=BINARY)
    assert flag not in argv


def test_the_probe_streams_so_that_what_loaded_is_the_first_thing_it_says() -> None:
    argv = probe(BINARY)
    assert pair(argv, "--output-format") == "stream-json"
    assert "--verbose" in argv
    assert "--json-schema" not in argv


def test_the_probe_runs_under_the_same_lockdown_as_a_call() -> None:
    """It used to run with no lockdown at all, to see what *would* load and deny it by name.
    What it saw was what a probe loads, which ToolSearch made different from what a call
    loads. `EVERY_ARGV` covers the flags one by one; this is the whole run of them, unbroken
    and in order, in both -- so the probe cannot drift from `build` by one flag either."""
    lockdown = [
        "--tools",
        "",
        "--disable-slash-commands",
        "--mcp-config",
        EMPTY_MCP,
        "--strict-mcp-config",
    ]
    for argv in (build(Call(prompt="hi"), binary=BINARY), probe(BINARY)):
        start = argv.index("--tools")
        assert argv[start : start + len(lockdown)] == lockdown


# --- fitting the command line ---------------------------------------------------------
#
# Every one of these is the same bug: Windows caps a command line at 32767 characters and
# reports the overflow as `FileNotFoundError`, so an oversized argument looks exactly like a
# missing binary. Lucy's plan schema crossed it on the first real turn.


def huge(chars: int) -> dict[str, object]:
    """A schema whose compact JSON is at least `chars` long."""
    return {"type": "object", "properties": {"a": {"description": "x" * chars}}}


def test_a_call_that_already_fits_is_returned_unchanged() -> None:
    call = Call(prompt="hi", system="s", model="opus", json_schema={"type": "object"})
    assert fit(call, binary=BINARY) is call


def test_an_oversized_schema_moves_onto_stdin() -> None:
    call = Call(prompt="do the thing", json_schema=huge(ARGV_CEILING))
    fitted = fit(call, binary=BINARY)
    assert fitted.json_schema is None
    assert "--json-schema" not in build(fitted, binary=BINARY)
    assert fitted.prompt.startswith("do the thing")
    assert SCHEMA_IN_WORDS in fitted.prompt
    assert json.dumps(call.json_schema, separators=(",", ":")) in fitted.prompt


def test_an_oversized_system_prompt_moves_onto_stdin_without_somewhere_to_spill_it() -> None:
    call = Call(prompt="p", system="s" * (ARGV_CEILING + 1))
    fitted = fit(call, binary=BINARY)
    assert fitted.system == ""
    assert "--system-prompt" not in build(fitted, binary=BINARY)
    assert fitted.prompt.startswith("s" * 100)
    assert fitted.prompt.endswith("p")


# --- spilling ---------------------------------------------------------------------------


SPILL_DIR = "/var/spool/clyde-test"
"""A path, not a real directory: `spiller` never writes, it only reports where it would."""


def spiller(written: list[str]) -> Callable[[str], str]:
    def spill(text: str) -> str:
        written.append(text)
        return f"{SPILL_DIR}/spilled-{len(written)}.txt"

    return spill


def test_a_spilled_system_prompt_keeps_its_role() -> None:
    """The point of the file: the model sees the same words in the same place. Folding them
    into the conversation is what made one turn re-plan twelve times."""
    written: list[str] = []
    call = Call(prompt="p", system="s" * (ARGV_CEILING + 1))
    fitted = fit(call, binary=BINARY, spill=spiller(written))
    assert written == ["s" * (ARGV_CEILING + 1)]
    assert fitted.system == ""
    assert fitted.prompt == "p", "the conversation is untouched"
    assert (
        pair(build(fitted, binary=BINARY), "--system-prompt-file") == f"{SPILL_DIR}/spilled-1.txt"
    )


def test_spilling_is_tried_before_anything_is_degraded() -> None:
    """When the file buys enough room, the schema stays a flag and nothing is traded away."""
    written: list[str] = []
    call = Call(prompt="p", system="s" * 26_000, json_schema=huge(4_000))
    fitted = fit(call, binary=BINARY, spill=spiller(written))
    assert fitted.json_schema == call.json_schema
    assert "--json-schema" in build(fitted, binary=BINARY)
    assert fitted.system_file
    assert length(build(fitted, binary=BINARY)) <= ARGV_CEILING


def test_a_schema_too_large_to_survive_a_spill_still_moves() -> None:
    """Lucy's schema is 30,984 characters: past the ceiling on its own, so the file buys room
    but not enough, and the schema still has to be asked for in words."""
    written: list[str] = []
    call = Call(prompt="p", system="s" * 22_000, json_schema=huge(31_000))
    fitted = fit(call, binary=BINARY, spill=spiller(written))
    assert fitted.system_file
    assert fitted.json_schema is None
    assert SCHEMA_IN_WORDS in fitted.prompt
    assert length(build(fitted, binary=BINARY)) <= ARGV_CEILING


def test_only_a_system_prompt_is_worth_spilling() -> None:
    """A call that already fits writes nothing: a file per request, for nothing, is a file
    per request full of somebody's prompt."""
    written: list[str] = []
    fit(Call(prompt="p", system="short"), binary=BINARY, spill=spiller(written))
    assert written == []


def test_both_move_when_both_are_oversized() -> None:
    call = Call(prompt="p", system="s" * ARGV_CEILING, json_schema=huge(ARGV_CEILING))
    fitted = fit(call, binary=BINARY)
    assert fitted.json_schema is None
    assert fitted.system == ""
    assert length(build(fitted, binary=BINARY)) <= ARGV_CEILING


def test_a_call_that_cannot_be_shrunk_further_is_returned_anyway() -> None:
    """Nothing is left to move: the model name alone is over the ceiling. Handing back an argv
    that will fail, and say so, is better than quietly dropping something that was asked for."""
    call = Call(prompt="p", model="m" * (ARGV_CEILING + 1))
    fitted = fit(call, binary=BINARY)
    assert length(build(fitted, binary=BINARY)) > ARGV_CEILING
    assert fitted is call


def test_length_counts_the_separators_between_arguments() -> None:
    assert length(["ab", "cde"]) == len("ab cde")


def test_the_ceiling_leaves_room_under_the_windows_limit() -> None:
    assert ARGV_CEILING < 32767


def test_the_schema_in_words_says_how_to_stop_asking() -> None:
    """The sentence that ends a reply. Without it the first real caller re-emitted the same
    plan eight rounds running: "reply with JSON only" is an instruction a good model follows
    forever, and the caller's loop ends only on a reply that is not JSON."""
    assert "prose" in SCHEMA_IN_WORDS
    assert "That is how a reply ends." in SCHEMA_IN_WORDS


def test_the_schema_in_words_asks_for_no_code_fence() -> None:
    """Told once, the model fenced its JSON anyway, and a caller parsing the whole reply read
    that as prose."""
    assert "no code fence" in SCHEMA_IN_WORDS


def test_the_schema_is_named_right_next_to_the_words_asking_for_it() -> None:
    call = Call(prompt="do the thing", json_schema=huge(ARGV_CEILING))
    fitted = fit(call, binary=BINARY)
    assert fitted.prompt.endswith(json.dumps(call.json_schema, separators=(",", ":")))


def test_the_schema_in_words_forbids_tool_call_syntax() -> None:
    """Claude Code emits `<invoke name="...">` when it decides to call something, even with no
    tool to call -- a schema full of named operations reads like a set of tools. Four lines of
    it in front of a perfectly good plan made the plan parse as nothing."""
    assert "<invoke>" in SCHEMA_IN_WORDS
    assert "no tools in this conversation" in SCHEMA_IN_WORDS

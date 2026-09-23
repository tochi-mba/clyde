"""The argv is the security posture, so it is asserted flag by flag.

The test that matters most is `test_the_mcp_lockdown_is_in_every_argv`. Without that pair of
flags the account's own connectors load: measured at 34,933 input tokens against 701, and
about 140 tools offered against none. No later test would catch it, because the service would
still answer correctly -- expensively, and next to a live Gmail tool.
"""

from __future__ import annotations

import json

import pytest

from clyde.cli.argv import EMPTY_MCP, MAX_TURNS, PERMISSION_MODE, Call, build, probe

BINARY = "/usr/local/bin/claude"


def pair(argv: list[str], flag: str) -> str:
    """The value that follows a flag, so a test reads as a claim rather than an index."""
    return argv[argv.index(flag) + 1]


def test_the_mcp_lockdown_is_in_every_argv() -> None:
    """The pair that removes the account's connectors. Absent, everything else still works."""
    for call in (
        Call(prompt="hi"),
        Call(prompt="hi", system="s", model="opus", disallowed_tools=("Bash",)),
        Call(prompt="hi", json_schema={"type": "object"}),
    ):
        argv = build(call, binary=BINARY)
        assert "--strict-mcp-config" in argv
        assert pair(argv, "--mcp-config") == EMPTY_MCP
        assert json.loads(EMPTY_MCP) == {"mcpServers": {}}


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


def test_the_system_prompt_is_passed_verbatim() -> None:
    """Replacing rather than appending is what makes the caller's prompt the only prompt."""
    argv = build(Call(prompt="hi", system="You are terse."), binary=BINARY)
    assert pair(argv, "--system-prompt") == "You are terse."


def test_no_system_prompt_flag_when_there_is_no_system_message() -> None:
    assert "--system-prompt" not in build(Call(prompt="hi"), binary=BINARY)


def test_the_model_is_passed_when_named_and_omitted_when_not() -> None:
    assert pair(build(Call(prompt="hi", model="opus"), binary=BINARY), "--model") == "opus"
    assert "--model" not in build(Call(prompt="hi"), binary=BINARY)


def test_disallowed_tools_are_one_comma_separated_value() -> None:
    argv = build(Call(prompt="hi", disallowed_tools=("Bash", "Read", "Edit")), binary=BINARY)
    assert pair(argv, "--disallowedTools") == "Bash,Read,Edit"


def test_no_disallow_flag_when_there_is_nothing_to_disallow() -> None:
    assert "--disallowedTools" not in build(Call(prompt="hi"), binary=BINARY)


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
    argv = build(
        Call(prompt="hi", system="s", model="opus", json_schema={}, disallowed_tools=("Bash",)),
        binary=BINARY,
    )
    assert flag not in argv


def test_the_probe_asks_nothing_of_a_model_but_reports_what_loaded() -> None:
    argv = probe(BINARY)
    assert pair(argv, "--output-format") == "stream-json"
    assert "--verbose" in argv
    assert "--json-schema" not in argv
    # No lockdown here on purpose: the probe exists to see what *would* load.
    assert "--strict-mcp-config" not in argv

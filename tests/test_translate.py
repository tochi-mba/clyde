"""Rendering a conversation in, and mapping one run back out."""

from __future__ import annotations

import pytest

from clyde.cli.run import Outcome
from clyde.openai.translate import (
    CLOSING_LINE,
    CONVERSATION_OPEN,
    finish_reason,
    render,
    schema_of,
    to_call,
    to_completion,
    usage_of,
)


def outcome(**kwargs: object) -> Outcome:
    base: dict[str, object] = {"result": "hello"}
    base.update(kwargs)
    return Outcome(**base)  # type: ignore[arg-type]


# --- rendering in --------------------------------------------------------------------------


def test_a_lone_user_message_is_sent_bare() -> None:
    """The common case should look like what it is, not like a transcript of one turn."""
    system, prompt = render([{"role": "user", "content": "say ok"}])
    assert system == ""
    assert prompt == "say ok"
    assert CONVERSATION_OPEN not in prompt


def test_the_system_message_is_lifted_out() -> None:
    system, prompt = render(
        [{"role": "system", "content": "You are terse."}, {"role": "user", "content": "hi"}]
    )
    assert system == "You are terse."
    assert prompt == "hi"


def test_several_system_messages_join() -> None:
    system, _ = render(
        [
            {"role": "system", "content": "One."},
            {"role": "system", "content": "Two."},
            {"role": "user", "content": "hi"},
        ]
    )
    assert system == "One.\n\nTwo."


def test_a_real_conversation_is_delimited_and_labelled() -> None:
    _, prompt = render(
        [
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "second"},
            {"role": "user", "content": "third"},
        ]
    )
    assert prompt.startswith(CONVERSATION_OPEN)
    assert '<turn role="assistant">second</turn>' in prompt
    assert prompt.endswith(CLOSING_LINE)


def test_an_assistant_only_history_is_still_wrapped() -> None:
    """Only a single *user* message skips the wrapper; anything else keeps its labels."""
    _, prompt = render([{"role": "assistant", "content": "alone"}])
    assert CONVERSATION_OPEN in prompt


def test_no_messages_at_all_renders_to_nothing() -> None:
    assert render([]) == ("", f"{CONVERSATION_OPEN}\n\n</conversation>\n\n{CLOSING_LINE}")


def test_missing_content_does_not_crash() -> None:
    system, prompt = render([{"role": "user"}])
    assert (system, prompt) == ("", "")


# --- the schema ----------------------------------------------------------------------------


def test_openai_nests_the_schema_one_deeper() -> None:
    inner = {"type": "object", "properties": {}}
    assert schema_of({"type": "json_schema", "json_schema": {"schema": inner}}) == inner


def test_a_bare_schema_block_is_accepted_too() -> None:
    """Several compatible servers send it flat; refusing would be pedantry."""
    block = {"type": "object"}
    assert schema_of({"type": "json_schema", "json_schema": block}) == block


@pytest.mark.parametrize(
    "response_format",
    [None, {}, {"type": "json_object"}, {"type": "json_schema"}, {"type": "text"}],
)
def test_anything_that_is_not_a_schema_is_none(response_format: dict[str, object] | None) -> None:
    assert schema_of(response_format) is None


def test_to_call_carries_everything_through() -> None:
    call = to_call(
        {
            "model": "opus",
            "messages": [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}],
            "response_format": {"type": "json_schema", "json_schema": {"schema": {"a": 1}}},
        },
        disallowed=["Bash", "Read"],
        default_model="sonnet",
    )
    assert (call.model, call.system, call.prompt) == ("opus", "s", "u")
    assert call.json_schema == {"a": 1}
    assert call.disallowed_tools == ("Bash", "Read")


def test_the_default_model_is_used_when_none_is_named() -> None:
    call = to_call({"messages": []}, disallowed=[], default_model="sonnet")
    assert call.model == "sonnet"


def test_a_body_with_no_messages_key_is_not_a_crash() -> None:
    assert to_call({}, disallowed=[], default_model="sonnet").prompt.startswith(CONVERSATION_OPEN)


# --- mapping out ---------------------------------------------------------------------------


def test_a_normal_reply_stops() -> None:
    assert finish_reason(outcome()) == "stop"


def test_reaching_the_turn_limit_reads_as_length() -> None:
    """It wanted to keep going. That is the same situation as running out of tokens, and a
    caller treats `length` as resumable where it treats a refusal as final."""
    assert finish_reason(outcome(subtype="error_max_turns")) == "length"


def test_a_structured_output_call_is_not_truncated() -> None:
    """Measured against the real CLI: `--json-schema` reports two turns and `tool_use` on a
    successful run. Calling that `length` would tell a caller every plan ran out of room."""
    assert finish_reason(outcome(num_turns=2, stop_reason="tool_use")) == "stop"


def test_usage_folds_the_cache_into_the_prompt_total_and_names_it_again() -> None:
    mapped = usage_of(
        outcome(
            usage={
                "input_tokens": 2,
                "cache_read_input_tokens": 26225,
                "cache_creation_input_tokens": 8687,
                "output_tokens": 4,
            }
        )
    )
    assert mapped["prompt_tokens"] == 2 + 26225 + 8687
    assert mapped["completion_tokens"] == 4
    assert mapped["total_tokens"] == 2 + 26225 + 8687 + 4
    assert mapped["prompt_tokens_details"]["cached_tokens"] == 26225


def test_absent_usage_is_zeroes_rather_than_a_crash() -> None:
    assert usage_of(outcome()) == {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "prompt_tokens_details": {"cached_tokens": 0},
    }


def test_the_completion_puts_the_reply_in_message_content() -> None:
    body = to_completion(outcome(result="hello there"), model="sonnet", now=1)
    assert body["choices"][0]["message"] == {"role": "assistant", "content": "hello there"}
    assert body["object"] == "chat.completion"
    assert body["created"] == 1


def test_structured_output_arrives_as_a_string_in_content() -> None:
    """The trap in the module docstring: a sibling field would be silently ignored."""
    plan = '{"steps":[{"id":"a","op":"notes.search"}]}'
    body = to_completion(outcome(result=plan), model="sonnet", now=1)
    assert body["choices"][0]["message"]["content"] == plan
    assert isinstance(body["choices"][0]["message"]["content"], str)


def test_the_model_the_cli_reported_wins_over_the_one_requested() -> None:
    """A caller asking for `sonnet` and being served opus should be able to see that."""
    body = to_completion(outcome(model="claude-opus-5-5"), model="sonnet", now=1)
    assert body["model"] == "claude-opus-5-5"


def test_the_requested_model_is_used_when_the_cli_names_none() -> None:
    assert to_completion(outcome(), model="sonnet", now=1)["model"] == "sonnet"


def test_the_session_id_becomes_the_completion_id() -> None:
    body = to_completion(outcome(session_id="abc"), model="sonnet", now=1)
    assert body["id"] == "chatcmpl-abc"
    assert to_completion(outcome(), model="s", now=1)["id"] == "chatcmpl-clyde"


def test_created_defaults_to_now() -> None:
    assert to_completion(outcome(), model="s")["created"] > 0

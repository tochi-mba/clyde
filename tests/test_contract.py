"""Lucy's own parser reads what this emits.

Every other test in this repository checks that clyde does what clyde intends. This one checks
the thing that actually matters: that the bytes on the wire are the bytes the consumer expects.
It imports `reply_from` from the hub itself and runs it over a body this service built.

The failure it exists to catch is silent. A plan handed back in a sibling field instead of
inside `message.content` produces a perfectly valid response, a perfectly happy client, and a
`Reply.plan` of `None` -- so a turn that should have run tools becomes a turn that said
something. Nothing logs, nothing errors, and the symptom is "the model seems worse today".

Skipped rather than failed when the hub is not checked out beside this repository: a
contributor with only this repository should still get a green suite.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from clyde.cli.run import Outcome
from clyde.openai.translate import to_completion

HUB = Path(__file__).resolve().parents[2] / "LUCY-assistant" / "src"

if not (HUB / "lucy_api" / "model" / "chat.py").is_file():
    pytest.skip("the hub is not checked out beside this repository", allow_module_level=True)

sys.path.insert(0, str(HUB))

from lucy_api.model.chat import reply_from  # noqa: E402
from lucy_api.model.types import Stop  # noqa: E402

PLAN = {
    "steps": [
        {"id": "a", "op": "notes.search", "input": {"query": "tea"}},
        {"id": "b", "op": "notes.remember", "input": {"body": "prefers tea"}},
    ]
}


def completion(result: str, **kwargs: object) -> dict[str, object]:
    return to_completion(Outcome(result=result, **kwargs), model="sonnet", now=1)  # type: ignore[arg-type]


def test_a_plain_reply_reaches_the_hub_as_text() -> None:
    reply = reply_from(completion("hello there"), provider="clyde", want_plan=False)
    assert reply.text == "hello there"
    assert reply.plan is None
    assert reply.stop is Stop.end_turn


def test_a_plan_survives_the_round_trip() -> None:
    """The one that matters. `--json-schema` puts the JSON in `result`; `result` becomes
    `message.content`; the hub parses the plan back out of that string."""
    reply = reply_from(completion(json.dumps(PLAN)), provider="clyde", want_plan=True)
    assert reply.plan == PLAN
    assert reply.stop is Stop.tool_use
    # The hub blanks the text when a plan parsed: the JSON answered a machine, not a person.
    assert reply.text == ""


def test_prose_when_a_plan_was_wanted_is_not_a_plan() -> None:
    """A repair round quotes the raw text back, so it has to survive."""
    reply = reply_from(completion("I could not do that"), provider="clyde", want_plan=True)
    assert reply.plan is None
    assert reply.text == "I could not do that"


def test_the_turn_limit_reaches_the_hub_as_resumable() -> None:
    """`max_tokens` offers a retry where a refusal does not, so this mapping is load-bearing."""
    reply = reply_from(
        completion("cut off", subtype="error_max_turns"), provider="clyde", want_plan=False
    )
    assert reply.stop is Stop.max_tokens


def test_a_plan_call_does_not_reach_the_hub_as_truncated() -> None:
    """The bug this file exists to catch, found by running it: `--json-schema` reports two
    turns on success, and calling that truncation would make the hub offer to resume every
    plan-shaped turn it ever took."""
    reply = reply_from(
        completion(json.dumps(PLAN), num_turns=2, stop_reason="tool_use"),
        provider="clyde",
        want_plan=True,
    )
    assert reply.stop is Stop.tool_use
    assert reply.plan == PLAN


def test_usage_arrives_with_the_cache_split_out() -> None:
    """The hub subtracts the cached tokens from the input total, so a cache-read ratio is
    computable at the far end. Getting the nesting wrong would silently report zero."""
    reply = reply_from(
        completion(
            "hi",
            usage={
                "input_tokens": 2,
                "cache_read_input_tokens": 26225,
                "cache_creation_input_tokens": 8687,
                "output_tokens": 4,
            },
        ),
        provider="clyde",
        want_plan=False,
    )
    assert reply.usage.cache_read_tokens == 26225
    assert reply.usage.input_tokens == 2 + 8687
    assert reply.usage.output_tokens == 4


def test_the_model_name_reaches_the_hub() -> None:
    reply = reply_from(completion("hi", model="claude-opus-5-5"), provider="clyde", want_plan=False)
    assert reply.model == "claude-opus-5-5"

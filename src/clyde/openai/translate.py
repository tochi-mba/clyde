"""Between the chat-completions dialect and one `claude -p` run.

Two directions, and the second one carries a trap worth stating plainly.

**In.** A list of role/content messages becomes one system prompt and one stdin blob. Claude
Code takes a single prompt; a conversation has to be rendered into it, and rendering it
badly is how a model starts answering the wrong turn.

**Out.** A caller parsing a plan reads `choices[0].message.content` and parses JSON *out of
that string*. So structured output has to arrive there, as a string. Putting it in a sibling
field instead -- which is the obvious thing to do, since the CLI hands back a parsed
`structured_output` object as well -- means the caller sees prose, finds no plan, and treats a
tool-using turn as a chat turn. Nothing errors. That is the whole failure.

Conveniently the CLI's own `result` field already holds the JSON as a string when a schema was
used, so the correct thing and the simple thing are the same thing: pass `result` through.

**Streaming.** `claude -p --output-format json` returns the whole reply at once, so there is
nothing to stream token by token. But a caller that asks for `stream: true` is not asking for
tokens, it is asking for *this protocol*, and several hosts have no non-streaming path at all
-- Lucy attaches a chunk projector to every turn, so it always sends `stream: true`. Refusing
it means refusing them.

So :func:`stream_chunks` delivers the same reply down the streaming shape: one delta carrying
the text, one carrying the finish reason, one carrying usage, then `[DONE]`. It arrives in one
piece rather than progressively, which is a real limitation and is documented as one -- but the
caller gets its reply, assembled exactly as the dialect says it should be.
"""

from __future__ import annotations

import json
import time
from typing import TYPE_CHECKING, Any

from clyde.cli.argv import Call

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from clyde.cli.run import Outcome

CONVERSATION_OPEN = "<conversation>"
CONVERSATION_CLOSE = "</conversation>"
CLOSING_LINE = "Reply as the assistant to the last user turn."


def render(messages: Sequence[dict[str, Any]]) -> tuple[str, str]:
    """`(system, prompt)` from a chat-completions message list.

    A single user message is sent bare, with no wrapper at all: the common case should look
    like what it is, and wrapping it teaches the model that every prompt is a transcript.
    Anything longer is delimited and labelled, because "who said what" is the one thing a
    flattened conversation loses first.
    """
    system = "\n\n".join(
        str(m.get("content") or "") for m in messages if m.get("role") == "system"
    ).strip()
    rest = [m for m in messages if m.get("role") != "system"]

    if len(rest) == 1 and rest[0].get("role") == "user":
        return system, str(rest[0].get("content") or "")

    turns = "\n".join(
        f'<turn role="{m.get("role", "user")}">{m.get("content") or ""}</turn>' for m in rest
    )
    return system, f"{CONVERSATION_OPEN}\n{turns}\n{CONVERSATION_CLOSE}\n\n{CLOSING_LINE}"


def schema_of(response_format: Mapping[str, Any] | None) -> Mapping[str, Any] | None:
    """The JSON Schema a caller asked for, or `None`.

    Both spellings are accepted because both are in the wild: `json_schema.schema` is
    OpenAI's, and a bare `schema` is what several compatible servers send.
    """
    if not response_format or response_format.get("type") != "json_schema":
        return None
    block = response_format.get("json_schema")
    if isinstance(block, dict):
        inner = block.get("schema")
        return inner if isinstance(inner, dict) else block
    return None


def to_call(body: dict[str, Any], *, disallowed: Sequence[str], default_model: str) -> Call:
    """One chat-completions body as one :class:`Call`."""
    messages = body.get("messages")
    system, prompt = render(messages if isinstance(messages, list) else [])
    return Call(
        prompt=prompt,
        system=system,
        model=str(body.get("model") or default_model),
        json_schema=schema_of(body.get("response_format")),
        disallowed_tools=tuple(disallowed),
    )


def finish_reason(outcome: Outcome) -> str:
    """Lucy's `Stop` vocabulary is reached through this, so the mapping is load-bearing.

    `length` is resumable and a refusal is not, and a client shows them differently. Reaching
    the one-turn limit means the model wanted to keep going, which is the same situation as
    running out of tokens, so it maps to `length` rather than to `stop`.
    """
    if outcome.hit_turn_limit:
        return "length"
    return "stop"


def usage_of(outcome: Outcome) -> dict[str, Any]:
    """Anthropic's usage names, in OpenAI's.

    The CLI reports `input_tokens`, `cache_read_input_tokens` and `cache_creation_input_tokens`
    separately. A caller expecting OpenAI reads `prompt_tokens` as the whole input and finds
    the cached part under `prompt_tokens_details.cached_tokens`, so the cached tokens are
    *included* in the total here and named again underneath -- which is what OpenAI does, and
    what makes a cache-read ratio computable at the other end.
    """
    raw = outcome.usage
    fresh = int(raw.get("input_tokens") or 0)
    cached = int(raw.get("cache_read_input_tokens") or 0)
    created = int(raw.get("cache_creation_input_tokens") or 0)
    completion = int(raw.get("output_tokens") or 0)
    prompt = fresh + cached + created
    return {
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_tokens": prompt + completion,
        "prompt_tokens_details": {"cached_tokens": cached},
    }


def to_completion(outcome: Outcome, *, model: str, now: float | None = None) -> dict[str, Any]:
    """One :class:`Outcome` as a chat-completions response body.

    `message.content` carries the reply -- including structured output, as a string. See the
    module docstring for why that is not negotiable.
    """
    return {
        "id": f"chatcmpl-{outcome.session_id or 'clyde'}",
        "object": "chat.completion",
        "created": int(now if now is not None else time.time()),
        "model": outcome.model or model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": outcome.result},
                "finish_reason": finish_reason(outcome),
            }
        ],
        "usage": usage_of(outcome),
    }


DONE = "[DONE]"
"""The sentinel that ends a chat-completions stream. A word, not JSON."""


def stream_chunks(outcome: Outcome, *, model: str, now: float | None = None) -> list[str]:
    """One reply as the `data:` lines of a stream, ending with the sentinel.

    Three chunks and a sentinel, in the order the dialect specifies: the text, the finish
    reason, the usage. A reader assembles the reply from these, so the split has to match
    what a reader expects even though nothing here is actually incremental.
    """
    created = int(now if now is not None else time.time())
    identifier = f"chatcmpl-{outcome.session_id or 'clyde'}"
    named = outcome.model or model

    def frame(payload: Mapping[str, Any]) -> str:
        chunk = {
            "id": identifier,
            "object": "chat.completion.chunk",
            "created": created,
            "model": named,
            **payload,
        }
        return "data: " + json.dumps(chunk, separators=(",", ":")) + "\n\n"

    text = {"role": "assistant", "content": outcome.result}
    return [
        frame({"choices": [{"index": 0, "delta": text}]}),
        frame({"choices": [{"index": 0, "delta": {}, "finish_reason": finish_reason(outcome)}]}),
        frame({"choices": [], "usage": usage_of(outcome)}),
        "data: " + DONE + "\n\n",
    ]


__all__ = [
    "CLOSING_LINE",
    "CONVERSATION_CLOSE",
    "CONVERSATION_OPEN",
    "DONE",
    "finish_reason",
    "render",
    "schema_of",
    "stream_chunks",
    "to_call",
    "to_completion",
    "usage_of",
]

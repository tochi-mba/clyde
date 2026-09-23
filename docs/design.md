# Why the flags are the flags

Measured on 2026-09-23, before any of this existed, against `claude` 2.1.280.

| Configuration | Input tokens | Cost | Tools offered |
| --- | ---: | ---: | ---: |
| `claude -p "reply with the single word ok"` | 34,933 | $0.0758 | ~140 |
| `+ --system-prompt` | 18,983 | $0.1529 | ~140 |
| `+ --strict-mcp-config` + the full disallow list | **701** | **$0.0066** | **0** |

The middle row is the trap. `--system-prompt` replaces Claude Code's own prompt, which looks
like it should be enough, and it halves the context. The remaining 19k is the **account's** MCP
connectors — Gmail, Google Drive, Linear, Google Calendar — loaded, connected and offered to
the model as tools. They come from the account (`source: claudeai`), not from any file, so
`--disallowedTools` cannot name them and an empty working directory does not affect them.

That is not only a cost problem. A caller's prompt would sit beside a live Gmail tool, in a
context that also carries that caller's own tool results.

`--mcp-config '{"mcpServers":{}}' --strict-mcp-config` removes them, and keeps subscription
auth. `--bare` would also work and is the documented mode for scripted calls, but it "never
reads OAuth credentials or the system keychain" — it needs an API key, which is the thing this
exists to avoid.

## The disallow list is learned, not written

Shortening it in one experiment let the built-in tools back in: $0.012 became $0.294 for the
same prompt. So the harness runs one `stream-json` probe at startup, reads the `tools` array
out of `system/init`, and passes exactly that back as `--disallowedTools`. On this machine it
learns 149 names. A Claude Code release that adds a tool cannot quietly widen what a caller's
model can reach.

## `num_turns` is not a truncation signal

Found by running it. A `--json-schema` call reports `num_turns: 2` and `stop_reason: tool_use`
on a completely successful run, because the CLI emits the structured output through an internal
tool. Reading either as truncation told every caller that every plan-shaped turn had run out of
room — and a caller that believes that offers to resume a turn which already finished.

The honest signal is `subtype`, which is `success` or an `error_*` reason.

## Structured output goes in `message.content`, as a string

A consumer parsing a plan reads `choices[0].message.content` and parses JSON out of that
string. The CLI hands back both a `result` string and a parsed `structured_output` object; the
obvious move is to use the object, and it is wrong. Putting it anywhere but `content` produces
a valid response, a happy client, and no plan — nothing errors, and the symptom is "the model
seems worse today".

`result` already holds the JSON as a string, so the correct thing and the simple thing agree.

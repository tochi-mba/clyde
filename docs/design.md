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

## No tools, by construction

Until 2026-09-24 the built-in tools were removed by a disallow list learned at startup.
Shortening a hand-written list in one experiment had let them back in — $0.012 became $0.294
for the same prompt — so the harness ran one `stream-json` probe, read the `tools` array out
of `system/init`, and passed exactly that back as `--disallowedTools`. When it was written it
learned 149 names.

It leaked anyway. The probe ran with ToolSearch available, which defers some tools out of the
init listing; real calls disallowed ToolSearch, so the deferred tools came back. Measured
against `claude` 2.1.280: a real call's init line read `"tools":["TodoWrite"]` while `/ready`
reported 157 names disallowed. Haiku, asked to build a website, called TodoWrite — which
spends the only turn — and the run ended `error_max_turns`, or with an empty reply. A deny-list
can only name what somebody saw.

Every call now passes `--tools ""` (`claude --help`: *Use "" to disable all tools*) and
`--disable-slash-commands` (*Disable all skills*) beside the MCP pair, which stays because
`--tools` governs only the built-in set. Replayed with them, the two requests that had failed
each answered in one turn with a plan, and a `--json-schema` call still returned its structured
output. The deny-list is gone rather than kept as a second layer: it is also what let `/ready`
vouch for a lockdown that was not there.

The startup probe now runs under exactly the same flags, and its job is to show that nothing
loads rather than to learn what to deny. Under the lockdown its init line reads `"tools":[]`
and `"mcp_servers":[]`. `/ready`'s `lockdown` check is `ok` only on that proof. Anything else
— a tool, a server, or an init line that could not be read — degrades it, names what loaded,
and refuses every call with a 503 `not_locked_down`: a service that knows a tool is loaded and
serves anyway is the situation this replaced.

## One turn, or four with a schema

Every call passes `--max-turns 1`. This is a model, not an agent: the caller runs the loop.

A call carrying `--json-schema` gets four (`STRUCTURED_MAX_TURNS`). Its answer travels through
the CLI's own structured-output tool, which costs a turn of its own, and a model that answers
in words first is asked again. Measured on 2026-09-24 against Haiku, with a schema small enough
to stay on the command line: allowed one turn, it ended `error_max_turns after 2 turns` three
times in three; allowed three, it answered in three every time. The fourth is room for one
answer that fails the schema. Under the lockdown there is no other tool to spend a turn on, so
it is still one request and one reply.

A schema that had to move into the prompt (below) is no longer a `--json-schema` call, and keeps
its single turn.

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

## What leaves the command line

Windows caps a command line at 32,767 characters, and `CreateProcess` reports the overflow as
`FileNotFoundError`: the same error as a missing binary, saying nothing about length. Lucy sends
a 22,232-character system prompt and a 30,984-character plan schema, 55,716 characters together.
So `fit` in `cli/argv.py` keeps every argv under 28,000 characters (`ARGV_CEILING`), on every
platform so that a Linux test proves what Windows does, and moves things in order of cost:

1. The system prompt goes to a file, passed as `--system-prompt-file`. The model sees the same
   words in the same role, so this costs nothing, and it is usually enough.
2. The schema goes into the prompt, in words (`SCHEMA_IN_WORDS`). That is a request where
   `--json-schema` is a guarantee, but it still reaches the model.
3. Only when there is nowhere to write a file is the system prompt folded into the
   conversation. That one is expensive, and was measured: the turn that kept its system prompt
   planned, ran its step and stopped after two rounds; the turn that folded the same words in
   re-planned twelve times and died on the caller's iteration cap.

The conversation itself always goes on stdin. The spilled file sits beside the empty working
directory rather than inside it, and is deleted when the call ends.

On Windows, `PATH` finds `claude.CMD`, a batch file, and running it means running `cmd.exe`,
whose command line stops at 8,191 characters. `through_shim` in `cli/locate.py` reads the batch
file and runs the executable it points at instead.

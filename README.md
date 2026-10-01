# clyde

Site: <https://tochi-mba.github.io/clyde/>

The Claude Code CLI as an OpenAI-compatible model provider on localhost.

Point anything that speaks `POST /v1/chat/completions` at it and the replies come from
`claude -p` running under your own Claude Code login.

    uv run clyde serve --port 8127
    curl localhost:8127/v1/chat/completions -H 'content-type: application/json' \
      -d '{"model":"sonnet","messages":[{"role":"user","content":"say ok"}]}'

## What it is not

It is not an agent. Every call runs with no tools, no MCP servers, no skills and an empty
working directory, and with one turn -- four when the call asks for a JSON schema, because the
CLI spends turns handing a structured answer back. The thing on the other end behaves like a
model rather than like Claude Code. `src/clyde/cli/argv.py` builds the flags, and
[docs/design.md](docs/design.md) records the measurements that chose them.

That is checked, not assumed: at startup the CLI runs once under the same flags, `/ready`
reports what it loaded, and every call is refused unless that was nothing.

It binds to loopback and authenticates nobody, which is the same posture as any other local
runtime on the same machine. Do not expose the port.

## Before you start

- Python 3.12 or 3.13, and [uv](https://docs.astral.sh/uv/).
- The Claude Code CLI, signed in: `npm install -g @anthropic-ai/claude-code`, then run
  `claude` once in a terminal and sign in. clyde uses that login and never needs an API key.

clyde looks for `claude` on `PATH`, then in the places npm usually puts a global binary:
`~/.local/bin`, `~/.npm-global/bin`, `/usr/local/bin`, `/opt/homebrew/bin`, and `%APPDATA%\npm`
on Windows. A service manager often runs with a shorter `PATH` than your shell, so set
`CLYDE_CLAUDE_BINARY` when it cannot be found. A missing binary does not stop the server:
`/ready` says what is wrong, and every call is refused with the same sentence.

## Run it

```bash
uv sync
uv run clyde serve                  # 127.0.0.1:8127
uv run clyde serve --port 9000      # --host and --port win over CLYDE_HOST and CLYDE_PORT
make run                            # always :8127, reloading on code changes
```

Before it accepts a connection, startup runs `claude` once under the same flags as every call
to see what a call would load. That takes a few seconds, and at most two minutes.

## Routes

| Route | What it answers |
| --- | --- |
| `GET /healthy` | `{"status": "ok", "version": ...}` whenever the process is up. It checks nothing. |
| `GET /ready` | 200 when a call would work, otherwise 503. `checks.claude` says whether the binary was found. `checks.lockdown` lists the tools and MCP servers the startup probe saw load, and is `ok` only when both lists are empty. |
| `GET /v1/models` | The aliases the CLI accepts, in OpenAI's list shape: `sonnet`, `opus`, `haiku`, `fable`. |
| `POST /v1/chat/completions` | One chat completion, answered by one `claude -p` run. |

### What a completion request may carry

- `messages`: system messages become the system prompt. A single user message is sent as it
  is; anything longer is rendered into one `<conversation>` of `<turn role="...">` blocks.
- `model`: any alias the CLI accepts. When it is missing, `CLYDE_DEFAULT_MODEL`.
- `response_format`: `{"type": "json_schema", "json_schema": {"schema": {...}}}` asks for
  structured output. The JSON arrives as a string in `choices[0].message.content`, which is
  where a chat-completions client looks for it.
- `stream: true`: the same reply as server-sent events: one chunk with the text, one with the
  finish reason, one with usage, then `data: [DONE]`. `claude -p` returns the whole reply at
  once, so it arrives in one piece rather than token by token.
- `tools`: refused with a 400. Ask for structured output instead.

Other fields are ignored. A reply that comes back as tool-call syntax (`<invoke>` and the like)
is asked for again, once. An empty reply is an error, never a finished turn.

### Errors

Errors use OpenAI's envelope, `{"error": {"message": ..., "type": ..., "code": ...}}`, and the
message is a sentence you can act on. Every refusal is also written to clyde's log.

| Status | `code` | Meaning |
| --- | --- | --- |
| 400 | `invalid_request_error` | The body is not JSON or not an object, or the conversation is over `CLYDE_STDIN_LIMIT_BYTES`. |
| 400 | `tools_unsupported` | The request carried `tools`. |
| 403 | `not_logged_in` | The CLI is not signed in. Run `claude` once and sign in; retrying will not help. |
| 500 | `sandbox_contaminated` | Something appeared in the empty working directory that the CLI would load. |
| 500 | `internal_error` | Anything else that went wrong inside clyde. |
| 502 | `claude_failed` | The CLI ran and did not answer: it exited non-zero, ran out of turns, produced no text, or printed something clyde could not read. |
| 503 | `claude_not_found` | No `claude` binary. Sent with `Retry-After: 30`. |
| 503 | `not_locked_down` | The startup probe did not prove that nothing loads. Sent with `Retry-After: 30`; restart clyde to probe again. |
| 504 | `timeout` | The call outlived `CLYDE_TIMEOUT_SECONDS` and the process was killed. |

## Configure it

Every setting is an environment variable with a `CLYDE_` prefix, and a `.env` file in the
working directory is read too. A `CLYDE_*` name that no setting declares stops startup, so a
typo is a crash rather than a setting that silently did nothing.

| Variable | Default | What it does |
| --- | --- | --- |
| `CLYDE_HOST` | `127.0.0.1` | Interface to bind. Keep it on loopback. |
| `CLYDE_PORT` | `8127` | Port to listen on. |
| `CLYDE_LOG_LEVEL` | `INFO` | Level for clyde's own log lines, written to stderr. |
| `CLYDE_LOG_FORMAT` | `json` | `json` for one JSON object per line; anything else for `LEVEL: message`. |
| `CLYDE_CLAUDE_BINARY` | empty | Path to `claude`. Empty means search `PATH` and the usual places. |
| `CLYDE_DEFAULT_MODEL` | `sonnet` | The model used when a request names none. |
| `CLYDE_TIMEOUT_SECONDS` | `600` | Ceiling on one call, in seconds. |
| `CLYDE_MAX_CONCURRENT` | `2` | How many `claude` processes may run at once. Each is a whole Node runtime. |
| `CLYDE_STDIN_LIMIT_BYTES` | `9000000` | Largest conversation accepted, under the CLI's 10 MB stdin cap. |

`make run` starts uvicorn directly, so it ignores `CLYDE_HOST` and `CLYDE_PORT`.

## Using it from Lucy

The [Lucy hub](https://github.com/tochi-mba/LUCY-assistant) knows clyde as the `clyde`
provider at `http://localhost:8127/v1`. A hub on the same machine can use it as soon as
`lucy models` lists `clyde` as ready: `lucy eval run --model clyde:haiku`, for example, holds
the hub's eval conversations against Haiku. A hub running in Docker
reaches the host through `host.docker.internal`; the hub's README shows the
`LUCY_MODEL_BASE_URLS` value that points it there.

## Working on it

```bash
make install    # uv sync --group dev
make check      # lint, type, imports, test: what CI runs
make fmt        # format, and apply lint fixes
make cov        # HTML coverage report in htmlcov/
make help       # every target
```

The bar is the Lucy family's, though clyde is not a family service: ruff, mypy `strict`,
100% branch coverage, and three import-linter contracts in `pyproject.toml`. The layers run
`api -> openai -> cli -> core`, the CLI layer knows nothing about OpenAI's shapes, and only
`cli/run.py` may spawn a process. CI runs the same checks on Ubuntu and Windows, on Python 3.12
and 3.13, because on Windows the binary clyde spawns is a `.cmd` shim.

Nothing in `make check` spawns a real `claude`. `tests/test_contract.py` runs the hub's own
reply parser over what clyde emits, and skips itself unless
[LUCY-assistant](https://github.com/tochi-mba/LUCY-assistant) is checked out beside this
repository. `make live` runs the tests marked `live`, which are for calling the real CLI; there
are none yet, so today it says so and exits zero.

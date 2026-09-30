# clyde

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

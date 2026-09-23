# clyde

The Claude Code CLI as an OpenAI-compatible model provider on localhost.

Point anything that speaks `POST /v1/chat/completions` at it and the replies come from
`claude -p` running under your own Claude Code login.

    uv run clyde serve --port 8127
    curl localhost:8127/v1/chat/completions -H 'content-type: application/json' \
      -d '{"model":"sonnet","messages":[{"role":"user","content":"say ok"}]}'

## What it is not

It is not an agent. Every call runs with one turn, no tools, no MCP servers and an empty
working directory, so the thing on the other end behaves like a model rather than like Claude
Code. See `src/clyde/cli/argv.py` for the flags and the measurements that chose them.

It binds to loopback and authenticates nobody, which is the same posture as any other local
runtime on the same machine. Do not expose the port.

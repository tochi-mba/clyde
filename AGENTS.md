# Working notes for clyde

clyde turns one chat-completions request into one `claude -p` run and hands back what the CLI
said. It is small, and most of it is there to keep one promise: the model on the other end can
reach nothing. [README.md](README.md) is how to use it; [docs/design.md](docs/design.md) is why
the flags are what they are.

## Where things are

| Path | What it holds |
| --- | --- |
| `src/clyde/api/app.py` | The four routes, the startup probe, and the refusals made before anything spawns. |
| `src/clyde/openai/` | The chat-completions dialect: `translate.py` in and out, `errors.py` from failure to status code, `service.py` for what is decided once at startup. |
| `src/clyde/cli/` | Everything about the process: `argv.py` builds the flags, `run.py` spawns and parses, `locate.py` finds the binary, `sandbox.py` owns the empty working directory. |
| `src/clyde/core/` | Settings (`CLYDE_*`) and logging. |

## Invariants

**The flags are the security posture.** `argv.LOCKDOWN` is in every argv `build` and `probe`
produce: `--tools ""`, `--disable-slash-commands`, and the empty `--mcp-config` with
`--strict-mcp-config`. `tests/test_argv.py` asserts each flag literally, so deleting one from
the constant fails a test rather than passing with it. Never add a way to skip them, and never
add `--bare`, `--resume`, `--continue` or `--allowedTools`.

**A call is refused unless the startup probe proved nothing loads.** The probe runs under the
same flags as a call. An unreadable probe is not proof, so it refuses too. Do not turn a
refusal into a warning.

**Only `cli/run.py` spawns a process**, and the `cli` layer knows nothing about OpenAI's shapes.
Both are import-linter contracts in `pyproject.toml`, beside the layer order
`api -> openai -> cli -> core`.

**An empty reply is an error.** A run the CLI calls a success with no text in it becomes a 502,
checked once before the streaming and blocking paths part.

**Nothing in `make check` spawns a real `claude`.** Tests fake the subprocess. A test that needs
the real CLI is marked `live` and runs only under `make live`.

**It binds to loopback and authenticates nobody.** Adding auth would invent a threat this does
not have; exposing the port would hand a signed-in Claude Code to whoever finds it.

## Gates

`make check` is `lint type imports test`: ruff, mypy `strict`, the import contracts, and pytest
with 100% branch coverage and `filterwarnings = ["error"]`. CI runs the same steps on Ubuntu
and Windows, on Python 3.12 and 3.13. `.pre-commit-config.yaml` is a faster subset for a staged
diff; `make check` remains the authority.

clyde is not a Lucy family service, so it does not use the family's reusable workflow or its
parity check; `.github/workflows/ci.yml` says why. `tests/test_contract.py` still runs the hub's
own reply parser over clyde's output when the hub is checked out beside this repository.

## Commits

A subject is a plain imperative sentence that says what changed for a caller, with no type
prefix and no trailing period. The body says why, and what was measured when a measurement
decided it.

"""Finding `claude`, and saying something useful when it is not there.

`shutil.which` is most of it. The rest is two things that only show up on a real machine.

npm's global bin is routinely absent from the `PATH` a service manager hands a process, even
when it is present in the shell where somebody tested the command by hand -- so "works for me,
missing in the service" is the common failure, and a message naming the install command is
worth more than a stack trace.

And on Windows, what `PATH` finds is `claude.CMD`: a five-line batch file that calls the real
executable. Running it means running `cmd.exe`, whose command line caps at 8191 characters
against `CreateProcess`'s 32767. Lucy's system prompt alone is past that, and the failure is
`claude exited 1: The command line is too long.` -- from cmd, about cmd, naming nothing that
would lead anyone here. :func:`through_shim` reads the batch file and returns the executable
it points at, so the shim is never spawned.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

BINARY = "claude"

INSTALL_HINT = (
    "install it with `npm install -g @anthropic-ai/claude-code`, then run `claude` once to sign in"
)


class ClaudeNotFoundError(RuntimeError):
    """No `claude` on `PATH` or at the configured location."""


def candidates() -> list[Path]:
    """Places npm puts a global binary that a service's `PATH` often misses."""
    home = Path.home()
    appdata = os.environ.get("APPDATA", "")
    found = [
        home / ".local" / "bin" / BINARY,
        home / ".npm-global" / "bin" / BINARY,
        Path("/usr/local/bin") / BINARY,
        Path("/opt/homebrew/bin") / BINARY,
    ]
    if appdata:
        found += [Path(appdata) / "npm" / f"{BINARY}.cmd", Path(appdata) / "npm" / BINARY]
    return found


SHIM_SUFFIXES = (".cmd", ".bat")
"""Extensions that mean "batch file", and therefore "spawned through cmd.exe"."""


def through_shim(path: str) -> str:
    """The executable a Windows npm shim calls, or `path` unchanged.

    The shim's last line is the quoted path to the real binary followed by `%*`. Anything that
    does not look like that is returned untouched: a shim this cannot read still runs, just
    with cmd's shorter command line, which is the behaviour before this existed.
    """
    if not path.lower().endswith(SHIM_SUFFIXES):
        return path
    try:
        lines = Path(path).read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return path
    for line in reversed(lines):
        stripped = line.strip()
        if not stripped.startswith('"'):
            continue
        quoted = stripped[1:].split('"', 1)[0]
        resolved = Path(quoted.replace("%dp0%", str(Path(path).parent)))
        if resolved.is_file():
            return str(resolved)
    return path


def find(configured: str = "") -> str:
    """The binary to run. `configured` wins; otherwise `PATH`, then the usual places."""
    if configured:
        if Path(configured).is_file():
            return through_shim(configured)
        msg = f"CLYDE_CLAUDE_BINARY points at {configured}, which is not a file"
        raise ClaudeNotFoundError(msg)
    on_path = shutil.which(BINARY)
    if on_path:
        return through_shim(on_path)
    for candidate in candidates():
        if candidate.is_file():
            return through_shim(str(candidate))
    msg = f"could not find `{BINARY}` on PATH or in the usual places; {INSTALL_HINT}"
    raise ClaudeNotFoundError(msg)


__all__ = [
    "BINARY",
    "INSTALL_HINT",
    "SHIM_SUFFIXES",
    "ClaudeNotFoundError",
    "candidates",
    "find",
    "through_shim",
]

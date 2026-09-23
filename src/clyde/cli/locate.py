"""Finding `claude`, and saying something useful when it is not there.

`shutil.which` is most of it. The rest is that npm's global bin is routinely absent from the
`PATH` a service manager hands a process, even when it is present in the shell where somebody
tested the command by hand -- so "works for me, missing in the service" is the common failure,
and a message naming the install command is worth more than a stack trace.
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


def find(configured: str = "") -> str:
    """The binary to run. `configured` wins; otherwise `PATH`, then the usual places."""
    if configured:
        if Path(configured).is_file():
            return configured
        msg = f"CLYDE_CLAUDE_BINARY points at {configured}, which is not a file"
        raise ClaudeNotFoundError(msg)
    on_path = shutil.which(BINARY)
    if on_path:
        return on_path
    for candidate in candidates():
        if candidate.is_file():
            return str(candidate)
    msg = f"could not find `{BINARY}` on PATH or in the usual places; {INSTALL_HINT}"
    raise ClaudeNotFoundError(msg)


__all__ = ["BINARY", "INSTALL_HINT", "ClaudeNotFoundError", "candidates", "find"]

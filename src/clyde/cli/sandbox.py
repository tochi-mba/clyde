"""The working directory every call runs in, and the reason it is empty.

Without `--bare`, Claude Code reads the *working directory* to decide what loads: hooks and
skills from `.claude/`, auto-memory from `CLAUDE.md`, servers from `.mcp.json`. A `-p` session
shows no workspace-trust dialog before doing it.

So this process never runs `claude` anywhere a person works. It owns one directory under the
system temp root, creates it empty, and checks it is still empty before each call. The check is
cheap and the failure it catches is silent: a stray `CLAUDE.md` would be folded into the prompt
ahead of the caller's own, and nothing downstream would report that it had happened.

This closes the *project-local* half of the problem only. The account's own MCP connectors load
regardless of where the process runs, and `cli/argv.py` is what removes those.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

CONTAMINANTS = (
    ".claude",
    ".claude.json",
    "CLAUDE.md",
    "CLAUDE.local.md",
    "AGENTS.md",
    ".mcp.json",
    ".git",
)
"""Everything that would change what a run loads. Named rather than globbed, so adding one is
a decision somebody makes in a diff."""


class ContaminatedSandboxError(RuntimeError):
    """Something appeared in the sandbox that would change what a call loads."""


class Sandbox:
    """One empty directory, owned by this process for its lifetime."""

    def __init__(self, root: Path) -> None:
        self.root = root

    @classmethod
    def create(cls, parent: Path | None = None) -> Sandbox:
        """A fresh empty directory under the temp root, or under `parent` in a test."""
        base = Path(tempfile.mkdtemp(prefix="clyde-", dir=parent))
        return cls(base)

    def contaminants(self) -> list[str]:
        """Names present that should not be. Empty is the only acceptable answer."""
        return [name for name in CONTAMINANTS if (self.root / name).exists()]

    def verify(self) -> None:
        """Raise if anything would change what the next call loads."""
        found = self.contaminants()
        if found:
            msg = (
                f"the sandbox at {self.root} contains {', '.join(found)}, which Claude Code "
                "would load into the prompt; refusing to run there"
            )
            raise ContaminatedSandboxError(msg)


__all__ = ["CONTAMINANTS", "ContaminatedSandboxError", "Sandbox"]

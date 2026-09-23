"""The working directory is empty, and stays empty.

This is the cheap half of the isolation problem. Without `--bare`, Claude Code reads the
working directory for hooks, memory and MCP config, and a `-p` run shows no trust dialog
before doing it. A stray `CLAUDE.md` would be folded into the caller's prompt and nothing
downstream would say so -- which is exactly the failure a test is for.

The expensive half is the account's own connectors, which no directory can affect. That is
`cli/argv.py`'s `--strict-mcp-config`, and `test_argv.py` covers it.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from clyde.cli.sandbox import CONTAMINANTS, ContaminatedSandboxError, Sandbox


def test_a_fresh_sandbox_is_empty(tmp_path: Path) -> None:
    sandbox = Sandbox.create(tmp_path)
    assert sandbox.root.is_dir()
    assert sandbox.contaminants() == []
    sandbox.verify()


def test_two_sandboxes_do_not_share_a_directory(tmp_path: Path) -> None:
    assert Sandbox.create(tmp_path).root != Sandbox.create(tmp_path).root


@pytest.mark.parametrize("name", CONTAMINANTS)
def test_each_contaminant_is_refused(tmp_path: Path, name: str) -> None:
    sandbox = Sandbox.create(tmp_path)
    target = sandbox.root / name
    if name.endswith((".md", ".json")):
        target.write_text("anything", encoding="utf-8")
    else:
        target.mkdir()
    assert sandbox.contaminants() == [name]
    with pytest.raises(ContaminatedSandboxError, match=name.replace(".", r"\.")):
        sandbox.verify()


def test_the_refusal_names_every_contaminant_it_found(tmp_path: Path) -> None:
    sandbox = Sandbox.create(tmp_path)
    (sandbox.root / "CLAUDE.md").write_text("x", encoding="utf-8")
    (sandbox.root / ".mcp.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ContaminatedSandboxError) as raised:
        sandbox.verify()
    assert "CLAUDE.md" in str(raised.value)
    assert ".mcp.json" in str(raised.value)


def test_an_unrelated_file_is_not_a_contaminant(tmp_path: Path) -> None:
    """Claude Code loads named things. A scratch file it never reads is not a problem, and
    refusing on one would make this unusable."""
    sandbox = Sandbox.create(tmp_path)
    (sandbox.root / "notes.txt").write_text("x", encoding="utf-8")
    sandbox.verify()


def test_the_list_names_what_claude_code_actually_loads() -> None:
    """A regression guard: each of these corresponds to something documented as loaded from
    the working directory. Removing one should be a decision, not an edit."""
    assert {".claude", "CLAUDE.md", ".mcp.json"} <= set(CONTAMINANTS)

"""Settings, and the crash that names a misspelled variable.

`extra="forbid"` catches a typo in a name pydantic knows about. `check_for_unknown_env_vars`
catches the other half: a `CLYDE_*` nothing declares at all, which would otherwise sit in the
environment looking like it was doing something.
"""

from __future__ import annotations

from typing import Any

import pytest

from clyde.core.config import PREFIX, Settings, check_for_unknown_env_vars, load_settings


def test_the_defaults_are_the_safe_ones() -> None:
    settings = Settings()
    assert settings.host == "127.0.0.1", "binding anywhere else exposes a logged-in Claude Code"
    assert settings.default_model == "sonnet", "the cheap one is the polite default"
    assert settings.max_concurrent >= 1


def test_a_declared_variable_is_read(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(f"{PREFIX}PORT", "9000")
    assert load_settings({f"{PREFIX}PORT": "9000"}).port == 9000


def test_an_undeclared_variable_is_named() -> None:
    assert check_for_unknown_env_vars({f"{PREFIX}NOT_A_SETTING": "x"}) == [f"{PREFIX}NOT_A_SETTING"]


def test_declared_variables_are_not_flagged() -> None:
    assert check_for_unknown_env_vars({f"{PREFIX}PORT": "1", "PATH": "/usr/bin"}) == []


def test_loading_refuses_to_start_on_an_unknown_variable() -> None:
    """Silently ignoring it is how a setting appears to be applied and is not."""
    with pytest.raises(RuntimeError, match=f"{PREFIX}TYPO"):
        load_settings({f"{PREFIX}TYPO": "x"})


def test_the_environment_is_read_when_none_is_passed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(f"{PREFIX}ALSO_NOT_A_SETTING", "x")
    assert f"{PREFIX}ALSO_NOT_A_SETTING" in check_for_unknown_env_vars()


@pytest.mark.parametrize("field", ["timeout_seconds", "max_concurrent", "stdin_limit_bytes"])
def test_the_bounded_numbers_reject_nonsense(field: str) -> None:
    values: dict[str, Any] = {field: 0}
    with pytest.raises(ValueError, match=field):
        Settings(**values)

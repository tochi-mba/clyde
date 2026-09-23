"""Finding the binary, and turning a failure into a status code.

The error mapping carries the weight here. A caller retries a 503 and does not retry a 403, so
putting "this machine has no claude" and "you are not logged in" in the same bucket produces
either a retry storm against something that will never work, or a real outage nobody retries.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from clyde.cli.locate import BINARY, ClaudeNotFoundError, candidates, find
from clyde.cli.run import ClaudeFailedError, ClaudeTimeoutError
from clyde.cli.sandbox import ContaminatedSandboxError
from clyde.openai.errors import Problem, from_error, needs_login

# --- locating ---------------------------------------------------------------------------------


def test_a_configured_binary_that_exists_wins(tmp_path: Path) -> None:
    binary = tmp_path / "claude"
    binary.write_text("", encoding="utf-8")
    assert find(str(binary)) == str(binary)


def test_a_configured_binary_that_does_not_exist_is_an_error(tmp_path: Path) -> None:
    with pytest.raises(ClaudeNotFoundError, match="which is not a file"):
        find(str(tmp_path / "nowhere"))


def test_the_path_is_used_when_nothing_is_configured(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("clyde.cli.locate.shutil.which", lambda _: "/usr/local/bin/claude")
    assert find() == "/usr/local/bin/claude"


def test_a_known_location_is_tried_when_the_path_misses(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """npm's global bin is routinely absent from a service manager's PATH."""
    found = tmp_path / "claude"
    found.write_text("", encoding="utf-8")
    monkeypatch.setattr("clyde.cli.locate.shutil.which", lambda _: None)
    monkeypatch.setattr("clyde.cli.locate.candidates", lambda: [tmp_path / "no", found])
    assert find() == str(found)


def test_nothing_anywhere_names_the_install_command(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("clyde.cli.locate.shutil.which", lambda _: None)
    monkeypatch.setattr("clyde.cli.locate.candidates", list)
    with pytest.raises(ClaudeNotFoundError, match="npm install -g"):
        find()


def test_the_candidate_list_covers_windows_when_appdata_is_set(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("APPDATA", r"C:\Users\someone\AppData\Roaming")
    assert any("npm" in str(path) for path in candidates())


def test_the_candidate_list_survives_no_appdata(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("APPDATA", raising=False)
    assert all(BINARY in str(path) for path in candidates())


# --- the error mapping -------------------------------------------------------------------------


def test_a_missing_binary_is_retryable_with_a_delay() -> None:
    problem = from_error(ClaudeNotFoundError("nowhere"))
    assert (problem.status, problem.code, problem.retry_after) == (503, "claude_not_found", 30)


def test_a_contaminated_sandbox_is_this_services_fault() -> None:
    """Not the caller's: they asked correctly and this machine refused to run."""
    assert from_error(ContaminatedSandboxError("dirty")).status == 500


def test_a_timeout_is_retryable() -> None:
    problem = from_error(ClaudeTimeoutError("too slow"))
    assert (problem.status, problem.retry_after) == (504, 1)


def test_a_generic_failure_is_a_bad_gateway() -> None:
    assert from_error(ClaudeFailedError("exited 2")).status == 502


@pytest.mark.parametrize(
    "message",
    ["Not logged in", "please run /login", "Authentication failed", "invalid api key"],
)
def test_a_login_problem_is_never_retried(message: str) -> None:
    """Retrying a logged-out CLI forever is the failure this case exists to prevent."""
    problem = from_error(ClaudeFailedError(message))
    assert (problem.status, problem.code, problem.retry_after) == (403, "not_logged_in", 0)
    assert "sign in" in problem.message


def test_a_bad_request_is_the_callers_to_fix() -> None:
    assert from_error(ValueError("too big")).status == 400


def test_an_unrecognised_error_names_its_type_and_nothing_else() -> None:
    """A message can carry a response body; a type name cannot."""
    problem = from_error(RuntimeError("sk-ant-secret-in-the-message"))
    assert problem.status == 500
    assert "sk-ant" not in problem.message
    assert "RuntimeError" in problem.message


def test_needs_login_is_case_insensitive() -> None:
    assert needs_login("NOT LOGGED IN")
    assert not needs_login("everything is fine")


def test_the_body_is_openais_error_envelope() -> None:
    body = Problem(400, "nope", code="invalid_request_error").body()
    assert body == {
        "error": {
            "message": "nope",
            "type": "invalid_request_error",
            "code": "invalid_request_error",
        }
    }

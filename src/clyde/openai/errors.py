"""A failed run, as a status code and a sentence.

The mapping is the whole point of the file. A caller retries a 503 and does not retry a 400,
so putting a missing binary and a malformed request in the same bucket means either a retry
storm against something that will never work, or a real outage that nobody retries.

`ModelUnavailableError` on the other side of this is retryable and falls back to another
provider; a refusal is not. So: anything about *this machine* is 503, anything about *this
request* is 400, and the one case where the CLI declined on its own becomes a 403 so it is
never retried.
"""

from __future__ import annotations

from dataclasses import dataclass

from clyde.cli.locate import ClaudeNotFoundError
from clyde.cli.run import ClaudeFailedError, ClaudeTimeoutError
from clyde.cli.sandbox import ContaminatedSandboxError

NOT_LOGGED_IN = ("not logged in", "please run /login", "authentication", "invalid api key")
"""Phrases the CLI uses when the login is the problem. Matched case-insensitively against its
own result text, which is the only place it says so in `-p` mode."""


@dataclass(frozen=True, slots=True)
class Problem:
    """An HTTP status and a message, ready to become a body.

    A value rather than an exception, because it is only ever constructed and returned. Making
    it raisable would invite a second error path through code that already has one.
    """

    status: int
    message: str
    code: str
    retry_after: int = 0

    def body(self) -> dict[str, object]:
        """OpenAI's error envelope, which is what a chat-completions client expects."""
        return {"error": {"message": self.message, "type": self.code, "code": self.code}}


def needs_login(text: str) -> bool:
    return any(phrase in text.lower() for phrase in NOT_LOGGED_IN)


def _failed(error: ClaudeFailedError) -> Problem:
    """A run that started and did not finish well. Login is its own case: retrying it is
    pointless, and a caller that treats it as an outage will retry it forever."""
    message = str(error)
    if needs_login(message):
        return Problem(
            403,
            f"{message}. Run `claude` once in a terminal and sign in.",
            code="not_logged_in",
        )
    return Problem(502, message, code="claude_failed")


def from_error(error: Exception) -> Problem:
    """The one place a failure becomes a status code."""
    if isinstance(error, ClaudeNotFoundError):
        # The machine is not set up. Retryable in principle -- somebody may install it -- and
        # a `Retry-After` keeps a caller from hammering it in the meantime.
        return Problem(503, str(error), code="claude_not_found", retry_after=30)
    if isinstance(error, ContaminatedSandboxError):
        # Refusing to run is the correct outcome, and it is this service's fault, not the
        # caller's, so it is a 500 rather than a 400.
        return Problem(500, str(error), code="sandbox_contaminated")
    if isinstance(error, ClaudeTimeoutError):
        return Problem(504, str(error), code="timeout", retry_after=1)
    if isinstance(error, ClaudeFailedError):
        return _failed(error)
    if isinstance(error, ValueError):
        return Problem(400, str(error), code="invalid_request_error")
    return Problem(500, f"clyde failed: {type(error).__name__}", code="internal_error")


__all__ = ["NOT_LOGGED_IN", "Problem", "from_error", "needs_login"]

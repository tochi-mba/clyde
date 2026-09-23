"""Every knob, declared, with `CLYDE_` in front of it.

`extra="forbid"` plus :func:`check_for_unknown_env_vars` means a misspelled `CLYDE_*` is a
startup crash rather than a setting that silently did nothing. That matters more here than in
most services: the flags this process builds decide what a model is allowed to reach, and a
typo in one of them is the difference between zero tools and a live Gmail connector.
"""

from __future__ import annotations

import os

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

PREFIX = "CLYDE_"


class Settings(BaseSettings):
    """Read once at startup and held. Nothing here is re-read per request."""

    model_config = SettingsConfigDict(env_prefix=PREFIX, extra="forbid", env_file=".env")

    host: str = "127.0.0.1"
    """Loopback by default, and it should stay there. This process runs arbitrary prompts
    through a logged-in Claude Code; binding it to a public interface hands that to whoever
    finds the port."""

    port: int = 8127

    log_level: str = "INFO"
    log_format: str = "json"

    claude_binary: str = ""
    """Where `claude` lives. Empty means look on `PATH`, which is right on a developer's
    machine and wrong in a service manager whose `PATH` is not yours."""

    default_model: str = "sonnet"
    """Used when a caller names no model. `sonnet` rather than `opus` because this sits on a
    subscription and the cheap one is the polite default."""

    timeout_seconds: float = Field(default=600.0, gt=0)
    """A ceiling on one call. Generous, because a real turn with thinking is not fast, and a
    timeout here surfaces as a retryable error rather than a wrong answer."""

    max_concurrent: int = Field(default=2, ge=1)
    """How many `claude` processes may run at once. Small on purpose: each one is a whole
    Node runtime, and twenty of them help nobody."""

    stdin_limit_bytes: int = Field(default=9_000_000, gt=0)
    """Below the CLI's documented 10 MB piped-stdin cap, so an oversized conversation is
    refused here with a clear message rather than by the subprocess with an opaque one."""


def check_for_unknown_env_vars(environ: dict[str, str] | None = None) -> list[str]:
    """`CLYDE_*` names nothing declares. Returned rather than raised, so a caller decides."""
    known = {f"{PREFIX}{name}".upper() for name in Settings.model_fields}
    seen = environ if environ is not None else dict(os.environ)
    return sorted(name for name in seen if name.startswith(PREFIX) and name.upper() not in known)


def load_settings(environ: dict[str, str] | None = None) -> Settings:
    """Settings, or a crash naming exactly which variable nobody reads."""
    unknown = check_for_unknown_env_vars(environ)
    if unknown:
        msg = f"unknown environment variables: {', '.join(unknown)}"
        raise RuntimeError(msg)
    return Settings()


__all__ = ["PREFIX", "Settings", "check_for_unknown_env_vars", "load_settings"]

"""Where this service's own log lines go.

Without this, `logging.getLogger("clyde")` has no handler and nothing above it does either, so
Python falls back to `logging.lastResort` -- which emits `WARNING` and above and silently drops
everything else. The symptom is specific and misleading: refusals appeared in the log because
they are warnings, and the line saying *why* a call had been rearranged did not, because it is
an `INFO`. Debugging then looks like reading a log that is telling you everything it knows.

`Settings.log_level` and `Settings.log_format` have existed since the first commit and were
read by nothing. This is what reads them.
"""

from __future__ import annotations

import json
import logging
import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import TextIO

NAME = "clyde"
"""The one logger this service writes to. Everything under it inherits the handler."""


class JsonFormatter(logging.Formatter):
    """One line of JSON per record, for somewhere that collects logs.

    Only the fields a reader acts on. Deliberately not the whole `LogRecord`: this service
    handles other people's prompts, and a formatter that reaches for every attribute it can
    find is one refactor away from writing one into a log.
    """

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            payload["error"] = self.formatException(record.exc_info)
        return json.dumps(payload, separators=(",", ":"))


def configure(
    level: str = "INFO", form: str = "json", stream: TextIO | None = None
) -> logging.Logger:
    """Attach one handler to the `clyde` logger, at most once.

    Idempotent because the app factory is called per app and the tests call it many times;
    a handler added each time would multiply every line by the number of apps ever built.

    Idempotent on *this* handler rather than on "are there any handlers", because there are
    other people's: pytest's `caplog.at_level(logger="clyde")` attaches its own and leaves it
    there, so a "no handlers yet" guard quietly decided the service needed none. Naming the
    handler and looking for that name is the difference between "already configured" and
    "somebody else is also listening".

    `stream` exists so a test can read what was written without going through pytest's
    capture, which is process-wide and shared with every other test that ever configured
    this logger.
    """
    logger = logging.getLogger(NAME)
    logger.setLevel(level.upper())
    if not any(handler.name == NAME for handler in logger.handlers):
        handler = logging.StreamHandler(stream if stream is not None else sys.stderr)
        handler.name = NAME
        handler.setFormatter(
            JsonFormatter() if form == "json" else logging.Formatter("%(levelname)s: %(message)s")
        )
        logger.addHandler(handler)
    # The root logger belongs to whoever is hosting this; writing there twice is their problem
    # to discover, so this one stops here.
    logger.propagate = False
    return logger


__all__ = ["NAME", "JsonFormatter", "configure"]

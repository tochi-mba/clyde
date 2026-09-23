"""Whether this service's own log lines reach anywhere.

Every test here exists because they did not. With no handler on `clyde` and none above it,
Python falls back to `logging.lastResort`, which emits `WARNING` and above and drops the rest
without a word. Refusals showed up; the line explaining why a call had been rearranged did not.
Debugging then looks like reading a log that is telling you everything it knows.

These read a stream they own rather than pytest's capture. A logger is process-wide state
shared with every other test that ever built an app, and going through the capture plugin made
an earlier version of this file pass alone and fail in a suite.
"""

from __future__ import annotations

import io
import json
import logging
import sys
from collections.abc import Iterator

import pytest

from clyde.core.logs import NAME, JsonFormatter, configure


@pytest.fixture(autouse=True)
def _isolated() -> Iterator[None]:
    """Each test gets the logger as it is at import, and leaves it that way."""
    logger = logging.getLogger(NAME)
    handlers, level, propagate = logger.handlers[:], logger.level, logger.propagate
    logger.handlers.clear()
    yield
    logger.handlers[:] = handlers
    logger.setLevel(level)
    logger.propagate = propagate


def written(level: str = "INFO", form: str = "json") -> tuple[logging.Logger, io.StringIO]:
    stream = io.StringIO()
    return configure(level, form, stream), stream


def test_an_info_line_is_not_dropped() -> None:
    """The whole point. `lastResort` swallows this one, which is how three real failures were
    diagnosed by guessing instead of by reading."""
    logger, stream = written("INFO", "text")
    logger.info("fitted: 55716 -> 24732 chars")
    assert "fitted: 55716 -> 24732 chars" in stream.getvalue()


def test_json_format_is_one_object_per_line() -> None:
    logger, stream = written()
    logger.warning("refused: 502 claude_failed: boom")
    assert json.loads(stream.getvalue().strip()) == {
        "level": "WARNING",
        "logger": NAME,
        "message": "refused: 502 claude_failed: boom",
    }


def test_the_level_is_honoured() -> None:
    logger, stream = written("WARNING", "text")
    logger.info("not this one")
    logger.warning("but this one")
    assert "not this one" not in stream.getvalue()
    assert "but this one" in stream.getvalue()


def test_configuring_twice_does_not_double_every_line() -> None:
    """The app factory runs per app, and the tests build many."""
    logger, stream = written("INFO", "text")
    configure("INFO", "text", stream)
    logger.info("once")
    assert stream.getvalue().count("once") == 1
    assert sum(handler.name == NAME for handler in logger.handlers) == 1


def boom() -> None:
    msg = "planted"
    raise ValueError(msg)


def test_an_exception_is_carried_as_a_field() -> None:
    logger, stream = written()
    try:
        boom()
    except ValueError:
        logger.exception("it failed")
    record = json.loads(stream.getvalue().strip())
    assert record["message"] == "it failed"
    assert "planted" in record["error"]


def test_nothing_is_written_to_the_root_logger() -> None:
    """Whoever hosts this owns the root logger; propagating would duplicate every line into
    their handlers as well as ours."""
    logger, _ = written()
    assert logger.propagate is False


def test_the_default_stream_is_stderr() -> None:
    """Passing a stream is a testing seam, not the production path."""
    logger = configure("INFO", "json")
    handler = next(h for h in logger.handlers if h.name == NAME)
    assert isinstance(handler, logging.StreamHandler)
    assert handler.stream is sys.stderr


def test_somebody_elses_handler_does_not_count_as_ours() -> None:
    """pytest attaches a `LogCaptureHandler` to this very logger and leaves it there. A guard
    that asked "are there any handlers" then decided the service needed none, and every INFO
    line this repo added went nowhere for the rest of the suite."""
    logger = logging.getLogger(NAME)
    logger.addHandler(logging.NullHandler())
    _, stream = written("INFO", "text")
    logger.info("still ours")
    assert "still ours" in stream.getvalue()


def test_the_formatter_reports_only_the_fields_a_reader_acts_on() -> None:
    """A formatter that reaches for every `LogRecord` attribute is one refactor away from
    writing somebody's prompt into a log."""
    record = logging.LogRecord(NAME, logging.INFO, "f.py", 1, "hello %s", ("there",), None)
    assert set(json.loads(JsonFormatter().format(record))) == {"level", "logger", "message"}
    assert json.loads(JsonFormatter().format(record))["message"] == "hello there"

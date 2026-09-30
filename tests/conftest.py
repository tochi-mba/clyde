"""Suite-wide pytest hooks."""

from __future__ import annotations

import pytest

NOTHING_LIVE = "no tests are marked live yet, so there was nothing to run against the real CLI"


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    """Let a run of the live tests that selects none succeed, and say why it ran nothing.

    pytest calls a run that selects no test exit status 5. For `make live` that is not a
    failure: nothing is marked `live` yet. Any other run that selects nothing -- a mistyped
    `-k`, a path with no tests in it -- keeps its 5.
    """
    if exitstatus != pytest.ExitCode.NO_TESTS_COLLECTED:
        return
    if session.config.getoption("markexpr") != "live":
        return
    session.exitstatus = pytest.ExitCode.OK
    reporter = session.config.pluginmanager.get_plugin("terminalreporter")
    if reporter is not None:
        reporter.write_line(NOTHING_LIVE)

"""`make live` with nothing to run says so and succeeds.

`make live` is `pytest -m live`, and no test is marked `live` yet. pytest calls a run that
selects nothing exit status 5, so the target failed on a clean checkout -- a red verb that
means "nothing was wrong". A run of the live tests that selects none now exits 0 and says why.

This runs pytest in a child process, collecting only: selecting `live` deselects every test
here, so nothing spawns a real `claude`.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_a_live_run_with_no_live_tests_succeeds_and_says_so() -> None:
    finished = subprocess.run(
        [sys.executable, "-m", "pytest", "-m", "live", "--no-cov", "-p", "no:cacheprovider"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )

    assert finished.returncode == 0, finished.stdout + finished.stderr
    assert "no tests are marked live" in finished.stdout

"""Fixtures and reporting for the SPEC harness.

* ``gui``: a shown ``MainWindow`` over a session the test feeds event by event.
* ``commands`` / ``commands_without_window``: a command context after epoch 1 of 5.
* benchmark-marked tests are skipped unless ``HT_RUN_BENCHMARKS=1``.
* After the run, a SPEC coverage table (per requirement: passed / failed / skipped) and the
  list of aspects that need manual verification are printed.
"""

from __future__ import annotations

import contextlib
import os
import time
from collections import Counter

import pytest

from .helpers import (
    CommandEnv,
    GuiDriver,
    GuiSession,
    build_process,
    make_command_env,
    session_started,
    window_process_children,
)
from .requirements import MANUAL_CHECKS, REQUIREMENTS

_SPEC_OF: dict[str, tuple[str, ...]] = {}
_OUTCOMES: dict[str, str] = {}


# ------------------------------------------------------------------------ fixtures
@pytest.fixture
def gui(qtbot) -> GuiSession:
    """A shown window displaying the start of a 5-epoch session of a small network."""
    from hollywood_training.commands import create_registry
    from hollywood_training.gui import MainWindow
    from hollywood_training.sync import SessionState

    bundle = build_process(epochs=5, normalization="z-score")
    tracker, started = session_started(bundle.process)
    session = SessionState()
    window = MainWindow(session, completer=create_registry().complete)
    qtbot.addWidget(window)
    with qtbot.waitExposed(window):  # mapped, so the synchronous renders below happen
        window.show()
    payload = GuiSession(window, session, tracker, bundle, started)
    payload.push(started)
    return payload


@pytest.fixture
def drive(qapp):
    """Start a scripted user of the in-process window (``GuiDriver``). Every driver is
    stopped at teardown, so a failed test cannot leave one typing into a later window."""
    drivers: list[GuiDriver] = []

    def start(steps, **options) -> GuiDriver:
        driver = GuiDriver(steps, **options)
        drivers.append(driver)
        return driver

    yield start
    for driver in drivers:
        driver.stop()


@pytest.fixture
def commands() -> CommandEnv:
    return make_command_env(frontend=True)


@pytest.fixture
def commands_without_window() -> CommandEnv:
    return make_command_env(frontend=False)


@pytest.fixture(autouse=True)
def _no_stray_window_processes():
    """Window processes must not outlive the test that started them (killed if they do,
    so one failure cannot disturb the following tests)."""
    yield
    deadline = time.monotonic() + 10.0
    while window_process_children() and time.monotonic() < deadline:
        time.sleep(0.1)
    for child in window_process_children():
        with contextlib.suppress(Exception):
            child.kill()


# ---------------------------------------------------------------------- collection
def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    run_benchmarks = os.environ.get("HT_RUN_BENCHMARKS") == "1"
    skip_benchmark = pytest.mark.skip(reason="benchmark: set HT_RUN_BENCHMARKS=1 to run it")
    for item in items:
        ids = tuple(str(arg) for marker in item.iter_markers(name="spec") for arg in marker.args)
        if ids:
            _SPEC_OF[item.nodeid] = ids
        if not run_benchmarks and item.get_closest_marker("benchmark") is not None:
            item.add_marker(skip_benchmark)


def pytest_runtest_logreport(report: pytest.TestReport) -> None:
    if report.nodeid not in _SPEC_OF:
        return
    if report.when == "call" or (report.when == "setup" and not report.passed):
        outcome = report.outcome
        if hasattr(report, "wasxfail"):
            outcome = "xfailed" if report.skipped else "xpassed"
        _OUTCOMES[report.nodeid] = outcome
    elif report.when == "teardown" and report.failed:
        _OUTCOMES[report.nodeid] = "failed"


def pytest_terminal_summary(terminalreporter, exitstatus: int, config: pytest.Config) -> None:
    if not _OUTCOMES:
        return
    try:
        per_requirement: dict[str, Counter] = {key: Counter() for key in REQUIREMENTS}
        for nodeid, outcome in _OUTCOMES.items():
            for requirement in _SPEC_OF.get(nodeid, ()):
                per_requirement.setdefault(requirement, Counter())[outcome] += 1
        terminalreporter.write_sep("=", "SPEC requirement coverage (tests/spec)")
        for requirement, counts in per_requirement.items():
            summary = ", ".join(f"{counts[k]} {k}" for k in sorted(counts)) or "not run"
            title = REQUIREMENTS.get(requirement, "unknown requirement id")
            terminalreporter.write_line(f"{requirement:<4} {summary:<34} {title[:70]}")
        terminalreporter.write_sep("-", "needs manual verification (not fully automatable)")
        for requirement, text in MANUAL_CHECKS.items():
            terminalreporter.write_line(f"{requirement:<4} {text}")
    except Exception as exc:  # reporting must never break a test run
        terminalreporter.write_line(f"(SPEC coverage summary unavailable: {exc!r})")


@pytest.fixture
def window_process(qtbot):
    """The window process's GUI side, in this process, connected to the test by a pipe."""
    from .helpers import WindowProcessHarness

    harness = WindowProcessHarness(qtbot)
    harness.start()
    yield harness
    harness.shutdown()

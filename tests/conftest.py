"""SPEC harness for hollywood-training: the requirements of ``specs/R1.md`` and ``specs/R2.md``.

* The environment is fixed before Qt or torch load (``htspec.env``): offscreen Qt on a
  1920 x 1080 virtual screen, no access to a real display, temporary files in ``.temp/``.
* Every test names the requirement(s) it verifies with ``@pytest.mark.req(...)``; a test
  without one, or with an unknown id, is a collection error. The session ends with a
  requirement coverage table (also written to ``.temp/tests/reports/``). Tests that verify
  an R1 requirement count for R2 requirement 5 as well ("all other R1 features not affected").

Run with ``tests/run_tests.sh`` (see ``tests/README.md``).
"""

from __future__ import annotations

import json
import sys
import time
import warnings
from collections import defaultdict
from pathlib import Path

HARNESS_DIR = Path(__file__).resolve().parent
if str(HARNESS_DIR) not in sys.path:
    sys.path.insert(0, str(HARNESS_DIR))

from htspec import env as _env  # noqa: E402

_env.configure_current_process()

import pytest  # noqa: E402
from htspec import paths  # noqa: E402

# ------------------------------------------------------------------ requirements
REQUIREMENTS: dict[str, str] = {
    "R1.1": "track the network architecture, including the weights of each layer",
    "R1.2": "manage the training: native PyTorch definition, wrapped, passed to the manager",
    "R1.3": "requirements 1 and 2 do not significantly change the native PyTorch workflow",
    "R1.4": "native GUI window (not a web page) with smooth animations of the real training",
    "R1.5": "training synchronised with the GUI: updated every epoch, before the next epoch",
    "R1.6": "training runs as fast as the synchronisation allows",
    "R1.7": "well-defined commands control the training at runtime",
    "R1.8": "no 'dark magic' Python methods",
    "R1.9": "completely decomposed into modules with a clear architecture",
    "R1.10": "all package files in the package folder hollywood_training",
    "R1.11": "a demo with an illustrative network and dataset, in demos/",
    "R1.12": "a README.md for quick start",
    "R1.13": "a wiki folder: a page for every component and the overall architecture",
    "R1.G1": "GUI: architecture, neuron connections, real weights changing every epoch",
    "R1.G2": "GUI: training status (epochs, batch size, lr, loss, optimizer, ..., timer)",
    "R1.G3": "GUI: training and validation loss drawn dynamically every epoch",
    "R1.G4": "GUI: memory, CPU and (when applicable) GPU usage",
    "R1.G5": "GUI: an integrated terminal controls the training with the commands",
    "R1.G6": "GUI: the name of the training network (its class name)",
    "R1.G7": "GUI: blue based",
    "R2.1": "no mouse pointer displayed anywhere inside the window",
    "R2.2": "all mouse-based interactions disabled within the application",
    "R2.3": "all implemented functionality accessible and operational without the mouse",
    "R2.4": "additional commands may replace the mouse-based interactions",
    "R2.5": "all other R1 features not affected (every R1 test counts here as well)",
    "R2.6": "pointer and mouse interactions outside the window not affected",
    "R2.7": "all keyboard input rerouted to the internal terminal while on the window",
}

MARKERS = (
    "req(*ids): the SPEC requirement(s) a test verifies, e.g. req('R1.5', 'R2.5')",
    "subprocess: starts Python subprocesses (window process, demos, scenarios)",
    "slow: takes more than a few seconds",
)


def pytest_configure(config: pytest.Config) -> None:
    for line in MARKERS:
        config.addinivalue_line("markers", line)


def requirement_ids(item: pytest.Item) -> list[str]:
    ids: list[str] = []
    for mark in item.iter_markers("req"):
        ids.extend(str(arg) for arg in mark.args)
    return ids


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    problems = []
    for item in items:
        ids = requirement_ids(item)
        if not ids:
            problems.append(f"{item.nodeid}: no @pytest.mark.req(...)")
        unknown = [i for i in ids if i not in REQUIREMENTS]
        if unknown:
            problems.append(f"{item.nodeid}: unknown requirement id(s) {unknown}")
    if problems:
        raise pytest.UsageError("SPEC traceability errors:\n  " + "\n  ".join(problems))


# ------------------------------------------------------------- coverage summary
_outcomes: dict[str, str] = {}
_item_requirements: dict[str, list[str]] = {}


def pytest_itemcollected(item: pytest.Item) -> None:
    _item_requirements[item.nodeid] = requirement_ids(item)


def pytest_runtest_logreport(report: pytest.TestReport) -> None:
    previous = _outcomes.get(report.nodeid)
    if report.when == "call" or (report.when == "setup" and report.outcome != "passed"):
        outcome = report.outcome
        if hasattr(report, "wasxfail"):
            outcome = "xfailed" if report.skipped else "xpassed"
        _outcomes[report.nodeid] = outcome
    elif report.when == "teardown" and report.failed and previous == "passed":
        _outcomes[report.nodeid] = "error"


def pytest_terminal_summary(terminalreporter, exitstatus, config) -> None:
    if not _outcomes:
        return
    table: dict[str, dict[str, int]] = {rid: defaultdict(int) for rid in REQUIREMENTS}
    failing: dict[str, list[str]] = defaultdict(list)
    for nodeid, outcome in _outcomes.items():
        ids = set(_item_requirements.get(nodeid, []))
        if any(i.startswith("R1.") for i in ids):
            ids.add("R2.5")  # R2.5: every R1 feature must still work on this version
        for rid in ids:
            table[rid][outcome] += 1
            if outcome in ("failed", "error"):
                failing[rid].append(nodeid)
    write = terminalreporter.write_line
    terminalreporter.section("SPEC requirement coverage (specs/R1.md, specs/R2.md)")
    write(f"{'req':<7} {'pass':>5} {'fail':>5} {'skip':>5}     requirement")
    for rid, text in REQUIREMENTS.items():
        counts = table[rid]
        failed = counts["failed"] + counts["error"]
        mark = "  " if failed == 0 and counts["passed"] else "!!" if failed else "--"
        write(f"{rid:<7} {counts['passed']:>5} {failed:>5} {counts['skipped']:>5}  {mark} {text}")
    paths.REPORTS.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    report = {
        "requirements": REQUIREMENTS,
        "coverage": {rid: dict(counts) for rid, counts in table.items()},
        "failing": failing,
        "outcomes": _outcomes,
        "requirements_of": _item_requirements,
    }
    target = paths.REPORTS / f"coverage-{stamp}.json"
    target.write_text(json.dumps(report, indent=2))
    write(f"coverage report: {target}")


# ---------------------------------------------------------------------- fixtures
@pytest.fixture(autouse=True)
def _stop_leftover_runs():
    """Safety net: a ``HeadlessRun`` left training by a failed test is stopped."""
    yield
    from htspec.headless import stop_leftover_runs

    if stop_leftover_runs():
        warnings.warn("a training run still ran after the test and was stopped", RuntimeWarning,
                      stacklevel=1)  # fmt: skip


@pytest.fixture
def make_rig(qtbot):
    """Factory for :class:`htspec.rig.GuiRig` (an offscreen window + the real commands)."""
    from htspec.rig import GuiRig

    def factory(**kwargs):
        return GuiRig(qtbot, **kwargs)

    return factory


@pytest.fixture
def rig(make_rig):
    """A shown, active window whose session has started (10 epochs announced)."""
    instance = make_rig()
    instance.feeder.start()
    instance.process_events()
    return instance

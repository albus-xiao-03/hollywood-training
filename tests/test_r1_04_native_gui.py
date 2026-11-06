"""R1 requirement 4 - "The package **MUST** provide a native GUI window, **NOT** a Web page.
The GUI **MUST** provide smooth animations to demonstrate the **REAL** training process."

Native, not a web page:

* the window is a native Qt top-level window (``QMainWindow`` with its own native window),
  in-process and in the window process that ``ht.run`` starts by default;
* no web technology anywhere: no web engine / web channel / HTTP server / browser module in
  the training process or in the window process, no listening network socket, no such import
  in the package's source.

Smooth animations of the real process:

* the animation clock ticks at a smooth frame rate and every panel repaints continuously,
  frame after frame, in every training state;
* the isolated window keeps animating while a long training batch computes; the in-process
  window animates between batches;
* what is animated is the real session: the state badge follows the real state (the drawn
  weights following the real weights are verified in ``test_r1_05_synchronization.py``).
"""

from __future__ import annotations

import ast
import contextlib
import os
import sys
import time

import numpy as np
import psutil
import pytest
from htspec import paths
from htspec.models import make_setup
from htspec.probe_log import LINGER_CLOSE
from PySide6.QtCore import Qt
from PySide6.QtGui import QWindow
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QMainWindow

import hollywood_training as ht
from hollywood_training.sync.events import TrainingState

pytestmark = pytest.mark.req("R1.4")

WEB_MODULES = (
    "PySide6.QtWebEngineCore", "PySide6.QtWebEngineWidgets", "PySide6.QtWebEngineQuick",
    "PySide6.QtWebChannel", "PySide6.QtWebSockets", "PySide6.QtWebView", "PySide6.QtHttpServer",
    "http.server", "wsgiref.simple_server", "webbrowser", "flask", "tornado", "aiohttp",
    "streamlit", "gradio", "dash", "bokeh.server", "fastapi", "uvicorn", "django",
)  # fmt: skip


def image_of(widget) -> np.ndarray:
    image = widget.grab().toImage()
    width, height = image.width(), image.height()
    buffer = np.frombuffer(image.constBits(), np.uint8, count=image.sizeInBytes())
    return buffer.reshape(height, image.bytesPerLine())[:, : width * 4].copy()


# --------------------------------------------------------------------- native
def test_the_window_is_a_native_top_level_qt_window(rig):
    window = rig.window
    assert isinstance(window, QMainWindow) and window.isWindow()
    assert window.windowFlags() & Qt.WindowType.Window
    assert isinstance(window.windowHandle(), QWindow)
    assert int(window.winId()) != 0
    assert window.isVisible()


def test_a_real_in_process_run_shows_a_native_window(pilot):
    seen = {}

    def look(window) -> None:
        seen["native"] = isinstance(window, QMainWindow) and window.windowHandle() is not None
        seen["visible"] = window.isVisible()

    pilot.on_epoch(1, look)
    pilot.on_finish(pilot.close)
    pilot.run(make_setup(epochs=2).process)
    assert seen == {"native": True, "visible": True}


@pytest.mark.subprocess
def test_the_default_run_shows_a_native_window_in_its_own_process(probe):
    log = probe(script={"steps": [LINGER_CLOSE]})
    result = ht.run(make_setup(epochs=3).process)
    assert result.epochs_completed == 3
    hello = log.records("hello")
    assert hello and hello[0]["pid"] != os.getpid()
    shown = [s for s in log.states() if s["visible"]]
    assert shown, "the window process never showed its window"
    assert all(s["win_id"] and s["platform"] for s in shown)
    assert shown[-1]["title"] == "Hollywood Training — TinyNet"


@pytest.mark.subprocess
def test_no_web_technology_is_used(probe):
    sockets: list = []

    def look_at_sockets(_batch) -> None:
        if sockets:
            return
        me = psutil.Process()
        for process in (me, *me.children(recursive=True)):
            with contextlib.suppress(psutil.Error):
                sockets.extend(
                    (process.pid, c.status, c.laddr) for c in process.net_connections("inet")
                )
        sockets.append(("checked",))

    log = probe(script={"steps": [LINGER_CLOSE]})
    setup = make_setup(epochs=3, train_hook=look_at_sockets)
    ht.run(setup.process)
    assert sockets and sockets[-1] == ("checked",)
    listening = [s for s in sockets[:-1] if s[1] == psutil.CONN_LISTEN]
    assert listening == [], f"listening network sockets: {listening}"
    loaded = [name for name in WEB_MODULES if name in sys.modules]
    assert loaded == [], f"web modules in the training process: {loaded}"
    window_modules = {name for state in log.states() for name in state["modules"]}
    assert [m for m in window_modules if m != "torch"] == [], window_modules


def test_the_package_source_imports_no_web_technology():
    found = []
    for path in sorted(paths.PACKAGE_DIR.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            names = []
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                names = [node.module] + [f"{node.module}.{alias.name}" for alias in node.names]
            for name in names:
                if any(name == web or name.startswith(f"{web}.") for web in WEB_MODULES):
                    found.append(f"{path.relative_to(paths.PROJECT_ROOT)}: {name}")
                if "QtWeb" in name or "QtHttp" in name:
                    found.append(f"{path.relative_to(paths.PROJECT_ROOT)}: {name}")
    assert found == []


# ------------------------------------------------------------------ animations
def test_the_animation_clock_ticks_at_a_smooth_rate(rig):
    arrivals: list[float] = []
    rig.window.clock.tick.connect(lambda _dt: arrivals.append(time.perf_counter()))
    started = time.perf_counter()
    QTest.qWait(1000)
    rate = len(arrivals) / (time.perf_counter() - started)
    assert rate >= 30, f"{rate:.0f} frames per second"  # the clock aims at 60
    gaps = np.diff(arrivals)
    assert float(np.median(gaps)) <= 1 / 30, f"median frame gap {np.median(gaps) * 1000:.0f} ms"


def test_every_panel_repaints_continuously(rig, paint_capture):
    """Every panel draws one of these on each frame (with a probe line and a neuron
    highlight shown); count them over one second of idling."""
    rig.feeder.epochs(2)
    assert rig.type_line("probe 1").ok and rig.type_line("neuron fc1 0").ok
    paint_capture.clear()
    started = time.perf_counter()
    QTest.qWait(1000)
    seconds = time.perf_counter() - started
    rates = {
        "HeaderBar": paint_capture.strings("HeaderBar").count("TRAINING") / seconds,
        "StatusPanel": paint_capture.strings("StatusPanel").count("PROCESS TIMER") / seconds,
        "LossChart": paint_capture.strings("LossChart").count("EPOCH 1") / seconds,
        "NetworkView": paint_capture.strings("NetworkView").count("FC1 · neuron 0") / seconds,
        "UsagePanel": paint_capture.strings("UsagePanel").count("SENSORS OFFLINE") / seconds,
    }
    assert all(rate >= 25 for rate in rates.values()), rates


@pytest.mark.parametrize(
    "state",
    [TrainingState.RUNNING, TrainingState.PAUSED, TrainingState.FINISHED, TrainingState.STOPPED],
)
def test_successive_frames_differ_in_every_training_state(rig, state):
    rig.feeder.epochs(2)
    rig.feeder.state(state, state.value)
    for panel in (rig.window.network, rig.window.header):
        first = image_of(panel)
        QTest.qWait(250)
        second = image_of(panel)
        assert not np.array_equal(first, second), f"{type(panel).__name__} did not animate"


@pytest.mark.parametrize(
    ("state", "badge"),
    [
        (TrainingState.RUNNING, "TRAINING"),
        (TrainingState.PAUSED, "PAUSED"),
        (TrainingState.FINISHED, "COMPLETE"),
        (TrainingState.STOPPED, "STOPPED"),
        (TrainingState.FAILED, "FAILURE"),
    ],
)
def test_the_animated_state_badge_shows_the_real_state(rig, paint_capture, state, badge):
    rig.feeder.epochs(1)
    rig.feeder.state(state, state.value)
    assert badge in paint_capture.render(rig.window.header)


@pytest.mark.subprocess
@pytest.mark.slow
def test_the_isolated_window_keeps_animating_during_a_long_batch(probe):
    marks: dict = {}

    def slow_batch(_batch) -> None:
        if "start" not in marks and setup.calls["train"]:
            marks["start"] = time.monotonic()
            time.sleep(1.5)
            marks["end"] = time.monotonic()

    log = probe(script={"steps": [LINGER_CLOSE]}, interval_ms=100)
    setup = make_setup(epochs=3, train_hook=slow_batch)
    ht.run(setup.process)
    during = [s for s in log.states() if marks["start"] + 0.2 < s["t"] < marks["end"] - 0.1]
    assert len(during) >= 5, "too few window snapshots during the long batch"
    first, last = during[0], during[-1]
    seconds = last["t"] - first["t"]
    for panel in ("NetworkView", "HeaderBar", "StatusPanel"):
        rate = (last["paints"].get(panel, 0) - first["paints"].get(panel, 0)) / seconds
        assert rate >= 20, f"{panel}: {rate:.0f} frames per second during the batch"


def test_the_in_process_window_animates_between_batches(pilot, paint_capture):
    """Fast batches (5 ms): the window repaints smoothly while training runs."""
    marks: dict = {}

    def mark(name):
        def record(_window) -> None:
            frames = paint_capture.strings("StatusPanel").count("PROCESS TIMER")
            marks[name] = (time.perf_counter(), frames)

        return record

    setup = make_setup(epochs=4, samples=480, train_hook=lambda _b: time.sleep(0.005))
    pilot.on_epoch(1, mark("first"))
    pilot.on_epoch(3, mark("third"))
    pilot.on_finish(pilot.close)
    pilot.run(setup.process)
    (t1, n1), (t3, n3) = marks["first"], marks["third"]
    rate = (n3 - n1) / (t3 - t1)
    assert rate >= 20, f"{rate:.0f} status-panel frames per second while training"

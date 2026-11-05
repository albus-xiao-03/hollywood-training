"""R4: a native GUI window (not a web page) with smooth animations demonstrating the
REAL training process."""

from __future__ import annotations

import ast
import sys

import numpy as np
import pytest
from PySide6.QtCore import QEvent, QObject
from PySide6.QtWidgets import QMainWindow

import hollywood_training as ht
from hollywood_training.gui import AnimationClock, MainWindow
from hollywood_training.gui.animation import approach, pulse

from .helpers import (
    build_process,
    image_rgb,
    install_child_probe,
    module_name,
    package_sources,
    parse_source,
)

pytestmark = [pytest.mark.spec("R4"), pytest.mark.gui]

WEB_MODULES = (
    "webbrowser",
    "http.server",
    "socketserver",
    "wsgiref",
    "flask",
    "fastapi",
    "starlette",
    "django",
    "dash",
    "bokeh",
    "streamlit",
    "gradio",
    "tornado",
    "aiohttp",
    "panel",
    "nicegui",
    "PySide6.QtWebEngineCore",
    "PySide6.QtWebEngineWidgets",
    "PySide6.QtWebEngineQuick",
    "PySide6.QtWebView",
    "PySide6.QtWebChannel",
    "PySide6.QtWebSockets",
    "PySide6.QtHttpServer",
)


class PaintCounter(QObject):
    """Counts the paint events a widget receives (an ordinary Qt event filter)."""

    def __init__(self) -> None:
        super().__init__()
        self.count = 0

    def eventFilter(self, watched, event) -> bool:
        if event.type() == QEvent.Type.Paint:
            self.count += 1
        return False


def imported_modules(path) -> set[str]:
    names: set[str] = set()
    for node in ast.walk(parse_source(path)):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            names.add(node.module)
    return names


# ----------------------------------------------------------------- native window
def test_the_window_is_a_native_qt_main_window(gui):
    assert isinstance(gui.window, QMainWindow)
    assert gui.window.isVisible() and gui.window.isWindow()
    assert not [name for name in sys.modules if "WebEngine" in name or "QtWebView" in name]


def test_the_package_uses_no_web_technology():
    offenders = {
        module_name(path): sorted(
            name
            for name in imported_modules(path)
            if any(name == web or name.startswith(web + ".") for web in WEB_MODULES)
        )
        for path in package_sources()
    }
    assert {path: names for path, names in offenders.items() if names} == {}


@pytest.mark.e2e
def test_the_default_run_shows_a_native_window_in_its_own_process(monkeypatch, tmp_path):
    probe = install_child_probe(monkeypatch, tmp_path)
    result = ht.run(build_process(epochs=2).process, close_on_finish=True)
    assert result.state is ht.TrainingState.FINISHED
    probe.assert_installed()
    assert [row["epoch"] for row in probe.applied("EpochCompleted")] == [1, 2]
    (closing,) = probe.records("closing")
    assert closing["web_modules"] == []  # a Qt widget window, not a web view


# -------------------------------------------------------------------- animation
@pytest.mark.timing
def test_the_animation_clock_ticks_smoothly(qtbot):
    clock = AnimationClock(60)
    ticks: list[float] = []
    clock.tick.connect(ticks.append)
    clock.start()
    try:
        qtbot.wait(1000)
    finally:
        clock.stop()
    assert clock.running is False
    assert len(ticks) >= 24, f"only {len(ticks)} animation frames in one second"
    assert all(0.0 <= dt <= 0.1 for dt in ticks)
    count = len(ticks)
    qtbot.wait(100)
    assert len(ticks) == count  # a stopped clock stays stopped


@pytest.mark.timing
def test_the_window_repaints_continuously_between_epochs(gui, qtbot):
    gui.complete_epoch(1)
    counter = PaintCounter()
    gui.window.network.installEventFilter(counter)
    try:
        qtbot.wait(1000)  # no new training data arrives meanwhile
    finally:
        gui.window.network.removeEventFilter(counter)
    assert counter.count >= 20, f"network view repainted only {counter.count} times in 1 s"


def test_frames_change_while_waiting_for_the_next_epoch(gui, qtbot):
    gui.complete_epoch(1)
    first = image_rgb(gui.window.network.grab())
    qtbot.wait(400)
    second = image_rgb(gui.window.network.grab())
    assert first.shape == second.shape
    assert np.abs(first.astype(int) - second.astype(int)).sum() > 0  # it is animated


def test_the_animation_draws_the_real_session_data(gui):
    event = gui.complete_epoch(1)
    assert gui.window.network.last_painted_epoch == event.weights.epoch == 1
    assert gui.session.weights is event.weights  # what is drawn is what training sent


def test_animation_stops_with_the_window(gui):
    assert gui.window.clock.running
    gui.window.close()
    assert not gui.window.clock.running


def test_smoothing_helpers():
    assert approach(0.0, 1.0, dt=0.2, half_life=0.2) == pytest.approx(0.5)
    assert approach(0.3, 1.0, dt=0.5, half_life=0.0) == 1.0
    values = [pulse(t / 10, 1.0) for t in range(30)]
    assert min(values) >= 0.0 and max(values) <= 1.0 and max(values) - min(values) > 0.9


def test_main_window_is_exported_as_the_gui_entry_point():
    assert issubclass(MainWindow, QMainWindow)

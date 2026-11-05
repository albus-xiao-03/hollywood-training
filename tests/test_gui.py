"""GUI: widgets, the main window, and both frontends (headless via Qt "offscreen")."""

from __future__ import annotations

import sys

import pytest
import torch
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QImage
from PySide6.QtWidgets import QApplication
from torch import nn

import hollywood_training as ht
from hollywood_training.commands import create_registry
from hollywood_training.gui import MainWindow
from hollywood_training.gui.app import QtFrontend, ensure_display
from hollywood_training.gui.remote import RemoteFrontend
from hollywood_training.gui.widgets import build_columns, nice_ticks
from hollywood_training.sync import (
    EpochCompleted,
    EpochRecord,
    SessionStarted,
    SessionState,
    SyncChannel,
    TrainingControl,
    TrainingState,
)
from hollywood_training.tracking import NetworkTracker
from hollywood_training.training import TrainingManager
from hollywood_training.training.describe import build_session_info


@pytest.fixture
def window(qtbot, make_process):
    process = make_process(epochs=5)
    tracker = NetworkTracker(process.model)
    session = SessionState()
    info = build_session_info(
        process, tracker.architecture.describe(), total_epochs=5, neuron_limit=20
    )
    started = SessionStarted(info, tracker.snapshot(0), TrainingState.RUNNING)
    win = MainWindow(session, completer=create_registry().complete)
    qtbot.addWidget(win)
    win.show()
    qtbot.waitExposed(win)
    session.apply(started)
    win.apply(started)
    win.tracker = tracker
    return win


# ----------------------------------------------------------------------- window
def test_window_draws_the_session_and_every_epoch(window):
    assert window.network.last_painted_epoch == 0
    assert "SmallNet" in window.windowTitle()
    assert "network SmallNet" in window.terminal.text()
    assert "PgUp / PgDn scroll this log" in window.terminal.text()
    record = EpochRecord(1, 0.9, 0.8, {"accuracy": 0.5})
    event = EpochCompleted(record, window.tracker.snapshot(1), 5, 0.1, 16, 0.2)
    window.session.apply(event)
    window.apply(event)  # synchronous repaint inside
    assert window.network.last_painted_epoch == 1
    assert window.chart.last_painted_epoch == 1
    assert "epoch 1/5" in window.terminal.text()


def test_terminal_submits_commands_and_shows_results(window, qtbot):
    lines: list[str] = []
    window.command_submitted.connect(lines.append)
    qtbot.keyClicks(window.terminal.input, "pause")
    qtbot.keyClick(window.terminal.input, Qt.Key.Key_Return)
    assert lines == ["pause"] and "ht ❯ pause" in window.terminal.text()
    window.show_command_result(False, "refused")
    assert "refused" in window.terminal.text()


def test_tab_completion_and_history(window, qtbot):
    line = window.terminal.input
    qtbot.keyClicks(line, "hist")
    qtbot.keyClick(line, Qt.Key.Key_Tab)
    assert line.text() == "history "
    qtbot.keyClick(line, Qt.Key.Key_Return)
    qtbot.keyClick(line, Qt.Key.Key_Up)
    assert line.text() == "history"


def test_scroll_keys_in_the_command_line_scroll_the_log(window, qtbot):
    terminal, line = window.terminal, window.terminal.input
    for k in range(300):
        terminal.write(f"line {k}")
    bar = terminal.output.verticalScrollBar()
    bottom, page = bar.maximum(), bar.pageStep()
    assert bottom > 2 * page > 0 and bar.value() == bottom  # the log follows its end
    qtbot.keyClick(line, Qt.Key.Key_PageUp)
    assert bar.value() == bottom - page
    qtbot.keyClick(line, Qt.Key.Key_PageUp, Qt.KeyboardModifier.ShiftModifier)
    assert bar.value() == bottom - 2 * page
    qtbot.keyClick(line, Qt.Key.Key_PageDown)
    assert bar.value() == bottom - page
    qtbot.keyClick(line, Qt.Key.Key_Home, Qt.KeyboardModifier.ControlModifier)
    assert bar.value() == 0
    terminal.write("a new line")  # scrolled back: the view stays where it is
    assert bar.value() == 0
    qtbot.keyClick(line, Qt.Key.Key_End, Qt.KeyboardModifier.ControlModifier)
    assert bar.value() == bar.maximum()
    assert line.text() == ""  # the scroll keys never edit the line
    qtbot.keyClicks(line, "status")
    qtbot.keyClick(line, Qt.Key.Key_Home)  # without Ctrl: the text cursor moves
    assert line.cursorPosition() == 0 and bar.value() == bar.maximum()


def test_screenshot_writes_a_png(window, tmp_path):
    image = QImage(str(window.save_screenshot(tmp_path / "shots" / "window.png")))
    assert not image.isNull() and image.width() >= 960


def test_panels_paint_before_any_session(qtbot):
    win = MainWindow(SessionState())
    qtbot.addWidget(win)
    win.show()
    qtbot.waitExposed(win)
    win.render_now()  # "awaiting" states must not raise
    assert win.network.last_painted_epoch is None


# ----------------------------------------------------------------------- drawing
def test_columns_follow_the_layers_and_bridge_reshapes():
    model = nn.Sequential(nn.Conv2d(1, 4, 3), nn.ReLU(), nn.Flatten(), nn.Linear(4 * 4 * 4, 3))
    tracker = NetworkTracker(model, max_neurons=8)
    tracker.begin_trace()
    model(torch.randn(2, 1, 6, 6))
    tracker.end_trace()
    columns, links = build_columns(tracker.architecture.describe(), tracker.snapshot(0), 8)
    assert [link.bridge for link in links] == [False, True, False]
    assert [column.size for column in columns] == [1, 4, 64, 3]
    assert columns[2].caption == "8 of 64 features"
    assert columns[0].caption == "1 feature"


def test_nice_ticks():
    assert nice_ticks(0, 1, 5) == [0, 0.2, 0.4, 0.6, 0.8, 1.0]
    assert nice_ticks(0, 2.6, 5) == [0, 0.5, 1.0, 1.5, 2.0, 2.5]


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="X11/Wayland check")
def test_missing_display_is_reported_clearly(monkeypatch):
    for variable in ("QT_QPA_PLATFORM", "DISPLAY", "WAYLAND_DISPLAY"):
        monkeypatch.delenv(variable, raising=False)
    with pytest.raises(RuntimeError, match="no display"):
        ensure_display()


# ------------------------------------------------------- in-process frontend
def test_in_process_window_draws_each_epoch_before_training_continues(
    make_process, monkeypatch, qapp
):
    drawn = []
    original = QtFrontend._handle

    def spy(self, envelope):
        original(self, envelope)  # applies, draws, acknowledges
        if isinstance(envelope.event, EpochCompleted):
            window = self.window
            drawn.append(
                (
                    envelope.event.record.epoch,
                    window.network.last_painted_epoch,
                    window.chart.last_painted_epoch,
                )
            )

    monkeypatch.setattr(QtFrontend, "_handle", spy)
    result = ht.run(make_process(epochs=4), isolate_gui=False, close_on_finish=True)
    assert result.state is ht.TrainingState.FINISHED and result.epochs_completed == 4
    assert drawn == [(epoch, epoch, epoch) for epoch in range(1, 5)]
    assert torch.get_num_threads() >= 1  # thread cap restored without error


def test_in_process_commands_drive_the_run(make_process, qapp):
    process = make_process(epochs=10)
    transcript: list[str] = []

    @process.command("hello", summary="greet")
    def hello(ctx):
        return f"hello from epoch {ctx.session.epoch}"

    def window() -> MainWindow:
        return next(w for w in QApplication.topLevelWidgets() if isinstance(w, MainWindow))

    def drive() -> None:
        window().run_command("hello")
        window().run_command("step 2")
        QTimer.singleShot(1500, finish)

    def finish() -> None:
        transcript.append(window().terminal.text())
        window().run_command("quit")

    QTimer.singleShot(300, drive)
    result = ht.run(process, isolate_gui=False, start_paused=True)
    assert result.state is ht.TrainingState.STOPPED and result.epochs_completed == 2
    assert "hello from epoch 0" in transcript[0]
    assert "running 2 epochs, then pausing" in transcript[0]


# ------------------------------------------------------- isolated frontend
def test_window_process_end_to_end(make_process):
    result = ht.run(make_process(epochs=3), close_on_finish=True)
    assert result.state is ht.TrainingState.FINISHED and result.epochs_completed == 3


def test_window_process_commands_and_screenshot(make_process, tmp_path):
    process = make_process(epochs=3)
    control = TrainingControl(3)
    channel = SyncChannel()
    remote = RemoteFrontend(
        channel=channel, control=control, registry=create_registry(), close_on_finish=True
    )
    shot = tmp_path / "remote.png"

    @process.on_epoch_end
    def drive(record):
        if record.epoch == 2:
            remote._execute("status")  # as if typed in the window
            remote._execute("scale log")  # window-only commands travel to the window process
            remote._execute("probe 1")
            remote._execute("neuron fc1 2")
            remote._execute("layout chart 50")
            remote._execute("copy 3")
            remote.screenshot(shot)

    manager = TrainingManager(process, channel=channel, control=control, linger=False)
    result = remote.run(manager)
    assert result.epochs_completed == 3
    assert shot.exists() and not QImage(str(shot)).isNull()
    assert remote.session.epoch == 3  # the training-side mirror followed along


def test_window_process_failure_is_reported(make_process, monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "no-such-platform")
    with pytest.raises(RuntimeError, match="GUI process"):
        ht.run(make_process(epochs=1), close_on_finish=True)

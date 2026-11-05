"""Window-only commands (SPEC R2.3): keyboard replacements for the R1 mouse interactions."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication

import hollywood_training as ht
from hollywood_training.commands import CommandContext, ViewRequest, create_registry
from hollywood_training.gui import MainWindow
from hollywood_training.gui.app import QtFrontend
from hollywood_training.sync import (
    EpochCompleted,
    EpochRecord,
    LogMessage,
    SessionFinished,
    SessionStarted,
    SessionState,
    StateChanged,
    SyncChannel,
    TrainingControl,
    TrainingPaused,
    TrainingState,
    TrainingTarget,
)
from hollywood_training.tracking import NetworkTracker
from hollywood_training.training.describe import build_session_info

LOSSES = [(0.9, 1.0), (0.5, 0.7), (0.2, 0.4)]  # (train, val) of epochs 1-3
RUNNING_HINT = "training is running: the probe line clears at the next epoch ('pause' keeps it)"


def deliver(window: MainWindow, event) -> None:
    window.session.apply(event)  # as the frontends do: the state first, then the window
    window.apply(event)


@pytest.fixture
def gui(qtbot, make_process):
    """A window after three epochs of a six-epoch run, driven by the in-process frontend."""
    process = make_process(epochs=6)
    tracker = NetworkTracker(process.model)
    session = SessionState()
    info = build_session_info(
        process, tracker.architecture.describe(), total_epochs=6, neuron_limit=20
    )
    registry = create_registry(process.commands)
    window = MainWindow(session, completer=registry.complete)
    qtbot.addWidget(window)
    with qtbot.waitExposed(window):
        window.show()
    deliver(window, SessionStarted(info, tracker.snapshot(0), TrainingState.RUNNING))
    for epoch, (train, val) in enumerate(LOSSES, start=1):
        record = EpochRecord(epoch, train, val, {"accuracy": 0.5 + epoch / 10}, {}, 0.1, 0.25)
        deliver(window, EpochCompleted(record, tracker.snapshot(epoch), 6, 0.1, 16, epoch))
    control = TrainingControl(6)
    frontend = QtFrontend(channel=SyncChannel(), control=control, registry=registry)
    frontend.window = window
    ctx = CommandContext(
        control=control,
        session=session,
        registry=registry,
        target=TrainingTarget(process.model, process.optimizer),
        frontend=frontend,
    )
    return SimpleNamespace(window=window, ctx=ctx, tracker=tracker, process=process)


def run(gui, line: str):
    return gui.ctx.registry.execute(line, gui.ctx)


def test_view_commands_are_listed_under_view(gui):
    listing = run(gui, "help").output
    assert "VIEW" in listing
    assert listing.index("INSPECTION") < listing.index("VIEW") < listing.index("SESSION")


# ------------------------------------------------------------------------ scale
def test_scale_switches_the_loss_axis(gui):
    chart = gui.window.chart
    assert not chart.log_scale
    assert run(gui, "scale").output == "loss axis: logarithmic"  # no argument: switch
    assert chart.log_scale
    assert run(gui, "scale").output == "loss axis: linear"
    assert not chart.log_scale
    assert run(gui, "scale log").output == "loss axis: logarithmic" and chart.log_scale
    assert run(gui, "scale log").ok and chart.log_scale  # setting it twice keeps it
    assert run(gui, "scale lin").output == "loss axis: linear" and not chart.log_scale
    result = run(gui, "scale up")
    assert not result.ok and result.output == "mode must be one of: lin, log"


def test_scale_says_when_a_log_axis_cannot_be_drawn_yet(qtbot):
    window = MainWindow(SessionState())  # no epoch yet: no positive loss to draw
    qtbot.addWidget(window)
    ok, text = window.handle_view(ViewRequest("scale", ("log",)))
    assert ok and text == "loss axis: logarithmic (drawn linear until a loss is positive)"


# ------------------------------------------------------------------------ probe
def probe_column(chart, epoch: int) -> list[int]:
    """The chart's pixels (RGB) down the vertical line at ``epoch``, inside the plot."""
    image = chart.grab().toImage()
    axes = chart._axes
    x = round(axes.x(epoch))
    top, bottom = round(axes.plot.top()) + 2, round(axes.plot.bottom()) - 2
    return [image.pixelColor(x, y).rgb() for y in range(top, bottom)]


def test_probe_reports_an_epoch_and_draws_the_probe_line(gui):
    chart = gui.window.chart
    gui.window.clock.stop()  # still frames: compare two grabs pixel by pixel
    before = probe_column(chart, 2)
    report = run(gui, "probe 2")
    assert report.ok and report.output.splitlines() == [
        "epoch 2 of 6",
        "  train loss  0.5000",
        "  val loss    0.7000",
        "  lr          0.1",
        "  time        0.250 s",
        "  accuracy    70.0%",
        RUNNING_HINT,  # the fixture's run is running
    ]
    assert chart.probe_epoch == 2
    after = probe_column(chart, 2)
    changed = sum(a != b for a, b in zip(before, after, strict=True))
    assert changed > len(after) // 3  # the dashed vertical line (and its rings) at epoch 2
    assert chart.last_probe_readout == ["EPOCH 2", "TRAIN 0.5000", "VAL   0.7000"]


def test_probe_defaults_to_the_latest_epoch_and_marks_the_best(gui):
    report = run(gui, "probe").output
    assert report.startswith("epoch 3 of 6 (best)")
    gui.window.chart.repaint()
    assert gui.window.chart.last_probe_readout[0] == "EPOCH 3"


def test_probe_line_follows_the_axis_scale(gui):
    run(gui, "probe 1")
    run(gui, "scale log")
    gui.window.chart.repaint()
    assert gui.window.chart.last_probe_readout == ["EPOCH 1", "TRAIN 0.9000", "VAL   1.0000"]


def test_probe_off_removes_the_line(gui):
    chart = gui.window.chart
    run(gui, "probe 1")
    assert run(gui, "probe off").output == "probe line removed"
    assert chart.probe_epoch is None
    chart.repaint()
    assert chart.last_probe_readout is None
    assert run(gui, "probe off").output == "no probe line to remove"


def test_probe_refuses_epochs_that_do_not_exist(gui, qtbot):
    result = run(gui, "probe 9")
    assert not result.ok and result.output == "epoch 9 has not completed yet (last completed: 3)"
    for word in ("0", "first", "-1"):
        result = run(gui, f"probe {word}")
        assert not result.ok and "must be an epoch number (1 or more) or off" in result.output
    assert gui.window.chart.probe_epoch is None
    gui.ctx.session.history.clear()
    assert run(gui, "probe").output == "no completed epochs yet"


# ---------------------------------------------- probe line: removed when training continues
def state_changed(state: TrainingState, message: str = "", lr: float = 0.1) -> StateChanged:
    return StateChanged(state, 3, 6, lr, 0.0, 3.0, message)


def test_probe_line_is_removed_at_the_next_epoch(gui):
    chart = gui.window.chart
    assert run(gui, "probe 2").output.splitlines()[-1] == RUNNING_HINT
    record = EpochRecord(4, 0.1, 0.3)
    deliver(gui.window, EpochCompleted(record, gui.tracker.snapshot(4), 6, 0.1, 16, 4.0))
    assert chart.probe_epoch is None
    assert chart.last_probe_readout is None  # the epoch's synchronous repaint had no line
    assert chart.last_painted_epoch == 4


def test_probe_line_stays_while_paused_until_training_resumes(gui):
    window = gui.window
    deliver(window, TrainingPaused(4, None, 0, gui.tracker.snapshot(3), 3.0))
    deliver(window, state_changed(TrainingState.PAUSED, "paused (before epoch 4)"))
    assert RUNNING_HINT not in run(gui, "probe 2").output
    deliver(window, state_changed(TrainingState.PAUSED, lr=0.05))  # e.g. `lr 0.05` while paused
    deliver(window, LogMessage("learning rate set to 0.05"))
    assert window.chart.probe_epoch == 2
    deliver(window, state_changed(TrainingState.RUNNING, "resumed"))  # resume / continue / step
    assert window.chart.probe_epoch is None


def test_probe_line_is_removed_when_a_finished_run_continues(gui):
    window = gui.window
    deliver(window, state_changed(TrainingState.FINISHED, "all 6 epochs complete"))
    run(gui, "probe 1")
    deliver(window, state_changed(TrainingState.FINISHED, lr=0.02))  # a setting change
    assert window.chart.probe_epoch == 1
    deliver(window, state_changed(TrainingState.RUNNING, "continuing to epoch 9"))  # `epochs 9`
    assert window.chart.probe_epoch is None


def test_probe_line_stays_after_training_ended(gui):
    chart = gui.window.chart
    run(gui, "probe 3")
    deliver(gui.window, SessionFinished(TrainingState.STOPPED, 3, 5.0, None, "stopped"))
    assert chart.probe_epoch == 3
    assert chart.last_probe_readout[0] == "EPOCH 3"  # drawn by the final synchronous repaint
    assert run(gui, "probe off").output == "probe line removed"


def test_probe_line_in_a_real_run_is_removed_on_resume(make_process, qapp):
    process = make_process(epochs=8)
    seen: dict[str, object] = {}

    def window() -> MainWindow:
        return next(w for w in QApplication.topLevelWidgets() if isinstance(w, MainWindow))

    def step() -> None:
        window().run_command("step 2")  # two epochs, then paused
        QTimer.singleShot(1200, probe)

    def probe() -> None:
        window().run_command("probe 1")
        seen["paused"] = (window().session.state, window().chart.probe_epoch)
        window().run_command("resume")
        QTimer.singleShot(1200, finish)

    def finish() -> None:
        seen["resumed"] = window().chart.probe_epoch
        seen["report"] = window().terminal.text()
        window().run_command("quit")

    QTimer.singleShot(300, step)
    ht.run(process, isolate_gui=False, start_paused=True)
    assert seen["paused"] == (TrainingState.PAUSED, 1)
    assert seen["resumed"] is None
    assert "epoch 1 of 8" in seen["report"] and RUNNING_HINT not in seen["report"]

"""Window-only commands (SPEC R2.3): keyboard replacements for the R1 mouse interactions."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from hollywood_training.commands import CommandContext, ViewRequest, create_registry
from hollywood_training.gui import MainWindow
from hollywood_training.gui.app import QtFrontend
from hollywood_training.sync import (
    EpochCompleted,
    EpochRecord,
    SessionStarted,
    SessionState,
    SyncChannel,
    TrainingControl,
    TrainingState,
    TrainingTarget,
)
from hollywood_training.tracking import NetworkTracker
from hollywood_training.training.describe import build_session_info

LOSSES = [(0.9, 1.0), (0.5, 0.7), (0.2, 0.4)]  # (train, val) of epochs 1-3


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

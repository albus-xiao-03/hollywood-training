"""Window-only commands (SPEC R2.3): keyboard replacements for the R1 mouse interactions."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QGuiApplication
from PySide6.QtWidgets import QApplication
from torch import nn

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


# ----------------------------------------------------------------------- neuron
def column(gui, title: str):
    return next(c for c in gui.window.network.columns() if c.title == title)


def test_neuron_lists_the_drawn_columns_and_their_neurons(gui):
    listing = run(gui, "neuron").output.splitlines()
    assert listing[0] == "drawn columns ('neuron <column> <index>' highlights a neuron):"
    assert [line.split()[0] for line in listing[1:]] == ["INPUT", "FC1", "FC2"]
    assert "6 neurons" in listing[2] and listing[2].endswith("neurons 0-5")
    assert run(gui, "neuron fc1").output == "FC1 · drawn neurons 0-5"


def test_neuron_highlights_a_neuron_and_reports_it(gui):
    network = gui.window.network
    gui.window.clock.stop()  # still frames: compare two grabs
    fc1 = column(gui, "FC1")
    x, y, reach = round(fc1.x), round(float(fc1.ys[3])), round(network._radius) + 12

    def around() -> list[int]:
        image = network.grab().toImage()
        return [
            image.pixelColor(i, j).rgb()
            for i in range(x - reach, x + reach)
            for j in range(y - reach, y + reach)
        ]

    before = around()
    report = run(gui, "neuron fc1 3")
    expected = ["FC1 · neuron 3", f"bias {float(fc1.bias[3]):+.5f}"]
    assert report.ok and report.output.splitlines() == expected
    assert network.selected_neuron == ("FC1", 3)
    assert sum(a != b for a, b in zip(before, around(), strict=True)) > 50  # the rings
    assert network.last_selection_readout == expected


def test_neuron_columns_are_matched_loosely(gui):
    assert run(gui, "neuron FC2 1").output.startswith("FC2 · neuron 1")
    assert run(gui, "neuron input 0").output == "INPUT · neuron 0"  # inputs have no bias
    assert gui.window.network.selected_neuron == ("INPUT", 0)


def test_neuron_refusals(gui):
    for line, message in [
        ("neuron fc1 9", "FC1 has no drawn neuron 9 - drawn: 0-5"),
        ("neuron xyz 0", "no column 'xyz' - columns: INPUT, FC1, FC2"),
        ("neuron fc1 -1", "invalid index '-1': must be 0 or more"),
    ]:
        result = run(gui, line)
        assert not result.ok and result.output == message, line
    assert gui.window.network.selected_neuron is None


def test_neuron_off_removes_the_highlight(gui):
    network = gui.window.network
    run(gui, "neuron fc2 0")
    assert run(gui, "neuron off").output == "neuron highlight removed"
    network.repaint()
    assert network.selected_neuron is None and network.last_selection_readout is None
    assert run(gui, "neuron off").output == "no neuron highlighted"


def test_neuron_highlight_stays_when_the_weights_change(gui):
    run(gui, "neuron fc1 2")
    record = EpochRecord(4, 0.1, 0.3)
    deliver(gui.window, EpochCompleted(record, gui.tracker.snapshot(4), 6, 0.1, 16, 4.0))
    assert gui.window.network.selected_neuron == ("FC1", 2)
    assert gui.window.network.last_selection_readout[0] == "FC1 · neuron 2"


def test_neuron_on_sampled_columns_with_numeric_titles(qtbot, make_process):
    model = nn.Sequential(nn.Linear(2, 40), nn.Tanh(), nn.Linear(40, 2))
    process = make_process(model=model, optimizer=torch.optim.SGD(model.parameters(), lr=0.1))
    tracker = NetworkTracker(model, max_neurons=8)
    session = SessionState()
    architecture = tracker.architecture.describe()
    info = build_session_info(process, architecture, total_epochs=1, neuron_limit=8)
    window = MainWindow(session)
    qtbot.addWidget(window)
    window.show()
    deliver(window, SessionStarted(info, tracker.snapshot(0), TrainingState.RUNNING))
    columns = window.network.columns()
    assert [c.title for c in columns] == ["INPUT", "0", "2"]  # Sequential layer names
    drawn = [int(i) for i in columns[1].indices]
    assert len(drawn) == 8 and drawn[0] == 0 and drawn[-1] == 39
    ok, listing = window.handle_view(ViewRequest("neuron", (None, None)))
    assert ok and "8 of 40 neurons" in listing and "0 … 39 (8 drawn)" in listing
    ok, text = window.handle_view(ViewRequest("neuron", ("0", None)))
    assert ok and text == "0 · drawn neurons " + ", ".join(str(i) for i in drawn)
    ok, text = window.handle_view(ViewRequest("neuron", ("0", drawn[1])))
    assert ok and text.startswith(f"0 · neuron {drawn[1]}")
    hidden = next(i for i in range(40) if i not in drawn)
    ok, text = window.handle_view(ViewRequest("neuron", ("0", hidden)))
    assert not ok and text.startswith(f"0 has no drawn neuron {hidden} - drawn: 0, ")


def test_neuron_needs_a_network(qtbot):
    window = MainWindow(SessionState())
    qtbot.addWidget(window)
    assert window.handle_view(ViewRequest("neuron", (None, None))) == (
        False,
        "no network drawn yet",
    )
    assert window.handle_view(ViewRequest("neuron", ("off", None))) == (
        True,
        "no neuron highlighted",
    )


# ----------------------------------------------------------------------- layout
@pytest.fixture
def roomy(gui, qtbot):
    """The fixture's window, large enough that no minimum size bends the shares."""
    gui.window.resize(1600, 1000)
    qtbot.waitUntil(lambda: sum(gui.window.splitters["right"].sizes()) > 700)
    gui.window.reset_layout()
    return gui


def test_layout_shows_the_panel_shares(roomy):
    lines = run(roomy, "layout").output.splitlines()
    assert lines[0].startswith("columns ") and lines[0].endswith("left 62% · right 38%")
    assert lines[1].startswith("left column ") and lines[1].endswith("network 70% · terminal 30%")
    assert lines[2].startswith("right column ")
    assert lines[2].endswith("status 44% · chart 36% · usage 20%")
    assert lines[3] == "'layout <panel> <percent>' resizes, 'layout reset' restores"
    assert run(roomy, "layout usage").output == "usage: 20% of the right column"


def test_layout_sets_a_share_and_keeps_the_others_in_proportion(roomy):
    window = roomy.window
    before = window.splitters["right"].sizes()
    answer = run(roomy, "layout chart 50").output
    shares = window.panel_shares()["right"]
    assert answer == f"right column  status {shares['status']}% · chart 50% · usage 20%".replace(
        "usage 20%", f"usage {shares['usage']}%"
    )
    after = window.splitters["right"].sizes()
    assert abs(after[1] / sum(after) - 0.5) < 0.01 and sum(after) == sum(before)
    ratio_before, ratio_after = before[0] / before[2], after[0] / after[2]
    assert abs(ratio_after - ratio_before) / ratio_before < 0.05  # status : usage kept
    assert run(roomy, "layout left 75%").ok  # a trailing % is fine
    assert abs(window.panel_shares()["columns"]["left"] - 75) <= 1


def test_layout_keeps_minimum_sizes(roomy):
    answer = run(roomy, "layout chart 95").output
    assert answer.endswith("(limited by the panels' minimum sizes)")
    assert roomy.window.panel_shares()["right"]["chart"] < 95
    assert all(
        widget.height() >= widget.minimumHeight()
        for widget in (roomy.window.status, roomy.window.chart, roomy.window.usage)
    )


def test_layout_reset_restores_the_initial_shares(roomy):
    run(roomy, "layout network 80")
    assert run(roomy, "layout network").output == "network: 80% of the left column"
    lines = run(roomy, "layout reset").output.splitlines()
    assert lines[0] == "layout reset" and lines[2].endswith("network 70% · terminal 30%")
    assert roomy.window.panel_shares()["left"] == {"network": 70, "terminal": 30}


def test_layout_refusals(roomy):
    panels = "left, right, network, terminal, status, chart, usage"
    for line, message in [
        ("layout desk 50", f"no panel 'desk' - panels: {panels}"),
        ("layout chart 0", "invalid percent '0': must be a whole number from 1 to 99"),
        ("layout chart 100", "invalid percent '100': must be a whole number from 1 to 99"),
        ("layout reset 50", "'layout reset' takes no percentage"),
    ]:
        result = run(roomy, line)
        assert not result.ok and result.output == message, line
    assert run(roomy, "layout CHART 40").ok  # panel names ignore case


# ------------------------------------------------------------------------- copy
def test_copy_puts_the_log_on_the_clipboard(gui):
    terminal = gui.window.terminal
    terminal.clear()
    for line in ("alpha", "beta", "gamma"):
        terminal.write(line)
    terminal.echo("copy 2")  # typed in the terminal: echoed before the command runs
    answer = run(gui, "copy 2").output
    assert answer == "copied 2 lines to the clipboard"
    assert QGuiApplication.clipboard().text() == "beta\ngamma"  # without the echo
    terminal.write(answer)  # the window shows the answer
    terminal.echo("copy")
    assert run(gui, "copy").output == "copied 5 lines to the clipboard"
    copied = QGuiApplication.clipboard().text().splitlines()
    assert copied == ["alpha", "beta", "gamma", "ht ❯ copy 2", answer]  # earlier echoes stay
    terminal.write("one more line")
    assert run(gui, "copy 1").output == "copied 1 line to the clipboard"
    assert QGuiApplication.clipboard().text() == "one more line"


def test_copy_refusals(gui):
    gui.window.terminal.clear()
    assert run(gui, "copy").output == "nothing to copy: the log is empty"
    result = run(gui, "copy 0")
    assert not result.ok and result.output == "invalid lines '0': must be a positive integer"


# ----------------------------------------------------------------------- window
def test_window_shows_and_changes_the_window_mode(gui, qtbot):
    window = gui.window
    assert run(gui, "window").output == f"window: normal ({window.width()} × {window.height()})"
    assert run(gui, "window max").output == "window: maximized"
    assert window.windowState() & Qt.WindowState.WindowMaximized
    assert run(gui, "window full").output == "window: full screen"
    assert window.isFullScreen()
    assert run(gui, "window normal").output.startswith("window: normal (")
    assert not window.isFullScreen() and not window.isMaximized()
    assert run(gui, "window min").output == "window: minimized"
    assert window.isMinimized()
    result = run(gui, "window huge")
    assert not result.ok and result.output == "mode must be one of: normal, max, full, min"

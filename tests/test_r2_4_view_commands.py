"""R2 requirement 4 - "Additional commands may be added to replace existing mouse-based
interactions."

The package adds six window-only ("view") commands for the first version's mouse
interactions: ``scale`` (click the chart), ``probe`` (hover the chart), ``neuron`` (hover a
neuron), ``layout`` (drag the splitters), ``copy`` (select log text) and ``window`` (the title
bar's buttons). As commands they must be *well-defined* (R1 requirement 7 holds for them
too): registered and documented, arguments validated, every boundary handled with a clear
answer - and they must really do what replaces the mouse interaction (R2 requirement 3).

Covered here: registration and help, the requests they hand to the window (and that bad
arguments never get that far), every documented form and every boundary of each command in
a real window (offscreen), when the probe line clears by itself, and the same requests
crossing the process link to the real isolated window process.
"""

from __future__ import annotations

import math
import re
import time

import pytest
from htspec.events import SessionFeeder
from htspec.headless import FakeFrontend
from htspec.models import ConvNet, TinyNet, WideNet, make_setup
from htspec.probe_log import LINGER_CLOSE
from PySide6.QtCore import Qt
from PySide6.QtGui import QGuiApplication

import hollywood_training as ht
from hollywood_training import formatting as fmt
from hollywood_training.commands.builtin import create_registry
from hollywood_training.commands.context import CommandContext
from hollywood_training.commands.view import ViewRequest
from hollywood_training.gui.window import SPLITS
from hollywood_training.sync.control import TrainingControl, TrainingTarget
from hollywood_training.sync.events import TrainingState

pytestmark = pytest.mark.req("R2.4", "R1.7")

VIEW_COMMANDS = ("scale", "probe", "neuron", "layout", "copy", "window")


def run(rig, line: str):
    result = rig.type_line(line)
    assert result is not None, f"{line!r} ran no command"
    return result


def shares(rig, splitter: str) -> dict[str, int]:
    return rig.window.panel_shares()[splitter]


# ---------------------------------------------------------------- registration
def test_the_view_commands_are_registered_and_documented(rig):
    for name in VIEW_COMMANDS:
        command = rig.registry.get(name)
        assert command is not None and command.category == "view", name
        details = run(rig, f"help {name}")
        assert details.ok and details.output.startswith(f"usage: {name}"), name
        assert command.summary in details.output
    listing = run(rig, "help").output
    section = listing.split("VIEW", 1)[1].split("SESSION", 1)[0]
    for name in VIEW_COMMANDS:
        assert f"\n  {name}" in section, name


@pytest.fixture
def headless_context():
    """Commands with a frontend that records the window requests (no window)."""
    setup = make_setup()
    feeder = SessionFeeder(setup=setup)
    feeder.start()
    feeder.epochs(3)
    frontend = FakeFrontend()
    context = CommandContext(
        control=TrainingControl(10),
        session=feeder.session,
        registry=create_registry(),
        target=TrainingTarget(setup.model, setup.optimizer),
        frontend=frontend,
    )
    return context, frontend


@pytest.mark.parametrize(
    ("line", "request"),
    [
        ("scale", ViewRequest("scale", (None,))),
        ("scale log", ViewRequest("scale", ("log",))),
        ("scale lin", ViewRequest("scale", ("lin",))),
        ("probe", ViewRequest("probe", (3,))),
        ("probe 2", ViewRequest("probe", (2,))),
        ("probe off", ViewRequest("probe", (None,))),
        ("neuron", ViewRequest("neuron", (None, None))),
        ("neuron fc1", ViewRequest("neuron", ("fc1", None))),
        ("neuron fc1 3", ViewRequest("neuron", ("fc1", 3))),
        ("neuron off", ViewRequest("neuron", ("off", None))),
        ("layout", ViewRequest("layout", (None, None))),
        ("layout chart 40", ViewRequest("layout", ("chart", 40))),
        ("layout chart 40%", ViewRequest("layout", ("chart", 40))),
        ("layout reset", ViewRequest("layout", ("reset", None))),
        ("copy", ViewRequest("copy", (None,))),
        ("copy 5", ViewRequest("copy", (5,))),
        ("window", ViewRequest("window", (None,))),
        ("window full", ViewRequest("window", ("full",))),
    ],
)
def test_each_form_hands_the_window_one_validated_request(headless_context, line, request):
    context, frontend = headless_context
    result = context.registry.execute(line, context)
    assert result.ok, result.output
    assert frontend.calls == [("view", request)]


@pytest.mark.parametrize(
    "line",
    ["scale foo", "scale log lin", "probe 0", "probe -1", "probe abc", "probe 1.5", "probe 99",
     "neuron fc1 -1", "neuron fc1 x", "neuron fc1 1 2", "layout chart 0", "layout chart 100",
     "layout chart 101", "layout chart -5", "layout chart abc", "layout chart 40 50",
     "copy 0", "copy -1", "copy x", "copy 1 2", "window huge", "window max full"],
)  # fmt: skip
def test_bad_arguments_are_refused_before_reaching_the_window(headless_context, line):
    context, frontend = headless_context
    result = context.registry.execute(line, context)
    assert result.ok is False and result.output
    assert frontend.calls == []


@pytest.mark.parametrize("line", ["scale log", "probe", "neuron", "layout", "copy", "window"])
def test_view_commands_are_refused_where_there_is_no_window(line):
    setup = make_setup()
    feeder = SessionFeeder(setup=setup)
    feeder.start()
    feeder.epochs(1)
    context = CommandContext(
        control=TrainingControl(10),
        session=feeder.session,
        registry=create_registry(),
        target=TrainingTarget(setup.model, setup.optimizer),
        frontend=None,
    )
    result = context.registry.execute(line, context)
    assert result.ok is False and "not available" in result.output


# ----------------------------------------------------------------------- scale
def test_scale_without_an_argument_switches_the_axis(rig):
    rig.feeder.epochs(2)
    assert run(rig, "scale").output == "loss axis: logarithmic"
    assert rig.window.chart.log_scale is True
    assert run(rig, "scale").output == "loss axis: linear"
    assert rig.window.chart.log_scale is False


def test_scale_with_a_mode_sets_it(rig):
    rig.feeder.epochs(2)
    for line, expected in (("scale log", True), ("scale log", True), ("scale lin", False),
                           ("scale lin", False)):  # fmt: skip
        assert run(rig, line).ok
        assert rig.window.chart.log_scale is expected, line


def test_scale_rejects_an_unknown_mode(rig):
    result = run(rig, "scale foo")
    assert result.ok is False and "lin, log" in result.output
    assert rig.window.chart.log_scale is False


@pytest.mark.parametrize(
    "losses",
    [[], [0.0, 0.0], [-1.0, -2.0], [float("nan"), float("inf")], [0.0, -0.5]],
    ids=["no-epochs", "zero", "negative", "non-finite", "zero-and-negative"],
)
def test_scale_log_without_a_positive_loss_says_it_is_drawn_linear(rig, paint_capture, losses):
    for value in losses:
        rig.feeder.train_epoch(train_loss=value, val_loss=None)
    result = run(rig, "scale log")
    assert result.ok and "drawn linear until a loss is positive" in result.output
    assert rig.window.chart.log_scale is True
    if any(math.isfinite(value) for value in losses):  # there are axes to label
        assert "LIN" in paint_capture.render(rig.window.chart)


def test_scale_log_draws_decade_ticks(rig, paint_capture):
    for value in (0.5, 0.05, 0.005):
        rig.feeder.train_epoch(train_loss=value, val_loss=None)
    assert run(rig, "scale log").output == "loss axis: logarithmic"
    drawn = paint_capture.render(rig.window.chart)
    assert "LOG" in drawn
    for label in ("1e-03", "1e-02", "1e-01", "1e+00"):
        assert label in drawn, label


# ----------------------------------------------------------------------- probe
def test_probe_before_any_epoch_is_refused(rig):
    result = run(rig, "probe")
    assert result.ok is False and "no completed epochs yet" in result.output
    assert rig.window.chart.probe_epoch is None


def test_probe_without_an_argument_marks_the_latest_epoch(rig):
    rig.feeder.epochs(3)
    result = run(rig, "probe")
    assert result.ok and result.output.startswith("epoch 3 of 10")
    assert rig.window.chart.probe_epoch == 3


def test_probe_reports_the_epochs_values(rig):
    rig.feeder.train_epoch(metrics={"accuracy": 0.5}, val_metrics={"accuracy": 0.25})
    record = rig.feeder.records[0]
    rig.feeder.epochs(1)
    output = run(rig, "probe 1").output
    assert output.startswith("epoch 1 of 10")
    assert f"train loss  {fmt.loss(record.train_loss)}" in output
    assert f"val loss    {fmt.loss(record.val_loss)}" in output
    assert f"lr          {fmt.learning_rate(record.learning_rate)}" in output
    assert f"time        {record.duration:.3f} s" in output
    assert re.search(r"accuracy\s+50\.0%", output) and re.search(r"val accuracy\s+25\.0%", output)


def test_probe_marks_the_best_epoch(rig):
    for val in (0.9, 0.5, 0.7):
        rig.feeder.train_epoch(val_loss=val)
    assert "(best)" in run(rig, "probe 2").output.splitlines()[0]
    assert "(best)" not in run(rig, "probe 3").output.splitlines()[0]


@pytest.mark.parametrize("argument", ["0", "-1", "abc", "1.5", "-0"])
def test_probe_rejects_invalid_epochs(rig, argument):
    rig.feeder.epochs(3)
    result = run(rig, f"probe {argument}")
    assert result.ok is False and "epoch number" in result.output
    assert rig.window.chart.probe_epoch is None


def test_probe_rejects_an_epoch_that_has_not_completed(rig):
    rig.feeder.epochs(3)
    result = run(rig, "probe 4")
    assert result.ok is False
    assert "epoch 4 has not completed yet (last completed: 3)" in result.output


def test_probe_off(rig):
    rig.feeder.epochs(2)
    assert run(rig, "probe off").output == "no probe line to remove"
    run(rig, "probe 1")
    assert run(rig, "probe off").output == "probe line removed"
    assert rig.window.chart.probe_epoch is None


def test_the_probe_line_is_drawn_with_the_epochs_values(rig):
    rig.feeder.train_epoch()
    rig.feeder.train_epoch(val_loss=None)
    rig.feeder.state(TrainingState.PAUSED, "paused")
    chart = rig.window.chart
    first, second = rig.feeder.records
    run(rig, "probe 1")
    chart.repaint()
    assert chart.last_probe_readout == [
        "EPOCH 1",
        f"TRAIN {fmt.loss(first.train_loss)}",
        f"VAL   {fmt.loss(first.val_loss)}",
    ]
    run(rig, "probe 2")
    chart.repaint()
    assert chart.last_probe_readout == ["EPOCH 2", f"TRAIN {fmt.loss(second.train_loss)}"]


def test_the_running_note_comes_only_while_training_runs(rig):
    rig.feeder.epochs(2)
    note = "the probe line clears at the next epoch"
    assert note in run(rig, "probe 1").output  # the session is running
    rig.feeder.state(TrainingState.PAUSED, "paused")
    assert note not in run(rig, "probe 1").output


@pytest.mark.parametrize(
    ("event", "kept"),
    [
        ("next epoch", False),
        ("resume after a pause", False),
        ("new session", False),
        ("settings change while running", True),
        ("settings change while paused", True),
        ("pause", True),
        ("finished", True),
        ("log line", True),
    ],
)
def test_when_the_probe_line_clears_by_itself(rig, event, kept):
    """It clears when training continues, and only then."""
    feeder = rig.feeder
    feeder.epochs(2)
    if event in ("resume after a pause", "settings change while paused"):
        feeder.paused()
        feeder.state(TrainingState.PAUSED, "paused")
    run(rig, "probe 1")
    assert rig.window.chart.probe_epoch == 1
    {
        "next epoch": lambda: feeder.train_epoch(),
        "resume after a pause": lambda: feeder.state(TrainingState.RUNNING, "resumed"),
        "new session": lambda: feeder.start(),
        "settings change while running": lambda: feeder.state(TrainingState.RUNNING, "lr set"),
        "settings change while paused": lambda: feeder.state(TrainingState.PAUSED, "lr set"),
        "pause": lambda: (feeder.paused(), feeder.state(TrainingState.PAUSED, "paused")),
        "finished": lambda: feeder.state(TrainingState.FINISHED, "done"),
        "log line": lambda: feeder.log("something"),
    }[event]()
    assert (rig.window.chart.probe_epoch == 1) is kept


# ---------------------------------------------------------------------- neuron
def test_neuron_lists_the_drawn_columns(rig):
    lines = run(rig, "neuron").output.splitlines()
    assert lines[0].startswith("drawn columns")
    table = "\n".join(lines[1:])
    for title, caption, indices in (("INPUT", "2 features", "0-1"), ("FC1", "8 neurons", "0-7"),
                                    ("FC2", "3 neurons", "0-2")):  # fmt: skip
        assert re.search(rf"{title}\s+{caption}\s+neurons {indices}", table), title


def test_neuron_with_a_column_lists_its_drawn_neurons(rig):
    assert run(rig, "neuron fc1").output == "FC1 · drawn neurons 0-7"
    assert run(rig, "neuron input").output == "INPUT · drawn neurons 0-1"


def test_neuron_highlights_and_reports_index_label_and_bias(make_rig):
    model = ht.track(TinyNet(), input_labels=("x", "y"), output_labels=("A", "B", "C"))
    rig = make_rig(setup=make_setup(model=model))
    rig.feeder.start()
    bias = float(model.fc2.bias.detach()[1])
    result = run(rig, "neuron fc2 1")
    assert result.ok and result.output.splitlines() == ["FC2 · neuron 1", "B", f"bias {bias:+.5f}"]
    view = rig.window.network
    assert view.selected_neuron == ("FC2", 1)
    view.repaint()
    assert view.last_selection_readout == ["FC2 · neuron 1", "B", f"bias {bias:+.5f}"]
    assert run(rig, "neuron input 0").output.splitlines() == ["INPUT · neuron 0", "x"]


def test_neuron_off(rig):
    assert run(rig, "neuron off").output == "no neuron highlighted"
    run(rig, "neuron fc1 2")
    assert run(rig, "neuron off").output == "neuron highlight removed"
    assert rig.window.network.selected_neuron is None


def test_neuron_rejects_an_unknown_column(rig):
    result = run(rig, "neuron zz 1")
    assert result.ok is False
    assert result.output == "no column 'zz' - columns: INPUT, FC1, FC2"


def test_neuron_rejects_neurons_that_are_not_drawn(make_rig):
    rig = make_rig(setup=make_setup(model=WideNet(200), features=3), max_neurons=20)
    rig.feeder.start()
    result = run(rig, "neuron fc1 1")
    assert result.ok is False
    assert "FC1 has no drawn neuron 1 - drawn: 0, 10, 21" in result.output
    assert run(rig, "neuron fc1 10").ok
    assert run(rig, "neuron fc1 199").ok  # the last neuron is always drawn
    assert run(rig, "neuron fc1 200").ok is False


@pytest.mark.parametrize("line", ["neuron fc1 -1", "neuron fc1 x", "neuron fc1 1 2"])
def test_neuron_rejects_bad_indices(rig, line):
    result = run(rig, line)
    assert result.ok is False
    assert rig.window.network.selected_neuron is None


@pytest.mark.parametrize("spelling", ["fc-in", "FC_IN", "'Fc In'", "fc_in"])
def test_neuron_column_names_ignore_case_and_separators(make_rig, spelling):
    rig = make_rig(setup=make_setup(model=ConvNet()))
    rig.feeder.start()
    assert run(rig, f"neuron {spelling} 0").ok
    assert rig.window.network.selected_neuron == ("FC IN", 0)


def test_neuron_before_a_network_is_drawn(make_rig):
    rig = make_rig()  # no session yet
    result = run(rig, "neuron")
    assert result.ok is False and result.output == "no network drawn yet"
    assert run(rig, "neuron off").output == "no neuron highlighted"


def test_the_highlight_stays_through_epochs_and_shows_the_new_bias(rig):
    run(rig, "neuron fc1 2")
    rig.feeder.epochs(2)
    view = rig.window.network
    assert view.selected_neuron == ("FC1", 2)
    view.repaint()
    bias = float(rig.setup.model.fc1.bias.detach()[2])
    assert view.last_selection_readout == ["FC1 · neuron 2", f"bias {bias:+.5f}"]


# ---------------------------------------------------------------------- layout
def test_layout_lists_every_panels_share(rig):
    output = run(rig, "layout").output
    for label, (panel_names, _shares) in zip(
        ("columns", "left column", "right column"), SPLITS.values(), strict=True
    ):
        line = next(text for text in output.splitlines() if text.startswith(label))
        for panel in panel_names:
            assert re.search(rf"{panel} \d+%", line), (label, panel)
    assert "'layout reset' restores" in output


def test_layout_sets_a_panels_share(rig):
    result = run(rig, "layout chart 40")
    assert result.ok and result.output.startswith("right column")
    assert abs(shares(rig, "right")["chart"] - 40) <= 2
    assert sum(shares(rig, "right").values()) in (99, 100, 101)


def test_layout_accepts_a_percent_sign_and_any_case(rig):
    assert run(rig, "layout network 50%").ok
    assert abs(shares(rig, "left")["network"] - 50) <= 2
    assert run(rig, "layout CHART 40").ok
    assert abs(shares(rig, "right")["chart"] - 40) <= 2


def test_layout_reports_one_panel(rig):
    share = shares(rig, "right")["chart"]
    assert run(rig, "layout chart").output == f"chart: {share}% of the right column"


def test_layout_reset_restores_the_initial_shares(rig):
    initial = rig.window.panel_shares()
    run(rig, "layout chart 40")
    run(rig, "layout left 55")
    result = run(rig, "layout reset")
    assert result.ok and result.output.startswith("layout reset")
    for name, values in initial.items():
        for panel, share in values.items():
            assert abs(shares(rig, name)[panel] - share) <= 1, (name, panel)


def test_layout_reset_takes_no_percentage(rig):
    result = run(rig, "layout reset 10")
    assert result.ok is False and "takes no percentage" in result.output


@pytest.mark.parametrize(
    "line",
    ["layout foo 50", "layout chart 0", "layout chart 100", "layout chart 101",
     "layout chart -5", "layout chart abc", "layout chart 40 50"],
)  # fmt: skip
def test_layout_rejects_unknown_panels_and_bad_percentages(rig, line):
    before = rig.window.panel_shares()
    result = run(rig, line)
    assert result.ok is False
    assert rig.window.panel_shares() == before


def test_layout_names_the_panels_when_one_is_unknown(rig):
    output = run(rig, "layout foo 50").output
    for panel in ("left", "right", "network", "terminal", "status", "chart", "usage"):
        assert panel in output


@pytest.mark.parametrize(
    ("line", "splitter", "panel"),
    [("layout left 95", "columns", "left"), ("layout chart 99", "right", "chart")],
)
def test_layout_respects_the_panels_minimum_sizes(rig, line, splitter, panel):
    result = run(rig, line)
    assert result.ok and "(limited by the panels' minimum sizes)" in result.output
    assert shares(rig, splitter)[panel] < int(line.split()[-1]) - 1
    window = rig.window
    for widget in (window.network, window.terminal, window.status, window.chart, window.usage):
        assert widget.width() >= widget.minimumWidth()
        assert widget.height() >= widget.minimumHeight()


# ------------------------------------------------------------------------ copy
def test_copy_copies_the_whole_log_without_its_own_echo(rig):
    rig.feeder.epochs(1)
    before = rig.log_lines()
    result = run(rig, "copy")
    assert result.ok and result.output == f"copied {len(before)} lines to the clipboard"
    assert QGuiApplication.clipboard().text() == "\n".join(before)


@pytest.mark.parametrize("count", [1, 3])
def test_copy_copies_the_last_lines(rig, count):
    rig.feeder.epochs(1)
    before = rig.log_lines()
    result = run(rig, f"copy {count}")
    noun = "line" if count == 1 else "lines"
    assert result.ok and result.output == f"copied {count} {noun} to the clipboard"
    assert QGuiApplication.clipboard().text() == "\n".join(before[-count:])


def test_copy_more_lines_than_the_log_has(rig):
    before = rig.log_lines()
    assert run(rig, "copy 100000").ok
    assert QGuiApplication.clipboard().text() == "\n".join(before)


@pytest.mark.parametrize("line", ["copy 0", "copy -1", "copy x", "copy 1 2"])
def test_copy_rejects_bad_counts(rig, line):
    QGuiApplication.clipboard().setText("unchanged")
    result = run(rig, line)
    assert result.ok is False
    assert QGuiApplication.clipboard().text() == "unchanged"


def test_copy_with_an_empty_log(rig):
    QGuiApplication.clipboard().setText("unchanged")
    run(rig, "clear")
    result = run(rig, "copy")
    assert result.ok and result.output == "nothing to copy: the log is empty"
    assert QGuiApplication.clipboard().text() == "unchanged"


def test_copy_keeps_any_characters(rig):
    text = "unicode → αβγ ✓ 'quotes' \"double\" <tags> & tabs"
    rig.feeder.log(text)
    assert run(rig, "copy 1").ok
    assert QGuiApplication.clipboard().text() == text


# ---------------------------------------------------------------------- window
def test_window_reports_the_current_mode(rig):
    window = rig.window
    expected = f"window: normal ({window.width()} × {window.height()})"
    assert run(rig, "window").output == expected


@pytest.mark.parametrize(
    ("mode", "flag", "answer"),
    [
        ("max", Qt.WindowState.WindowMaximized, "window: maximized"),
        ("full", Qt.WindowState.WindowFullScreen, "window: full screen"),
        ("min", Qt.WindowState.WindowMinimized, "window: minimized"),
    ],
)
def test_window_modes(rig, mode, flag, answer):
    result = run(rig, f"window {mode}")
    assert result.ok and result.output == answer
    assert rig.window.windowState() & flag
    normal = rig.enter("window normal")
    assert normal.ok and normal.output.startswith("window: normal")
    assert not rig.window.windowState() & flag


def test_window_rejects_an_unknown_mode(rig):
    result = run(rig, "window huge")
    assert result.ok is False and "normal, max, full, min" in result.output


# --------------------------------------------------------- the isolated window
@pytest.mark.subprocess
def test_view_requests_reach_the_isolated_window_process(probe):
    lines = [
        "scale log", "probe 1", "neuron fc1 1", "layout chart 40", "copy 2", "window max",
        "neuron zz 1", "window normal", "probe off", "resume",
    ]  # fmt: skip
    steps: list[dict] = [{"on": "epoch", "value": 2, "do": "keys", "keys": "pause<Return>"}]
    steps.append({"on": "state", "value": "paused", "do": "keys", "keys": "status<Return>",
                  "delay_ms": 300})  # fmt: skip
    for line in lines:
        steps.append({"on": "step", "value": len(steps) - 1, "do": "keys",
                      "keys": f"{line}<Return>", "delay_ms": 300})  # fmt: skip
    steps.append(LINGER_CLOSE)
    log = probe(script={"steps": steps})
    # ~30 ms per epoch: the run is still going when the typed "pause" arrives
    result = ht.run(make_setup(epochs=30, train_hook=lambda _b: time.sleep(0.005)).process)
    assert result.epochs_completed == 30
    views = [(v["action"], v["ok"]) for v in log.views()]
    for action in ("scale", "probe", "neuron", "layout", "copy", "window"):
        assert (action, True) in views, action
    refused = [v for v in log.views() if v["action"] == "neuron" and not v["ok"]]
    assert refused and "no column 'zz'" in refused[0]["text"]
    states = log.states()
    assert any(s["chart"]["log"] for s in states)
    assert any(s["chart"]["probe"] == 1 for s in states)
    assert any(s["network"]["selected"] == ["FC1", 1] for s in states)
    assert any("Maximized" in s["window_state"] for s in states)
    assert any(len(s["clipboard"].splitlines()) == 2 for s in states)
    assert any(any("no column 'zz'" in line for line in s["output"]["tail"]) for s in states), (
        "the refusal was not shown in the terminal"
    )
    right = [s["splitters"]["right"] for s in states]
    assert any(abs(100 * sizes[1] / sum(sizes) - 40) <= 2 for sizes in right)

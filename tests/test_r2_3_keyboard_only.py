"""R2 requirement 3 - "**ALL** implemented functionality **MUST** remain accessible and
operational without mouse interaction."

Every key here is typed through the window's native ``QWindow`` (the platform input path);
no mouse event is used anywhere. The first version's mouse interactions and the keyboard
path that replaces each one (the package's README, "Keyboard-driven control"):

==========================================  ===============================================
first version (mouse)                       keyboard path tested here
==========================================  ===============================================
click into the command line                 the command line has the focus without a click
click the loss chart (linear / log axis)    ``scale``, ``scale log``, ``scale lin``
hover the loss chart (an epoch's values)    ``probe <epoch>``
hover a neuron (its index, label, bias)     ``neuron <column> <index>``
drag the splitters (panel sizes)            ``layout <panel> <percent>``, ``layout reset``
wheel / scroll bar on the log               PgUp, PgDn, Ctrl+Home, Ctrl+End
select log text and copy it                 ``copy [lines]``
paste into the command line (menu)          Ctrl+V
the title bar's buttons                     ``window max|full|normal|min``, ``quit``
==========================================  ===============================================

Plus: every command is reachable by typing, the command line is editable by keyboard, the
help explains the keyboard use, and whole sessions run by keyboard alone - in a real
in-process run and in the real isolated window process.
"""

from __future__ import annotations

import time

import pytest
from htspec.gui_state import fill_log
from htspec.models import make_setup
from htspec.probe_log import LINGER_CLOSE
from PySide6.QtCore import Qt
from PySide6.QtGui import QGuiApplication

import hollywood_training as ht
from hollywood_training import formatting as fmt
from hollywood_training.sync.events import TrainingPaused, TrainingState

pytestmark = pytest.mark.req("R2.3")


def shares(window, splitter: str) -> dict[str, int]:
    return window.panel_shares()[splitter]


# ------------------------------------------------------------------ the replacements
def test_the_command_line_takes_keys_without_a_click(make_rig):
    rig = make_rig()
    rig.feeder.start()
    assert rig.focus_widget() is rig.window.terminal.input
    result = rig.type_line("status")
    assert result is not None and result.ok and result.command.name == "status"


@pytest.mark.req("R2.4")
def test_the_loss_axis_switches_by_keyboard(rig, paint_capture):
    rig.feeder.epochs(3)
    chart = rig.window.chart
    assert rig.type_line("scale log").ok
    assert chart.log_scale is True
    assert "LOG" in paint_capture.render(chart)
    assert rig.type_line("scale lin").ok
    assert chart.log_scale is False
    assert "LIN" in paint_capture.render(chart)
    rig.type_line("scale")
    assert chart.log_scale is True  # without an argument it switches


@pytest.mark.req("R2.4")
def test_an_epochs_values_are_read_out_by_keyboard(rig):
    records = rig.feeder.epochs(3)
    rig.feeder.state(TrainingState.PAUSED, "paused")
    result = rig.type_line("probe 2")
    assert result is not None and result.ok, result
    record = records[1].record
    assert "epoch 2 of 10" in result.output
    assert fmt.loss(record.train_loss) in result.output
    chart = rig.window.chart
    assert chart.probe_epoch == 2
    chart.repaint()
    assert chart.last_probe_readout == [
        "EPOCH 2",
        f"TRAIN {fmt.loss(record.train_loss)}",
        f"VAL   {fmt.loss(record.val_loss)}",
    ]


@pytest.mark.req("R2.4")
def test_a_neurons_details_are_shown_by_keyboard(rig):
    rig.feeder.epochs(2)
    result = rig.type_line("neuron fc1 3")
    assert result is not None and result.ok, result
    bias = float(rig.setup.model.fc1.bias.detach()[3])
    assert "FC1 · neuron 3" in result.output
    assert f"bias {bias:+.5f}" in result.output
    view = rig.window.network
    assert view.selected_neuron == ("FC1", 3)
    view.repaint()
    assert view.last_selection_readout == ["FC1 · neuron 3", f"bias {bias:+.5f}"]


@pytest.mark.req("R2.4")
def test_panel_sizes_change_by_keyboard(rig):
    window = rig.window
    initial = {name: dict(values) for name, values in window.panel_shares().items()}
    assert rig.type_line("layout chart 40").ok
    assert abs(shares(window, "right")["chart"] - 40) <= 2
    assert rig.type_line("layout network 50").ok
    assert abs(shares(window, "left")["network"] - 50) <= 2
    assert rig.type_line("layout left 55").ok
    assert abs(shares(window, "columns")["left"] - 55) <= 2
    assert rig.type_line("layout reset").ok
    for name, values in initial.items():
        for panel, share in values.items():
            assert abs(shares(window, name)[panel] - share) <= 1, (name, panel)


def test_the_log_scrolls_by_keyboard(rig):
    fill_log(rig.window)
    bar = rig.window.terminal.output.verticalScrollBar()
    assert bar.maximum() > 0 and bar.value() == bar.maximum()
    rig.type("<PgUp>")
    first = bar.value()
    assert first < bar.maximum()
    rig.type("<PgUp>")
    second = bar.value()
    assert second < first
    rig.type("<PgDn>")
    assert bar.value() > second  # moved back down
    rig.type("<C-Home>")
    assert bar.value() == 0
    rig.type("<C-End>")
    assert bar.value() == bar.maximum()


@pytest.mark.req("R2.4")
def test_log_text_is_copied_by_keyboard(rig):
    rig.feeder.epochs(1)
    rig.feeder.log("first copied line")
    rig.feeder.log("second copied line")
    before = rig.log_lines()
    result = rig.type_line("copy 2")
    assert result is not None and result.ok, result
    assert QGuiApplication.clipboard().text() == "\n".join(before[-2:])
    assert rig.type_line("copy").ok
    assert QGuiApplication.clipboard().text().splitlines()[-1] == "copied 2 lines to the clipboard"


@pytest.mark.req("R2.4")
def test_window_modes_change_by_keyboard(rig):
    window = rig.window
    state = Qt.WindowState
    assert rig.type_line("window max").ok
    assert window.windowState() & state.WindowMaximized
    assert rig.type_line("window full").ok
    assert window.windowState() & state.WindowFullScreen
    assert rig.type_line("window normal").ok
    assert not window.windowState() & (state.WindowMaximized | state.WindowFullScreen)
    assert rig.type_line("window min").ok
    assert window.windowState() & state.WindowMinimized
    assert rig.enter("window normal").ok  # the window comes back from the keyboard path too
    assert not window.windowState() & state.WindowMinimized


def test_the_window_is_closed_by_keyboard_in_a_real_run(pilot):
    pilot.on_epoch(2, pilot.line("quit"))
    result = pilot.run(make_setup(epochs=200).process)
    assert result.state is TrainingState.STOPPED
    assert result.epochs_completed < 200
    assert "stopping and closing" in pilot.final_log


def test_pasting_into_the_command_line_by_keyboard(rig):
    QGuiApplication.clipboard().setText("status")
    rig.type("<C-v>")
    assert rig.input_text == "status"
    rig.type("<Return>")
    assert rig.results[-1][1].command.name == "status"


def test_the_command_line_is_edited_by_keyboard(rig):
    rig.type("satus")
    rig.type("<Home><Right>t")
    assert rig.input_text == "status"
    rig.type("<End><Backspace>")
    assert rig.input_text == "statu"
    rig.type("<Left><Delete>")
    assert rig.input_text == "stat"
    rig.type("<C-a>help")
    assert rig.input_text == "help"
    rig.type("<Return>")
    assert rig.results[-1][1].command.name == "help"
    assert rig.input_text == ""


TYPED_COMMANDS = [
    ("help", "help"), ("?", "help"), ("help lr", "help"), ("status", "status"),
    ("pause", "pause"), ("resume", "resume"), ("continue", "resume"), ("start", "resume"),
    ("step", "step"), ("step 2", "step"), ("epochs 20", "epochs"), ("lr 0.01", "lr"),
    ("delay 0", "delay"), ("delay 5", "delay"), ("history", "history"), ("history 1", "history"),
    ("arch", "arch"), ("layers", "arch"), ("architecture", "arch"), ("weights", "weights"),
    ("weights fc1", "weights"), ("scale", "scale"), ("scale log", "scale"), ("probe", "probe"),
    ("probe off", "probe"), ("neuron", "neuron"), ("neuron fc2", "neuron"), ("layout", "layout"),
    ("layout reset", "layout"), ("copy", "copy"), ("window", "window"), ("clear", "clear"),
    ("cls", "clear"), ("stop", "stop"),
]  # fmt: skip


@pytest.mark.parametrize(("line", "name"), TYPED_COMMANDS, ids=[c[0] for c in TYPED_COMMANDS])
def test_every_command_is_reachable_by_typing(rig, line, name):
    rig.feeder.epochs(2)
    result = rig.type_line(line)
    assert result is not None, f"typing {line!r} ran no command"
    assert result.command is not None and result.command.name == name
    assert result.ok, result.output


@pytest.mark.parametrize("line", ["quit", "exit"])
def test_quit_is_reachable_by_typing(rig, line):
    result = rig.type_line(line)
    assert result is not None and result.ok and result.command.name == "quit"
    assert rig.control.stop_requested
    rig.process_events(100)
    assert not rig.window.isVisible()


@pytest.mark.parametrize("command", ["save", "screenshot"])
def test_file_commands_are_reachable_by_typing(rig, tmp_path, command):
    target = tmp_path / f"by-keyboard-{command}.out"
    result = rig.type_line(f"{command} {target}")
    assert result is not None and result.ok and result.command.name == command, result
    if command == "screenshot":
        assert target.exists() and target.stat().st_size > 0


def test_custom_commands_are_reachable_by_typing(make_rig):
    setup = make_setup()

    @setup.process.command("double", summary="twice a number", arguments=[ht.Argument("n", int)])
    def double(_context, n):
        return str(2 * n)

    rig = make_rig(setup=setup)
    rig.feeder.start()
    result = rig.type_line("double 21")
    assert result is not None and result.ok and result.output == "42"
    assert "42" in rig.log_lines()[-1]


def test_the_help_explains_the_keyboard_use(rig):
    result = rig.type_line("help")
    assert result is not None and result.ok
    text = result.output
    assert "Tab" in text
    assert any(name in text for name in ("PgUp", "Page Up", "PageUp")), "the scroll keys"
    for command in ("scale", "probe", "neuron", "layout", "copy", "window"):
        assert f"\n  {command}" in text, command


# ------------------------------------------------------------------ real sessions
def test_training_is_controlled_by_keyboard_in_a_real_run(pilot):
    pauses = []

    def on_pause(window) -> None:
        pauses.append(1)
        command = "step 2" if len(pauses) == 1 else "resume"
        pilot.after(0.2, pilot.line(command))(window)

    pilot.on_epoch(1, pilot.chain(pilot.line("lr 0.02"), pilot.line("pause")))
    pilot.on_paused(on_pause, once=False)  # once per pause that takes effect
    pilot.on_finish(pilot.after(0.2, pilot.close))
    result = pilot.run(make_setup(epochs=6).process)
    assert result.epochs_completed == 6
    assert len(pilot.of_type(TrainingPaused)) == 2  # the pause, then the end of "step 2"
    assert len(pauses) == 2
    assert all(record.learning_rate == pytest.approx(0.02) for record in result.history[2:])


@pytest.mark.subprocess
def test_a_whole_session_by_keyboard_in_the_isolated_window(probe):
    lines = [
        "scale log", "probe 1", "neuron fc1 1", "layout chart 40", "copy 3", "window max",
        "window normal", "history 2", "resume",
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
    views = {(v["action"], v["ok"]) for v in log.views()}
    for action in ("scale", "probe", "neuron", "layout", "copy", "window"):
        assert (action, True) in views, action
    states = log.states()
    assert any(s["chart"]["log"] for s in states)
    assert any(s["chart"]["probe"] == 1 for s in states)
    assert any(s["network"]["selected"] == ["FC1", 1] for s in states)
    assert any("Maximized" in s["window_state"] for s in states)
    assert any(len(s["clipboard"].splitlines()) == 3 for s in states)
    assert any(s["session"]["state"] == "paused" for s in states)
    assert log.errors() == []

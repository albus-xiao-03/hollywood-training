"""R1 GUI layout 5 - "The GUI **MUST** provide a integrated terminal to allow users to
control the managed training process with defined commands."

* integrated: the terminal (a log plus a command line) is part of the window;
* control: lines typed into it run the defined commands - in a real in-process run they
  change the run (total epochs, learning rate, stop), in the default window process they
  reach the training process; every line is echoed and answered in the terminal, errors in
  their own colour; the session reports there too (start, epochs, state changes);
* a terminal: history (Up / Down), Tab completion, empty lines ignored, a bounded log,
  ``clear``, any characters.
"""

from __future__ import annotations

import time

import pytest
from htspec.models import make_setup
from htspec.probe_log import LINGER_CLOSE
from PySide6.QtCore import QEvent, Qt
from PySide6.QtGui import QGuiApplication, QKeyEvent
from PySide6.QtWidgets import QLineEdit, QPlainTextEdit

import hollywood_training as ht
from hollywood_training.gui.theme import PALETTE
from hollywood_training.gui.widgets import TerminalPanel
from hollywood_training.sync.events import LogLevel, TrainingState

pytestmark = pytest.mark.req("R1.G5")


def block_colors(output: QPlainTextEdit) -> list[tuple[str, str]]:
    """``(text, foreground colour)`` of every line of the log."""
    found = []
    block = output.document().firstBlock()
    while block.isValid():
        fragment = block.begin().fragment()
        color = fragment.charFormat().foreground().color().name() if fragment.isValid() else ""
        found.append((block.text(), color))
        block = block.next()
    return found


# --------------------------------------------------------------- integrated
def test_the_terminal_is_part_of_the_window(rig):
    terminal = rig.window.terminal
    assert isinstance(terminal, TerminalPanel)
    assert rig.window.isAncestorOf(terminal) and terminal.isVisible()
    assert isinstance(terminal.output, QPlainTextEdit) and terminal.output.isReadOnly()
    assert isinstance(terminal.input, QLineEdit) and terminal.input.isVisible()


def test_a_typed_command_is_echoed_and_answered_in_the_terminal(rig):
    result = rig.type_line("status")
    assert result is not None and result.ok
    lines = rig.log_lines()
    echo = lines.index("ht ❯ status")
    answer = lines[echo + 1 :]
    assert any(line.startswith("state       running") for line in answer)


def test_answers_and_errors_are_told_apart_by_colour(rig):
    rig.type_line("status")
    rig.type_line("xyzzy")
    colors = dict(block_colors(rig.window.terminal.output))
    assert colors["ht ❯ status"] == PALETTE.accent  # the echo
    assert colors["network     TinyNet"] == PALETTE.text  # an answer
    error = next(text for text in colors if text.startswith("unknown command 'xyzzy'"))
    assert colors[error] == PALETTE.error


def test_the_session_reports_in_the_terminal(rig):
    rig.feeder.epochs(2)
    rig.feeder.state(TrainingState.PAUSED, "paused before epoch 3")
    rig.feeder.log("careful", LogLevel.WARNING)
    lines = rig.log_lines()
    assert lines[0] == "▓▒░ HOLLYWOOD TRAINING :: COMMAND TERMINAL ░▒▓"
    assert "network TinyNet · 51 parameters · 2 connection layers" in lines
    settings = "epochs 10 · batch size 16 · loss CrossEntropyLoss"
    assert any(line.startswith(settings) for line in lines)
    assert any(line.startswith("epoch  1/10 | train") for line in lines)
    assert any(line.startswith("epoch  2/10 | train") for line in lines)
    colors = dict(block_colors(rig.window.terminal.output))
    assert colors["paused before epoch 3"] == PALETTE.warning
    assert colors["careful"] == PALETTE.warning


# ------------------------------------------------------------------- control
def test_typed_commands_control_a_real_run(pilot):
    setup = make_setup(epochs=4)
    # typed while each epoch is being shown, i.e. before the next one can start
    pilot.on_epoch(1, pilot.line("epochs 6"), sync=True)
    pilot.on_epoch(2, pilot.line("lr 0.07"), sync=True)
    pilot.on_epoch(4, pilot.line("stop"), sync=True)
    pilot.on_finish(pilot.after(0.3, pilot.close))
    result = pilot.run(setup.process)
    assert result.state is ht.TrainingState.STOPPED
    assert result.epochs_completed == 4  # the total grew to 6, then "stop" ended it
    assert result.history[-1].learning_rate == pytest.approx(0.07)
    log = pilot.final_log
    for line in ("ht ❯ epochs 6", "total epochs set to 6", "ht ❯ lr 0.07",
                 "learning rate set to 0.07", "ht ❯ stop"):  # fmt: skip
        assert line in log, line


def test_a_real_run_reports_in_the_terminal(pilot):
    pilot.on_finish(pilot.after(0.2, pilot.close))
    pilot.run(make_setup(epochs=3).process)
    log = pilot.final_log
    assert "training TinyNet for 3 epochs on cpu" in log
    assert "epoch 1/3 | train" in log and "epoch 3/3 | train" in log
    assert "all 3 epochs complete" in log


@pytest.mark.subprocess
def test_the_terminal_of_the_window_process_controls_the_run(probe):
    steps = [
        {"on": "epoch", "value": 1, "do": "keys", "keys": "status<Return>"},
        {"on": "step", "value": 0, "do": "keys", "keys": "pause<Return>", "delay_ms": 200},
        {"on": "state", "value": "paused", "do": "keys", "keys": "resume<Return>", "delay_ms": 400},
        LINGER_CLOSE,
    ]
    log = probe(script={"steps": steps})
    result = ht.run(make_setup(epochs=40, train_hook=lambda _b: time.sleep(0.005)).process)
    assert result.epochs_completed == 40
    assert [r["value"] for r in log.sends("command")] == ["status", "pause", "resume"]
    tails = [line for state in log.states() for line in state["output"]["tail"]]
    assert "ht ❯ status" in tails and any(line.startswith("state       ") for line in tails)
    assert any(state["session"]["state"] == "paused" for state in log.states())


# ---------------------------------------------------------- terminal behaviour
def test_history_recalls_the_entered_lines(rig):
    for line in ("status", "status", "help"):
        rig.type_line(line)
    assert rig.window.terminal.input.history == ["status", "help"]  # repeats kept once
    for key, expected in (("<Up>", "help"), ("<Up>", "status"), ("<Up>", "status"),
                          ("<Down>", "help"), ("<Down>", "")):  # fmt: skip
        rig.type(key)
        assert rig.input_text == expected, key


def test_tab_completes_command_names(rig):
    rig.type("pa<Tab>")
    assert rig.input_text == "pause "
    rig.type("<C-a><Backspace>scr<Tab>")
    assert rig.input_text == "screenshot "
    rig.type("<C-a><Backspace>s<Tab>")
    assert rig.input_text == "s"  # ambiguous: the candidates are listed instead
    candidates = rig.log_lines()[-1]
    for name in ("save", "scale", "status", "step", "stop"):
        assert name in candidates.split(), name
    rig.type("<C-a><Backspace>lr 0<Tab>")
    assert rig.input_text == "lr 0"  # no completion inside the arguments
    rig.type("<C-a><Backspace>xyz<Tab>")
    assert rig.input_text == "xyz"


@pytest.mark.parametrize("line", ["", "   ", "\t"])
def test_empty_lines_are_ignored(rig, line):
    before = (len(rig.log_lines()), len(rig.results))
    rig.window.terminal.input.setText(line)
    rig.type("<Return>")
    assert (len(rig.log_lines()), len(rig.results)) == before


def test_the_log_keeps_the_latest_lines(rig):
    terminal = rig.window.terminal
    for index in range(4500):
        terminal.write(f"line {index}")
    lines = rig.log_lines()
    assert len(lines) <= 4000
    assert lines[-1] == "line 4499" and "line 0" not in lines


def test_clear_empties_the_terminal(rig):
    rig.type_line("help")
    assert rig.log_lines()
    rig.type_line("clear")
    assert rig.log() == ""


def test_any_characters_can_be_typed(rig):
    text = "αβγ ✓ → ü"
    handle = rig.window.windowHandle()
    for char in text:
        for kind in (QEvent.Type.KeyPress, QEvent.Type.KeyRelease):
            event = QKeyEvent(kind, Qt.Key.Key_unknown, Qt.KeyboardModifier.NoModifier, char)
            QGuiApplication.sendEvent(handle, event)
    assert rig.input_text == text

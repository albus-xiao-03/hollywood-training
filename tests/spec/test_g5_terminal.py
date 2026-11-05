"""G5: the GUI has an integrated terminal through which the user controls the managed
training with the defined commands (R7)."""

from __future__ import annotations

import pytest
from PySide6.QtCore import Qt

import hollywood_training as ht
from hollywood_training.sync import TrainingState

from .helpers import build_process, type_command

pytestmark = [pytest.mark.spec("G5"), pytest.mark.gui]


def test_typing_a_command_submits_and_echoes_it(gui):
    lines: list[str] = []
    gui.window.command_submitted.connect(lines.append)
    type_command(gui.window, "  status  ")
    assert lines == ["status"]
    assert "status" in gui.window.terminal.text()  # echoed after the prompt
    assert gui.window.terminal.input.text() == ""  # ready for the next command


def test_empty_input_is_ignored(gui):
    lines: list[str] = []
    gui.window.command_submitted.connect(lines.append)
    type_command(gui.window, "   ")
    assert lines == []


def test_results_and_errors_are_shown(gui):
    gui.window.show_command_result(True, "all good")
    gui.window.show_command_result(False, "bad input")
    text = gui.window.terminal.text()
    assert "all good" in text and "bad input" in text


def test_the_terminal_introduces_the_session(gui):
    text = gui.window.terminal.text()
    assert "TinyNet" in text and "help" in text


def test_history_is_browsable_with_the_arrow_keys(gui, qtbot):
    field = gui.window.terminal.input
    for line in ("status", "history", "history"):
        type_command(gui.window, line)
    qtbot.keyClick(field, Qt.Key.Key_Up)
    assert field.text() == "history"  # repeated lines are kept once
    qtbot.keyClick(field, Qt.Key.Key_Up)
    assert field.text() == "status"
    qtbot.keyClick(field, Qt.Key.Key_Up)
    assert field.text() == "status"  # stays at the oldest
    qtbot.keyClick(field, Qt.Key.Key_Down)
    assert field.text() == "history"
    qtbot.keyClick(field, Qt.Key.Key_Down)
    assert field.text() == ""


@pytest.mark.parametrize(
    ("typed", "expected"),
    [("hist", "history "), ("scr", "screenshot "), ("st", "st"), ("zzz", "zzz")],
)
def test_tab_completes_command_names(gui, qtbot, typed, expected):
    field = gui.window.terminal.input
    field.clear()
    qtbot.keyClicks(field, typed)
    qtbot.keyClick(field, Qt.Key.Key_Tab)
    assert field.text() == expected


def test_tab_lists_the_candidates_when_several_match(gui, qtbot):
    field = gui.window.terminal.input
    qtbot.keyClicks(field, "st")
    qtbot.keyClick(field, Qt.Key.Key_Tab)
    listed = gui.window.terminal.text().splitlines()[-1]
    assert {"start", "status", "step", "stop"} <= set(listed.split())


def test_clear_empties_the_terminal(gui):
    gui.window.terminal.write("something to forget")
    gui.window.terminal.clear()
    assert gui.window.terminal.text() == ""


def test_the_terminal_answers_and_controls_a_live_run(drive):
    bundle = build_process(epochs=2)
    driver = drive(
        [
            ("paused", lambda w: w.session.state is TrainingState.PAUSED,
             lambda w: type_command(w, "status")),
            ("status shown", lambda w: "state       paused" in w.terminal.text(),
             lambda w: type_command(w, "frobnicate")),
            ("error shown", lambda w: "unknown command" in w.terminal.text(),
             lambda w: type_command(w, "quit")),
        ],
        timeout=30,
    )  # fmt: skip
    result = ht.run(bundle.process, isolate_gui=False, start_paused=True)
    driver.check()
    assert result.state is TrainingState.STOPPED and result.epochs_completed == 0


def test_the_window_process_terminal_shows_results_and_runs_window_actions(
    window_process, tmp_path
):
    harness = window_process
    assert harness.receive() == ("ready",)
    harness.send("completions", ["help", "history", "status"])
    harness.send("result", True, "status: all fine")
    harness.send("result", False, "unknown command 'x'")
    harness.wait_for_text("unknown command 'x'")
    assert "status: all fine" in harness.window.terminal.text()

    field = harness.window.terminal.input
    harness.qtbot.keyClicks(field, "hi")
    harness.qtbot.keyClick(field, Qt.Key.Key_Tab)
    assert field.text() == "history "  # names supplied by the training process

    shot = tmp_path / "remote" / "shot.png"
    harness.send("frontend", "screenshot", str(shot))
    harness.qtbot.waitUntil(shot.exists, timeout=5000)
    assert shot.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"

    harness.send("frontend", "clear", None)
    harness.qtbot.waitUntil(lambda: harness.window.terminal.text() == "", timeout=5000)

    harness.send("close")
    assert harness.receive(timeout=5) == ("closed",)
    assert not harness.window.isVisible()

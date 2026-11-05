"""Keyboard focus (SPEC R2.7): while the window is active, every key reaches the terminal."""

from __future__ import annotations

import pytest
from PySide6.QtCore import QEvent, QObject, Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QLineEdit, QWidget
from shiboken6 import isValid

from hollywood_training.commands import create_registry
from hollywood_training.gui import MainWindow
from hollywood_training.gui.keyboard_focus import KeyboardFocus
from hollywood_training.sync import SessionState


def activate(qtbot, widget: QWidget) -> None:
    widget.activateWindow()
    qtbot.waitUntil(lambda: QApplication.activeWindow() is widget)


@pytest.fixture
def window(qtbot):
    win = MainWindow(SessionState(), completer=create_registry().complete)
    qtbot.addWidget(win)
    with qtbot.waitExposed(win):
        win.show()
    activate(qtbot, win)
    return win


def test_the_command_line_is_the_only_focus_target(window):
    line = window.terminal.input
    others = [widget for widget in window.findChildren(QWidget) if widget is not line]
    assert others and all(w.focusPolicy() == Qt.FocusPolicy.NoFocus for w in others)
    assert window.focusPolicy() == Qt.FocusPolicy.NoFocus
    assert window.focusProxy() is line
    assert isinstance(window.keyboard, KeyboardFocus) and window.keyboard.target is line
    assert QApplication.focusWidget() is line


def test_tab_keys_never_move_the_focus(window, qtbot):
    line = window.terminal.input
    qtbot.keyClick(line, Qt.Key.Key_Backtab, Qt.KeyboardModifier.ShiftModifier)
    assert QApplication.focusWidget() is line
    qtbot.keyClicks(line, "hist")
    qtbot.keyClick(line, Qt.Key.Key_Tab)
    assert QApplication.focusWidget() is line and line.text() == "history "


def test_the_focus_is_handed_back_to_the_command_line(window, qtbot):
    line = window.terminal.input
    window.terminal.output.setFocus()  # setFocus() ignores the focus policy
    assert QApplication.focusWidget() is line
    qtbot.mouseClick(window.terminal.output.viewport(), Qt.MouseButton.LeftButton)
    assert QApplication.focusWidget() is line


def test_keys_sent_anywhere_in_the_window_reach_the_command_line(window, qtbot):
    entered: list[str] = []
    window.command_submitted.connect(entered.append)
    qtbot.keyClicks(window.network, "st")  # a panel
    qtbot.keyClicks(window, "at")  # the window itself
    qtbot.keyClicks(window.terminal.output, "us")  # the log
    assert window.terminal.input.text() == "status"
    qtbot.keyClick(window.chart, Qt.Key.Key_Return)
    assert entered == ["status"] and "ht ❯ status" in window.terminal.text()


def test_keys_from_the_window_system_reach_the_command_line(window):
    for key in (Qt.Key.Key_H, Qt.Key.Key_I):
        QTest.keyClick(window.windowHandle(), key)
    assert window.terminal.input.text() == "hi"


def test_the_command_line_consumes_every_key(window, qtbot):
    seen: list[QEvent.Type] = []

    class Spy(QObject):
        def eventFilter(self, watched, event):
            if event.type() in (QEvent.Type.KeyPress, QEvent.Type.KeyRelease):
                seen.append(event.type())
            return False

    spy = Spy()
    window.terminal.installEventFilter(spy)  # the panel around the command line
    for key in (Qt.Key.Key_F5, Qt.Key.Key_Escape, Qt.Key.Key_A, Qt.Key.Key_Return):
        qtbot.keyClick(window.terminal.input, key)
    assert seen == []


def test_widgets_added_later_cannot_take_the_focus(window, qtbot):
    extra = QLineEdit(window.status)
    extra.show()  # polished: adopted by the focus lock
    assert extra.focusPolicy() == Qt.FocusPolicy.NoFocus
    qtbot.keyClicks(extra, "lr")
    assert extra.text() == "" and window.terminal.input.text() == "lr"


def test_other_windows_keep_their_keyboard(window, qtbot):
    host = QLineEdit()  # e.g. a host application's window (in-process frontend)
    qtbot.addWidget(host)
    with qtbot.waitExposed(host):
        host.show()
    activate(qtbot, host)
    host.setFocus()
    for key in (Qt.Key.Key_A, Qt.Key.Key_B):
        QTest.keyClick(host.windowHandle(), key)
    assert host.text() == "ab" and window.terminal.input.text() == ""
    assert QApplication.focusWidget() is host
    activate(qtbot, window)
    assert QApplication.focusWidget() is window.terminal.input


def test_a_deleted_window_leaves_focus_handling_alone(qtbot):
    win = MainWindow(SessionState())
    win.show()
    win.close()
    win.deleteLater()
    qtbot.waitUntil(lambda: not isValid(win))
    host = QLineEdit()
    qtbot.addWidget(host)
    with qtbot.waitExposed(host):
        host.show()
    activate(qtbot, host)
    host.setFocus()  # focusChanged fires; nothing may refer to the deleted window
    assert QApplication.focusWidget() is host


def test_the_target_must_be_inside_the_window(qtbot):
    win = MainWindow(SessionState())
    qtbot.addWidget(win)
    stranger = QLineEdit()
    qtbot.addWidget(stranger)
    with pytest.raises(ValueError, match="inside the window"):
        KeyboardFocus(win, stranger)

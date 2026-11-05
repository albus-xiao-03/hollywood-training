"""``TerminalPanel``: the integrated command terminal.

An output log plus a command line with history (Up / Down) and Tab completion of command
names. PgUp / PgDn and Ctrl+Home / Ctrl+End, typed in the command line, scroll the log.
Entered lines are emitted as :attr:`TerminalPanel.command_entered`; the main window
echoes them and passes them on to its frontend, which executes them and writes the result
back with :meth:`TerminalPanel.write`.

The command line is the window's only keyboard focus target
(:mod:`~hollywood_training.gui.keyboard_focus`): Tab and Shift+Tab never move the focus,
and it consumes every key, so no key travels on to the panels around it.
"""

from __future__ import annotations

from collections.abc import Callable

from PySide6.QtCore import QEvent, Qt, Signal
from PySide6.QtGui import QTextCharFormat, QTextCursor
from PySide6.QtWidgets import (
    QAbstractSlider,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QVBoxLayout,
    QWidget,
)

from ..theme import PALETTE, color, mono_font
from .hud import HudPanel

LEVEL_COLORS = {
    "info": PALETTE.text,
    "success": PALETTE.success,
    "warning": PALETTE.warning,
    "error": PALETTE.error,
    "command": PALETTE.accent,
    "dim": PALETTE.text_dim,
    "accent": PALETTE.ice,
}

PROMPT = "ht ❯"

# Keys of the command line that scroll the log (Shift is ignored), as scroll-bar actions.
PAGE_KEYS = {
    Qt.Key.Key_PageUp: QAbstractSlider.SliderAction.SliderPageStepSub,
    Qt.Key.Key_PageDown: QAbstractSlider.SliderAction.SliderPageStepAdd,
}
EDGE_KEYS = {  # with Ctrl
    Qt.Key.Key_Home: QAbstractSlider.SliderAction.SliderToMinimum,
    Qt.Key.Key_End: QAbstractSlider.SliderAction.SliderToMaximum,
}


def _common_prefix(words: list[str]) -> str:
    """Longest string every word starts with."""
    shortest = min(words, key=len)
    for index, char in enumerate(shortest):
        if any(word[index] != char for word in words):
            return shortest[:index]
    return shortest


class CommandLine(QLineEdit):
    """Single-line input with history, Tab completion and scroll keys for the log."""

    submitted = Signal(str)
    candidates = Signal(list)
    scroll_requested = Signal(object)  # a QAbstractSlider.SliderAction for the log

    def __init__(
        self, completer: Callable[[str], list[str]], parent: QWidget | None = None
    ) -> None:
        super().__init__(parent)
        self.setObjectName("terminalInput")
        self.setPlaceholderText("type a command - 'help' lists them, Tab completes")
        self._completer = completer
        self._history: list[str] = []
        self._cursor = 0
        self.returnPressed.connect(self._submit)

    @property
    def history(self) -> list[str]:
        return list(self._history)

    def _submit(self) -> None:
        text = self.text().strip()
        if not text:
            return
        if not self._history or self._history[-1] != text:
            self._history.append(text)
        self._cursor = len(self._history)
        self.clear()
        self.submitted.emit(text)

    def event(self, event) -> bool:
        # Tab and Shift+Tab normally move the focus. Here Tab completes the command name and
        # Shift+Tab does nothing: the command line is the window's only focus target.
        if event.type() == QEvent.Type.KeyPress and event.key() in (
            Qt.Key.Key_Tab,
            Qt.Key.Key_Backtab,
        ):
            if event.key() == Qt.Key.Key_Tab:
                self.complete()
            return True
        return super().event(event)

    def keyPressEvent(self, event) -> None:
        key = event.key()
        if key in PAGE_KEYS:
            self.scroll_requested.emit(PAGE_KEYS[key])
        elif key in EDGE_KEYS and event.modifiers() & Qt.KeyboardModifier.ControlModifier:
            self.scroll_requested.emit(EDGE_KEYS[key])
        elif key == Qt.Key.Key_Up and self._history:
            self._cursor = max(0, self._cursor - 1)
            self.setText(self._history[self._cursor])
        elif key == Qt.Key.Key_Down and self._history:
            self._cursor = min(len(self._history), self._cursor + 1)
            self.setText(self._history[self._cursor] if self._cursor < len(self._history) else "")
        else:
            super().keyPressEvent(event)
        event.accept()  # consumed here: keys never travel on to the panels

    def keyReleaseEvent(self, event) -> None:
        super().keyReleaseEvent(event)
        event.accept()

    def complete(self) -> None:
        text = self.text()
        if not text.strip() or " " in text.strip():
            return
        matches = self._completer(text.strip())
        if len(matches) == 1:
            self.setText(matches[0] + " ")
        elif matches:
            common = _common_prefix(matches)
            if len(common) > len(text.strip()):
                self.setText(common)
            self.candidates.emit(matches)


class TerminalPanel(HudPanel):
    """Log output + command input."""

    command_entered = Signal(str)

    def __init__(
        self, completer: Callable[[str], list[str]], parent: QWidget | None = None
    ) -> None:
        super().__init__("Command terminal", "CMD-05", parent)
        self.output = QPlainTextEdit(self)
        self.output.setObjectName("terminalOutput")
        self.output.setReadOnly(True)
        self.output.setMaximumBlockCount(4000)
        self.output.setFrameShape(QFrame.Shape.NoFrame)
        self.output.setFont(mono_font(9.0))
        self.output.setLineWrapMode(QPlainTextEdit.LineWrapMode.WidgetWidth)

        prompt = QLabel(PROMPT, self)
        prompt.setObjectName("terminalPrompt")
        prompt.setFont(mono_font(10.0, bold=True))
        self.input = CommandLine(completer, self)
        self.input.setFont(mono_font(10.0))

        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(8)
        row.addWidget(prompt)
        row.addWidget(self.input, 1)
        layout = QVBoxLayout(self)
        layout.setSpacing(6)
        layout.addWidget(self.output, 1)
        layout.addLayout(row)

        self.input.submitted.connect(self.command_entered)
        self.input.candidates.connect(lambda names: self.write("   ".join(names), "dim"))
        self.input.scroll_requested.connect(self.scroll)
        self.setMinimumSize(360, 150)

    def write(self, text: str, level: str = "info") -> None:
        """Append ``text`` (may span lines) in the colour of ``level``."""
        if not text:
            return
        scrollbar = self.output.verticalScrollBar()
        at_bottom = scrollbar.value() >= scrollbar.maximum() - 4
        cursor = self.output.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        style = QTextCharFormat()
        style.setForeground(color(LEVEL_COLORS.get(level, PALETTE.text)))
        if not self.output.document().isEmpty():
            cursor.insertBlock()
        cursor.insertText(text, style)
        if at_bottom:
            scrollbar.setValue(scrollbar.maximum())

    def scroll(self, action: QAbstractSlider.SliderAction) -> None:
        """Scroll the log as its scroll bar would (a page up / down, to the top / bottom)."""
        self.output.verticalScrollBar().triggerAction(action)

    def echo(self, command: str) -> None:
        self.write(f"{PROMPT} {command}", "command")

    def clear(self) -> None:
        self.output.clear()

    def text(self) -> str:
        return self.output.toPlainText()

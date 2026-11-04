"""``TerminalPanel``: the integrated command terminal.

An output log plus a command line with history (Up / Down) and Tab completion of command
names. Entered lines are emitted as :attr:`TerminalPanel.command_entered`; the main window
echoes them and passes them on to its frontend, which executes them and writes the result
back with :meth:`TerminalPanel.write`.
"""

from __future__ import annotations

from collections.abc import Callable

from PySide6.QtCore import QEvent, Qt, Signal
from PySide6.QtGui import QTextCharFormat, QTextCursor
from PySide6.QtWidgets import (
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


def _common_prefix(words: list[str]) -> str:
    """Longest string every word starts with."""
    shortest = min(words, key=len)
    for index, char in enumerate(shortest):
        if any(word[index] != char for word in words):
            return shortest[:index]
    return shortest


class CommandLine(QLineEdit):
    """Single-line input with history and Tab completion."""

    submitted = Signal(str)
    candidates = Signal(list)

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
        # Tab normally moves focus; here it completes the command name.
        if event.type() == QEvent.Type.KeyPress and event.key() == Qt.Key.Key_Tab:
            self.complete()
            return True
        return super().event(event)

    def keyPressEvent(self, event) -> None:
        if event.key() == Qt.Key.Key_Up and self._history:
            self._cursor = max(0, self._cursor - 1)
            self.setText(self._history[self._cursor])
        elif event.key() == Qt.Key.Key_Down and self._history:
            self._cursor = min(len(self._history), self._cursor + 1)
            self.setText(self._history[self._cursor] if self._cursor < len(self._history) else "")
        else:
            super().keyPressEvent(event)

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

    def echo(self, command: str) -> None:
        self.write(f"{PROMPT} {command}", "command")

    def clear(self) -> None:
        self.output.clear()

    def text(self) -> str:
        return self.output.toPlainText()

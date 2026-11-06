"""``GuiRig``: a real ``MainWindow`` (offscreen) wired to the real command set.

The rig mirrors what the in-process frontend (``QtFrontend``) does, without a training
loop: an entered line is executed by the built-in ``CommandRegistry`` with a real
``CommandContext`` and the outcome is written back to the terminal; window-only commands
reach ``MainWindow.handle_view``. Session events come from a :class:`SessionFeeder`, so a
test decides exactly what the window has seen.

Typing goes through the keyboard (``htspec.keys``); ``rig.enter(line)`` is the shortcut
that submits a line as if it had been typed (``MainWindow.run_command``).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from PySide6.QtCore import QTimer
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from hollywood_training.commands.builtin import create_registry
from hollywood_training.commands.context import CommandContext
from hollywood_training.commands.registry import CommandError, CommandResult
from hollywood_training.commands.view import ViewRequest
from hollywood_training.gui.window import MainWindow
from hollywood_training.sync.control import TrainingControl, TrainingTarget
from hollywood_training.sync.session import SessionState

from . import keys
from .events import SessionFeeder
from .models import Setup, make_setup


class RigFrontend:
    """``FrontendActions`` of the rig: the same behaviour as the in-process frontend."""

    def __init__(self, window: MainWindow) -> None:
        self.window = window
        self.closed = False

    def clear_output(self) -> None:
        self.window.terminal.clear()

    def close(self) -> None:
        self.closed = True
        QTimer.singleShot(0, self.window.close)

    def screenshot(self, path: Path) -> Path:
        try:
            return self.window.save_screenshot(path)
        except OSError as exc:
            raise CommandError(str(exc)) from None

    def view(self, request: ViewRequest) -> str:
        ok, text = self.window.handle_view(request)
        if not ok:
            raise CommandError(text)
        return text


class GuiRig:
    """An offscreen Hollywood Training window plus its command plumbing.

    Args:
        qtbot: pytest-qt's fixture (registers the window for clean-up).
        setup: the native training setup the session describes.
        epochs: total epochs announced by the session.
        max_neurons: drawn neurons per column.
        extra_commands: custom commands (``process.commands`` of the setup by default).
        show: show the window and wait until it is exposed and active.
        title: custom window title (``ht.run(title=...)``).
    """

    def __init__(
        self,
        qtbot,
        *,
        setup: Setup | None = None,
        epochs: int = 10,
        max_neurons: int = 20,
        extra_commands: Any = None,
        show: bool = True,
        title: str | None = None,
    ) -> None:
        self.qtbot = qtbot
        self.setup = setup if setup is not None else make_setup(epochs=epochs)
        self.session = SessionState()
        self.control = TrainingControl(epochs)
        commands = self.setup.process.commands if extra_commands is None else extra_commands
        self.registry = create_registry(commands)
        self.window = MainWindow(self.session, completer=self.registry.complete, title=title)
        qtbot.addWidget(self.window)
        self.frontend = RigFrontend(self.window)
        self.context = CommandContext(
            control=self.control,
            session=self.session,
            registry=self.registry,
            target=TrainingTarget(self.setup.model, self.setup.optimizer),
            frontend=self.frontend,
        )
        self.results: list[tuple[str, CommandResult]] = []
        self.window.command_submitted.connect(self._execute)
        self.feeder = SessionFeeder(
            self.session,
            setup=self.setup,
            epochs=epochs,
            max_neurons=max_neurons,
            sinks=[self.window.apply],
        )
        if show:
            self.show()

    # ----------------------------------------------------------------- window
    def show(self) -> None:
        self.window.show()
        QTest.qWaitForWindowExposed(self.window, 5000)
        self.activate()

    def activate(self) -> bool:
        """Make the window the active window (what a window manager does on Alt+Tab)."""
        self.window.activateWindow()
        self.window.raise_()
        return QTest.qWaitForWindowActive(self.window, 5000)

    def process_events(self, ms: int = 0) -> None:
        QApplication.processEvents()
        if ms:
            QTest.qWait(ms)

    # --------------------------------------------------------------- commands
    def _execute(self, line: str) -> None:
        result = self.registry.execute(line, self.context)
        self.results.append((line, result))
        self.window.show_command_result(result.ok, result.output)

    def enter(self, line: str) -> CommandResult:
        """Submit ``line`` as if typed and entered; returns the command's result."""
        before = len(self.results)
        self.window.run_command(line)
        assert len(self.results) == before + 1, f"{line!r} was not executed"
        return self.results[-1][1]

    def type(self, sequence: str, target: Any = None) -> None:
        """Type ``sequence`` (``htspec.keys`` notation) - into the native window by default,
        i.e. through the platform input path."""
        keys.send(keys.native(self.window) if target is None else target, sequence)
        QApplication.processEvents()

    def type_line(self, line: str, target: Any = None) -> CommandResult | None:
        """Type ``line`` and Return; the result of the command it ran (None if nothing ran)."""
        before = len(self.results)
        self.type(keys.escape(line) + "<Return>", target)
        return self.results[-1][1] if len(self.results) > before else None

    # ---------------------------------------------------------------- terminal
    @property
    def input_text(self) -> str:
        return self.window.terminal.input.text()

    def log(self) -> str:
        return self.window.terminal.text()

    def log_lines(self) -> list[str]:
        return self.window.terminal.text().splitlines()

    def focus_widget(self):
        return QApplication.focusWidget()

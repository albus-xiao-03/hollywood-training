"""``MainWindow``: assembles the panels and routes session events to them.

Layout (resizable with splitters)::

    ┌──────────────────────────── header: title · NETWORK CLASS · state ────────────────┐
    │  neural architecture (neurons, connections, weights) │ training status            │
    │                                                     │ loss chart                 │
    │  command terminal                                   │ system usage               │
    └─────────────────────────────────────────────────────┴────────────────────────────┘

:meth:`MainWindow.render_now` repaints every panel that shows training data
synchronously; it runs while the training loop waits for an epoch's hand-off.

The window does not execute commands itself: an entered line is echoed and emitted as
:attr:`MainWindow.command_submitted`; the frontend runs it wherever the training objects
live and hands the outcome back with :meth:`MainWindow.show_command_result`. Commands that
only change the window come back as a ``ViewRequest`` for :meth:`MainWindow.handle_view`.

While the window is active, every key goes to the terminal's command line
(:class:`~hollywood_training.gui.keyboard_focus.KeyboardFocus`).
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QGuiApplication, QPainter, QPen, QPixmap, QRadialGradient
from PySide6.QtWidgets import QMainWindow, QSplitter, QVBoxLayout, QWidget

from .. import formatting as fmt
from ..commands.view import ViewError, ViewRequest
from ..sync.events import (
    EpochCompleted,
    Event,
    LogMessage,
    SessionFinished,
    SessionInfo,
    SessionStarted,
    StateChanged,
    TrainingPaused,
    TrainingState,
)
from ..sync.session import SessionState
from .animation import AnimationClock
from .keyboard_focus import KeyboardFocus
from .theme import PALETTE, app_icon, color, stylesheet
from .widgets import HeaderBar, LossChart, NetworkView, StatusPanel, TerminalPanel, UsagePanel

STATE_LEVEL = {
    TrainingState.RUNNING: "success",
    TrainingState.PAUSED: "warning",
    TrainingState.FINISHED: "success",
    TrainingState.STOPPED: "warning",
    TrainingState.FAILED: "error",
}


class Backdrop(QWidget):
    """Deep-blue gradient with a faint grid and vignette (cached per size)."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._cache: QPixmap | None = None
        self._key: tuple | None = None

    def paintEvent(self, _event) -> None:
        dpr = round(self.devicePixelRatioF(), 2)
        key = (self.width(), self.height(), dpr)
        if key != self._key:
            self._key = key
            self._cache = self._render(dpr)
        painter = QPainter(self)
        painter.drawPixmap(0, 0, self._cache)
        painter.end()

    def _render(self, dpr: float) -> QPixmap:
        width, height = max(1, self.width()), max(1, self.height())
        pixmap = QPixmap(int(width * dpr), int(height * dpr))
        pixmap.setDevicePixelRatio(dpr)
        painter = QPainter(pixmap)
        painter.fillRect(QRectF(0, 0, width, height), color(PALETTE.void))
        gradient = QRadialGradient(QPointF(width * 0.5, height * 0.38), max(width, height) * 0.8)
        gradient.setColorAt(0.0, color(PALETTE.background_bottom))
        gradient.setColorAt(0.55, color(PALETTE.background_top))
        gradient.setColorAt(1.0, color(PALETTE.void))
        painter.fillRect(QRectF(0, 0, width, height), gradient)
        step = 36
        for k, x in enumerate(range(0, width, step)):
            painter.setPen(QPen(color(PALETTE.grid, 46 if k % 4 == 0 else 22), 1.0))
            painter.drawLine(x, 0, x, height)
        for k, y in enumerate(range(0, height, step)):
            painter.setPen(QPen(color(PALETTE.grid, 46 if k % 4 == 0 else 22), 1.0))
            painter.drawLine(0, y, width, y)
        painter.end()
        return pixmap


class MainWindow(QMainWindow):
    """The Hollywood Training window."""

    closing = Signal()
    command_submitted = Signal(str)

    def __init__(
        self,
        session: SessionState,
        *,
        completer: Callable[[str], list[str]] = lambda prefix: [],
        title: str | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.session = session
        self._title = title
        self._milestone = 1
        self.clock = AnimationClock(60, self)

        self.setWindowIcon(app_icon())
        self.setStyleSheet(stylesheet())
        self._update_title()
        backdrop = Backdrop(self)
        self.setCentralWidget(backdrop)

        self.header = HeaderBar(session, self.clock)
        self.network = NetworkView(session, self.clock)
        self.terminal = TerminalPanel(completer)
        self.status = StatusPanel(session, self.clock)
        self.chart = LossChart(session, self.clock)
        self.usage = UsagePanel(self.clock)

        left = self._splitter(Qt.Orientation.Vertical, [self.network, self.terminal], [70, 30])
        right = self._splitter(
            Qt.Orientation.Vertical, [self.status, self.chart, self.usage], [44, 36, 20]
        )
        main = self._splitter(Qt.Orientation.Horizontal, [left, right], [62, 38])
        layout = QVBoxLayout(backdrop)
        layout.setContentsMargins(14, 8, 14, 14)
        layout.setSpacing(6)
        layout.addWidget(self.header)
        layout.addWidget(main, 1)

        self.terminal.command_entered.connect(self._submit_command)
        self.setMinimumSize(960, 680)
        self._fit_to_screen()
        self.clock.start()
        self.keyboard = KeyboardFocus(self, self.terminal.input)
        self.terminal.input.setFocus()
        self._view_actions: dict[str, Callable[..., str]] = {}

    @staticmethod
    def _splitter(orientation, widgets: list[QWidget], shares: list[int]) -> QSplitter:
        splitter = QSplitter(orientation)
        splitter.setChildrenCollapsible(False)
        splitter.setHandleWidth(8)
        for widget in widgets:
            splitter.addWidget(widget)
        for index, share in enumerate(shares):
            splitter.setStretchFactor(index, share)
        splitter.setSizes([share * 100 for share in shares])
        return splitter

    def _fit_to_screen(self) -> None:
        screen = self.screen() or QGuiApplication.primaryScreen()
        if screen is None:
            self.resize(1500, 920)
            return
        available = screen.availableGeometry()
        width = min(available.width(), max(960, min(1680, int(available.width() * 0.88))))
        height = min(available.height(), max(680, min(1020, int(available.height() * 0.88))))
        self.resize(width, height)
        self.move(available.center().x() - width // 2, available.center().y() - height // 2)

    def _update_title(self) -> None:
        if self._title:
            self.setWindowTitle(self._title)
            return
        name = self.session.network_name
        self.setWindowTitle(f"Hollywood Training — {name}" if name else "Hollywood Training")

    # ------------------------------------------------------------------ events
    def apply(self, event: Event) -> None:
        """Show ``event`` (already folded into the session state)."""
        if isinstance(event, SessionStarted):
            self._on_started(event.info)
            self.render_now()
        elif isinstance(event, EpochCompleted):
            record = event.record
            if (
                record.epoch == 1
                or record.epoch % self._milestone == 0
                or record.epoch == event.total_epochs
            ):
                self.terminal.write(fmt.epoch_summary(record, event.total_epochs), "dim")
            self.render_now()
        elif isinstance(event, TrainingPaused):
            self.render_now()
        elif isinstance(event, StateChanged):
            if event.message:
                self.terminal.write(event.message, STATE_LEVEL.get(event.state, "info"))
        elif isinstance(event, LogMessage):
            self.terminal.write(event.text, event.level.value)
        elif isinstance(event, SessionFinished):
            level = "error" if event.state is TrainingState.FAILED else "success"
            if event.state is TrainingState.STOPPED:
                level = "warning"
            self.terminal.write(
                f"{event.state.value.upper()}: {event.reason} in {fmt.duration(event.elapsed)}",
                level,
            )
            self.render_now()

    def render_now(self) -> None:
        """Synchronously repaint every panel that shows training data."""
        for widget in (self.header, self.network, self.status, self.chart):
            widget.repaint()

    def _on_started(self, info: SessionInfo) -> None:
        self._update_title()
        self._milestone = max(1, info.total_epochs // 20)
        name = info.network_class + (f" ({info.display_name})" if info.display_name else "")
        optimizer = info.optimizer + (
            f" ({info.optimizer_details})" if info.optimizer_details else ""
        )
        self.terminal.write("▓▒░ HOLLYWOOD TRAINING :: COMMAND TERMINAL ░▒▓", "accent")
        self.terminal.write(
            f"network {name} · {fmt.count(info.architecture.total_parameters)} parameters · "
            f"{len(info.architecture.layers)} connection layers",
            "info",
        )
        self.terminal.write(
            f"epochs {info.total_epochs} · batch size {fmt.count(info.batch_size)} · "
            f"loss {info.loss_function} · optimizer {optimizer}",
            "info",
        )
        if info.architecture.unsupported:
            self.terminal.write("not drawn: " + ", ".join(info.architecture.unsupported), "dim")
        self.terminal.write("type 'help' for commands · PgUp / PgDn scroll this log", "dim")

    # ---------------------------------------------------------------- commands
    def _submit_command(self, line: str) -> None:
        self.terminal.echo(line)
        self.command_submitted.emit(line)

    def run_command(self, line: str) -> None:
        """Enter ``line`` as if typed in the terminal."""
        self._submit_command(line)

    def show_command_result(self, ok: bool, output: str) -> None:
        if output:
            self.terminal.write(output, "info" if ok else "error")

    def handle_view(self, request: ViewRequest) -> tuple[bool, str]:
        """Carry out a window-only command request; returns ``(ok, message)``."""
        handler = self._view_actions.get(request.action)
        if handler is None:
            return False, f"unknown window action {request.action!r}"
        try:
            return True, handler(*request.arguments)
        except ViewError as exc:
            return False, str(exc)

    def save_screenshot(self, path: Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if not self.grab().save(str(path), "PNG"):
            raise OSError(f"could not write {path}")
        return path.resolve()

    def closeEvent(self, event) -> None:
        self.closing.emit()
        self.clock.stop()
        self.usage.stop()
        super().closeEvent(event)

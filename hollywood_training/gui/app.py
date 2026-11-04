"""``QtFrontend``: the in-process GUI frontend (``run(process, isolate_gui=False)``).

The default frontend runs the window in its own process
(:mod:`hollywood_training.gui.remote`). This one keeps it inside the training process -
for embedding in an existing Qt application, or where starting a second process is not
possible. Training and the GUI then share the caller's thread, which must be the main
thread (Qt requires it on every platform):

* the manager **pumps** the Qt event loop through :class:`QtPump` - between batches and
  whenever it waits - so the window keeps animating and accepting commands;
* every event is handled the moment it is sent: the frontend folds it into the session
  state, the window repaints the affected panels **synchronously**, and only then is a
  synchronous event acknowledged. The next epoch therefore cannot start before the GUI
  has completed the update for this one - by construction, not by timing.

Drawing between batches makes PyTorch's spinning intra-op thread pool lose its cores, so
:func:`hollywood_training.run` caps the pool at half the CPU cores in this mode
(``torch_threads``). During a single, very long batch the window refreshes only when the
batch ends.

Closing the window - or Ctrl+C in the launching terminal - stops training gracefully.
"""

from __future__ import annotations

import contextlib
import os
import signal
import sys
import threading
import time
import traceback
from collections.abc import Iterator
from pathlib import Path

from PySide6.QtCore import QEventLoop, QTimer
from PySide6.QtWidgets import QApplication

from ..commands.context import CommandContext
from ..commands.registry import CommandError, CommandRegistry
from ..monitor.system import SystemMonitor
from ..sync.channel import Envelope, SyncChannel
from ..sync.control import TrainingControl, TrainingTarget
from ..sync.events import SessionStarted
from ..sync.session import SessionState
from ..training.history import TrainingResult
from ..training.manager import TrainingManager
from .window import MainWindow

CLOSE_DELAY_MS = 400  # with close_on_finish: keep the final state visible briefly


def ensure_display() -> None:
    """Fail clearly (instead of Qt aborting the process) when no display is available."""
    if not sys.platform.startswith(("linux", "freebsd", "openbsd", "netbsd")):
        return
    if (
        os.environ.get("QT_QPA_PLATFORM")
        or os.environ.get("DISPLAY")
        or os.environ.get("WAYLAND_DISPLAY")
    ):
        return
    raise RuntimeError(
        "no display found (DISPLAY and WAYLAND_DISPLAY are unset). Run from a desktop "
        "session, use run(process, gui=False) for console output, or set "
        "QT_QPA_PLATFORM=offscreen to render without a screen."
    )


class QtPump:
    """Runs the Qt event loop on behalf of the training loop (an ``EventPump``)."""

    POLL_INTERVAL = 0.004  # seconds; polling more often only costs time

    def __init__(self) -> None:
        self._last_poll = 0.0
        self._loop: QEventLoop | None = None

    def poll(self) -> None:
        now = time.perf_counter()
        if now - self._last_poll >= self.POLL_INTERVAL:
            self._last_poll = now
            QApplication.processEvents()

    def wait(self, timeout: float) -> None:
        loop = QEventLoop()
        self._loop = loop
        QTimer.singleShot(max(1, int(timeout * 1000)), loop.quit)
        loop.exec()
        self._loop = None
        self._last_poll = time.perf_counter()

    def interrupt(self) -> None:
        """End a running :meth:`wait` early (e.g. right after a command was entered)."""
        if self._loop is not None:
            self._loop.quit()


@contextlib.contextmanager
def _ctrl_c_stops(window: MainWindow, control: TrainingControl) -> Iterator[list[bool]]:
    """Ctrl+C in the launching terminal stops training and closes the window.

    Yields a one-item list that becomes ``[True]`` if Ctrl+C was pressed.
    """
    pressed = [False]

    def handler(_signum, _frame) -> None:
        pressed[0] = True
        control.stop()
        QTimer.singleShot(0, window.close)

    try:
        previous = signal.signal(signal.SIGINT, handler)
    except ValueError:  # not the interpreter's main thread
        yield pressed
        return
    wake = QTimer()  # lets Python run the handler while Qt sits in an event loop
    wake.timeout.connect(lambda: None)
    wake.start(200)
    try:
        yield pressed
    finally:
        wake.stop()
        signal.signal(signal.SIGINT, previous)


class QtFrontend:
    """Native-window frontend; also implements the commands' ``FrontendActions``."""

    def __init__(
        self,
        *,
        channel: SyncChannel,
        control: TrainingControl,
        registry: CommandRegistry,
        close_on_finish: bool = False,
        title: str | None = None,
    ) -> None:
        self._channel = channel
        self._control = control
        self._registry = registry
        self._close_on_finish = close_on_finish
        self._title = title
        self.session = SessionState()
        self.pump = QtPump()
        self.window: MainWindow | None = None
        self._context: CommandContext | None = None
        self._closed = False
        channel.connect(self._handle)

    # ---------------------------------------------------------------- running
    def run(self, manager: TrainingManager) -> TrainingResult:
        """Show the window, run ``manager`` on this thread, return when the window closes."""
        if threading.current_thread() is not threading.main_thread():
            raise RuntimeError("the GUI must run on the main thread; call run() from it")
        ensure_display()
        app = QApplication.instance() or QApplication(
            [sys.argv[0] if sys.argv else "hollywood-training"]
        )
        quit_on_close = app.quitOnLastWindowClosed()
        app.setQuitOnLastWindowClosed(False)  # this window must not quit the caller's app

        window = MainWindow(self.session, completer=self._registry.complete, title=self._title)
        self._context = CommandContext(
            control=self._control,
            session=self.session,
            registry=self._registry,
            target=TrainingTarget(manager.process.model, manager.process.optimizer),
            frontend=self,
        )
        window.closing.connect(self._on_closing)
        window.command_submitted.connect(self._execute)
        self.window = window
        window.show()
        app.processEvents()  # map the window before the first synchronous render

        interrupted = False
        try:
            with _ctrl_c_stops(window, self._control) as pressed:
                try:
                    result = manager.run()
                except KeyboardInterrupt:
                    pressed[0] = True
                    result = manager.partial_result()
                interrupted = pressed[0]
                self._linger(window)
        finally:
            self._control.stop()
            if not self._closed:
                window.close()
            window.deleteLater()
            app.processEvents()
            app.setQuitOnLastWindowClosed(quit_on_close)
        if interrupted:
            raise KeyboardInterrupt
        return result

    def _linger(self, window: MainWindow) -> None:
        """After training: keep the window (results, commands) until it is closed."""
        if self._closed:
            return
        if self._close_on_finish:
            QTimer.singleShot(CLOSE_DELAY_MS, window.close)
        loop = QEventLoop()
        window.closing.connect(loop.quit)
        if not self._closed:
            loop.exec()

    def _execute(self, line: str) -> None:
        if self._context is None or self.window is None:
            return
        result = self._registry.execute(line, self._context)
        self.window.show_command_result(result.ok, result.output)
        self.pump.interrupt()  # let a waiting training loop see the request at once

    def _handle(self, envelope: Envelope) -> None:
        """Apply, draw, then acknowledge - all before the training loop continues."""
        try:
            self.session.apply(envelope.event)
            if self.window is not None:
                if isinstance(envelope.event, SessionStarted):
                    monitor = SystemMonitor(envelope.event.info.device)
                    self.window.usage.set_sampler(monitor.sample)
                self.window.apply(envelope.event)
        except Exception:  # a drawing bug must never stall training
            traceback.print_exc()
        finally:
            if envelope.synchronous:
                self._channel.acknowledge(envelope.sequence)

    def _on_closing(self) -> None:
        self._closed = True
        self._control.stop()
        self.pump.interrupt()

    # ------------------------------------------------- FrontendActions (commands)
    def clear_output(self) -> None:
        if self.window is not None:
            self.window.terminal.clear()

    def close(self) -> None:
        if self.window is not None:
            QTimer.singleShot(0, self.window.close)

    def screenshot(self, path: Path) -> Path:
        if self.window is None:
            raise CommandError("no window to capture")
        try:
            return self.window.save_screenshot(path)
        except OSError as exc:
            raise CommandError(str(exc)) from None

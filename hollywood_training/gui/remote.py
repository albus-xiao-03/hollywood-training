"""``RemoteFrontend``: shows a session in a window that runs in its own process.

This is the default frontend of :func:`hollywood_training.run`. Training stays on the
caller's thread and does no drawing at all; the window lives in a child process
(:mod:`hollywood_training.gui.window_process`) with its own interpreter, so

* training runs as fast as the per-epoch synchronisation allows - nothing else competes
  with it for the interpreter lock or for PyTorch's CPU threads;
* the window animates smoothly even while a long batch is computing.

Synchronisation is unchanged: every synchronous event is sent to the window and the
training loop blocks in :meth:`SyncChannel.deliver` until the window acknowledges it,
which it does only after drawing it. Command lines typed in the window come back over
the link and run here, next to the model (helper thread; model access goes through
``CommandContext.submit``). CPU / memory / GPU usage is sampled here as well, where the
training device is known.
"""

from __future__ import annotations

import os
import secrets
import subprocess
import sys
import threading
import time
from multiprocessing.connection import Listener
from pathlib import Path

from ..commands.context import CommandContext
from ..commands.registry import CommandRegistry
from ..monitor.system import SystemMonitor
from ..sync.channel import Envelope, SyncChannel
from ..sync.control import TrainingControl, TrainingTarget
from ..sync.events import SessionStarted
from ..sync.session import SessionState
from ..training.history import TrainingResult
from ..training.manager import TrainingManager
from .link import AUTHKEY_ENV, Link

STARTUP_TIMEOUT = 60.0  # seconds for the window process to start and connect
SHUTDOWN_TIMEOUT = 10.0
USAGE_INTERVAL = 0.5


def _package_root() -> str:
    """Directory containing the ``hollywood_training`` package (for the child's path)."""
    return str(Path(__file__).resolve().parents[2])


class RemoteFrontend:
    """Training-process side of the isolated GUI; implements ``FrontendActions``."""

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
        self.session = SessionState()  # mirror of what the window shows, for commands
        self._link: Link | None = None
        self._process: subprocess.Popen | None = None
        self._context: CommandContext | None = None
        self._window_closed = threading.Event()
        self._stop_sampling = threading.Event()
        self._sampler: threading.Thread | None = None
        channel.connect(self._forward)

    # ------------------------------------------------------------------ running
    def run(self, manager: TrainingManager) -> TrainingResult:
        """Start the window, run ``manager`` here, return once the window has closed."""
        from .app import ensure_display

        ensure_display()
        self._context = CommandContext(
            control=self._control,
            session=self.session,
            registry=self._registry,
            target=TrainingTarget(manager.process.model, manager.process.optimizer),
            frontend=self,
        )
        self._start_window()
        receiver = threading.Thread(
            target=self._receive, name="hollywood-training-link", daemon=True
        )
        receiver.start()
        interrupted = False
        try:
            try:
                result = manager.run()
            except KeyboardInterrupt:
                interrupted = True
                result = manager.partial_result()
            self._await_window_close(force=interrupted or self._close_on_finish)
        except KeyboardInterrupt:
            interrupted = True
            self._await_window_close(force=True)
        finally:
            self._shutdown(receiver)
        if interrupted:
            raise KeyboardInterrupt
        return result

    def _start_window(self) -> None:
        authkey = secrets.token_bytes(32)
        listener = Listener(authkey=authkey)  # local socket / named pipe, random address
        command = [
            sys.executable,
            "-m",
            "hollywood_training.gui.window_process",
            "--address",
            str(listener.address),
        ]
        if self._title:
            command += ["--title", self._title]
        env = dict(os.environ)
        env[AUTHKEY_ENV] = authkey.hex()
        env["PYTHONPATH"] = os.pathsep.join(filter(None, [_package_root(), env.get("PYTHONPATH")]))
        self._process = subprocess.Popen(command, env=env)

        accepted: dict[str, object] = {}

        def accept() -> None:
            try:
                accepted["connection"] = listener.accept()
            except Exception as exc:  # reported below
                accepted["error"] = exc

        acceptor = threading.Thread(target=accept, daemon=True)
        acceptor.start()
        deadline = time.monotonic() + STARTUP_TIMEOUT
        try:
            while acceptor.is_alive():
                acceptor.join(0.1)
                if self._process.poll() is not None:
                    raise RuntimeError(
                        f"the GUI process exited with code {self._process.returncode} before "
                        "connecting (see its error above); use run(..., gui=False) for "
                        "console output"
                    )
                if time.monotonic() > deadline:
                    self._process.terminate()
                    raise RuntimeError(
                        f"the GUI process did not connect within {STARTUP_TIMEOUT:.0f} s"
                    )
            if "connection" not in accepted:
                raise RuntimeError(f"could not connect to the GUI process: {accepted.get('error')}")
        finally:
            listener.close()
        link = Link(accepted["connection"])  # type: ignore[arg-type]
        try:
            ready = link.poll(STARTUP_TIMEOUT) and link.receive() == ("ready",)
        except EOFError:  # connected, then failed while creating the window
            ready = False
        if not ready:
            link.close()
            try:
                code = self._process.wait(5.0)
            except subprocess.TimeoutExpired:
                self._process.terminate()
                code = None
            raise RuntimeError(
                f"the GUI process failed while opening the window (exit code {code}; see "
                "its error above); use run(..., gui=False) for console output"
            )
        self._link = link
        link.send("completions", self._registry.names(include_aliases=True))

    # ------------------------------------------------- events to the window
    def _forward(self, envelope: Envelope) -> None:
        """Channel sink (training thread): mirror the event, then send it to the window."""
        self.session.apply(envelope.event)
        if isinstance(envelope.event, SessionStarted):
            self._start_sampling(envelope.event.info.device)
        if self._link is None or not self._link.send("event", envelope):
            self._lost_window()

    def _start_sampling(self, device: str) -> None:
        if self._sampler is not None:
            return
        monitor = SystemMonitor(device)

        def sample() -> None:
            while not self._stop_sampling.wait(USAGE_INTERVAL):
                if self._link is None or not self._link.send("usage", monitor.sample()):
                    return

        self._sampler = threading.Thread(
            target=sample, name="hollywood-training-usage", daemon=True
        )
        self._sampler.start()

    # ---------------------------------------------- messages from the window
    def _receive(self) -> None:
        assert self._link is not None
        while True:
            try:
                message = self._link.receive()
            except EOFError:
                break
            kind = message[0]
            if kind == "ack":
                self._channel.acknowledge(message[1])
            elif kind == "command":
                self._execute(message[1])
            elif kind == "closed":
                break
        self._lost_window()

    def _execute(self, line: str) -> None:
        assert self._context is not None and self._link is not None
        result = self._registry.execute(line, self._context)
        self._link.send("result", result.ok, result.output)

    def _lost_window(self) -> None:
        """The window closed (or vanished): stop training, release any waiting hand-off."""
        self._window_closed.set()
        self._control.stop()
        self._channel.close()

    # ---------------------------------------------------------------- shutdown
    def _await_window_close(self, *, force: bool) -> None:
        if self._window_closed.is_set():
            return
        if force and self._link is not None:
            self._link.send("close")
            self._window_closed.wait(SHUTDOWN_TIMEOUT)
            return
        while not self._window_closed.wait(0.2):  # results stay browsable until closed
            pass

    def _shutdown(self, receiver: threading.Thread) -> None:
        self._stop_sampling.set()
        if self._link is not None:
            self._link.close()
        receiver.join(2.0)
        if self._sampler is not None:
            self._sampler.join(2.0)
        if self._process is not None:
            try:
                self._process.wait(SHUTDOWN_TIMEOUT)
            except subprocess.TimeoutExpired:
                self._process.terminate()

    # ------------------------------------------------- FrontendActions (commands)
    def clear_output(self) -> None:
        if self._link is not None:
            self._link.send("frontend", "clear", None)

    def close(self) -> None:
        if self._link is not None:
            self._link.send("frontend", "close", None)

    def screenshot(self, path: Path) -> Path | None:
        if self._link is not None:
            self._link.send("frontend", "screenshot", str(Path(path).expanduser().resolve()))
        return None  # written by the window process, which reports the result

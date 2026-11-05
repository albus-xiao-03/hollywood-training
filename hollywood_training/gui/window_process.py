"""Entry point of the GUI process: ``python -m hollywood_training.gui.window_process``.

Started by :class:`~hollywood_training.gui.remote.RemoteFrontend`; not meant to be run by
hand. It connects back to the training process (address on the command line, key in the
environment), shows the window and then:

* applies every received event to its session state and lets the window draw it;
  a synchronous event is acknowledged only **after** the drawing is complete;
* forwards entered command lines to the training process and shows the replies;
* carries out window-only command requests (``ViewRequest``) and shows their outcome;
* tells the training process when the window is closed.

The process imports no torch: everything it receives is plain data.
"""

from __future__ import annotations

import argparse
import os
import signal
import sys
import threading
import traceback
from multiprocessing.connection import Client
from pathlib import Path

from PySide6.QtCore import QObject, QTimer, Signal
from PySide6.QtWidgets import QApplication

from ..sync.channel import Envelope
from ..sync.session import SessionState
from .link import AUTHKEY_ENV, Link
from .window import MainWindow

CLOSE_DELAY_MS = 400


class _Inbox(QObject):
    """Moves messages from the receiving thread to the GUI thread."""

    received = Signal(object)


class WindowProcess:
    """The GUI side of the link."""

    def __init__(self, link: Link, *, title: str | None = None) -> None:
        self.link = link
        self.session = SessionState()
        self._completions: list[str] = []
        self.window = MainWindow(self.session, completer=self._complete, title=title)
        self.window.command_submitted.connect(lambda line: self.link.send("command", line))
        self.window.closing.connect(self._on_closing)
        self._inbox = _Inbox()
        self._inbox.received.connect(self._dispatch)
        self._disconnected = False

    def _complete(self, prefix: str) -> list[str]:
        prefix = prefix.lower()
        return [name for name in self._completions if name.startswith(prefix)]

    def start(self) -> None:
        self.window.show()
        QApplication.processEvents()
        threading.Thread(target=self._receive, name="hollywood-training-link", daemon=True).start()
        self.link.send("ready")

    def _receive(self) -> None:
        while True:
            try:
                message = self.link.receive()
            except EOFError:
                self._inbox.received.emit(("disconnected",))
                return
            self._inbox.received.emit(message)

    def _dispatch(self, message: tuple) -> None:
        kind = message[0]
        if kind == "event":
            self._apply(message[1])
        elif kind == "usage":
            self.window.usage.set_sample(message[1])
        elif kind == "result":
            self.window.show_command_result(message[1], message[2])
        elif kind == "completions":
            self._completions = sorted(message[1])
        elif kind == "frontend":
            self._frontend_action(message[1], message[2])
        elif kind == "close":
            QTimer.singleShot(CLOSE_DELAY_MS, self.window.close)
        elif kind == "disconnected" and not self._disconnected:
            self._disconnected = True
            self.window.terminal.write("training process disconnected", "warning")

    def _apply(self, envelope: Envelope) -> None:
        try:
            self.session.apply(envelope.event)
            self.window.apply(envelope.event)  # synchronous events are drawn right here
        except Exception:  # a drawing bug must never stall training
            traceback.print_exc()
        finally:
            if envelope.synchronous:
                self.link.send("ack", envelope.sequence)

    def _frontend_action(self, action: str, argument: str | None) -> None:
        if action == "clear":
            self.window.terminal.clear()
        elif action == "close":
            QTimer.singleShot(0, self.window.close)
        elif action == "view" and argument is not None:
            ok, text = self.window.handle_view(argument)
            if text:
                self.window.terminal.write(text, "info" if ok else "error")
        elif action == "screenshot" and argument:
            try:
                saved = self.window.save_screenshot(Path(argument))
            except OSError as exc:
                self.window.terminal.write(f"screenshot failed: {exc}", "error")
            else:
                self.window.terminal.write(f"screenshot saved to {saved}", "success")

    def _on_closing(self) -> None:
        self.link.send("closed")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Hollywood Training window (internal)")
    parser.add_argument("--address", required=True)
    parser.add_argument("--title", default=None)
    args = parser.parse_args(argv)
    authkey = bytes.fromhex(os.environ.pop(AUTHKEY_ENV, ""))
    if not authkey:
        print("hollywood_training.gui.window_process: no authentication key", file=sys.stderr)
        return 2
    signal.signal(signal.SIGINT, signal.SIG_IGN)  # the training process decides when to stop

    link = Link(Client(args.address, authkey=authkey))
    app = QApplication(["hollywood-training"])
    window = WindowProcess(link, title=args.title)
    window.start()
    app.exec()
    link.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())

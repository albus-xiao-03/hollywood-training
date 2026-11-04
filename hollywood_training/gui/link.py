"""The link between the training process and the GUI process (torch-free).

A thin wrapper around :mod:`multiprocessing.connection` (an authenticated local socket on
Linux/macOS, a named pipe on Windows) that exchanges small picklable tuples:

=====================================  ==================================================
training process → GUI process         meaning
=====================================  ==================================================
``("event", Envelope)``                a session event; synchronous ones must be acked
``("usage", UsageSample)``             a CPU / memory / GPU reading
``("result", ok, text)``               outcome of a command line
``("completions", [names])``           command names for Tab completion
``("frontend", action, argument)``     ``clear`` / ``close`` / ``screenshot`` (path)
``("close",)``                         close the window (run is over)
=====================================  ==================================================

=====================================  ==================================================
GUI process → training process         meaning
=====================================  ==================================================
``("ready",)``                         the window is up
``("ack", sequence)``                  a synchronous event was applied *and drawn*
``("command", line)``                  the user entered a command line
``("closed",)``                        the window was closed
=====================================  ==================================================
"""

from __future__ import annotations

import contextlib
import threading
from multiprocessing.connection import Connection

AUTHKEY_ENV = "HOLLYWOOD_TRAINING_AUTHKEY"


class Link:
    """A connection with serialised sends (several threads may send; one receives)."""

    def __init__(self, connection: Connection) -> None:
        self._connection = connection
        self._send_lock = threading.Lock()
        self._closed = False

    @property
    def closed(self) -> bool:
        return self._closed

    def send(self, *message: object) -> bool:
        """Send one message; False if the other side is gone."""
        if self._closed:
            return False
        try:
            with self._send_lock:
                self._connection.send(message)
        except (OSError, EOFError, ValueError):
            self._closed = True
            return False
        return True

    def receive(self) -> tuple:
        """Next message (blocking). Raises ``EOFError`` when the other side is gone."""
        try:
            return self._connection.recv()
        except (OSError, ValueError) as exc:
            raise EOFError(str(exc)) from exc

    def poll(self, timeout: float) -> bool:
        try:
            return self._connection.poll(timeout)
        except (OSError, ValueError, EOFError):
            return False

    def close(self) -> None:
        self._closed = True
        with contextlib.suppress(OSError):
            self._connection.close()

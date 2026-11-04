"""Event pumps: how the training loop gives its frontend time to run.

A frontend that runs on the training thread itself - the in-process window, where
training and the GUI share the caller's main thread - needs the training loop to give it
time. The manager therefore *pumps* its frontend:

* :meth:`EventPump.poll` at every safe point between batches - cheap, non-blocking;
* :meth:`EventPump.wait` whenever it has nothing to do for a moment (paused, finished
  but lingering, the optional per-epoch delay) - blocks up to ``timeout`` seconds while
  the frontend keeps processing input and animating.

Frontends with no event loop of their own on the training thread - the console, and the
default window process (whose window lives in another process) - use the default
:class:`WaitingPump`, which simply sleeps until a control request arrives.
"""

from __future__ import annotations

from typing import Protocol

from .control import TrainingControl


class EventPump(Protocol):
    def poll(self) -> None:
        """Let the frontend handle pending events; return quickly."""

    def wait(self, timeout: float) -> None:
        """Let the frontend run for up to ``timeout`` seconds (may return earlier)."""


class WaitingPump:
    """Pump for frontends that need no event loop: waits on the control's condition."""

    def __init__(self, control: TrainingControl) -> None:
        self._control = control

    def poll(self) -> None:
        pass

    def wait(self, timeout: float) -> None:
        self._control.wait(timeout)

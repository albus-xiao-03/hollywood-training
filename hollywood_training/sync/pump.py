"""Event pumps: how the training loop gives its frontend time to run.

Training and the GUI share one thread (the caller's - which Qt requires to be the main
thread). The manager therefore *pumps* the frontend:

* :meth:`EventPump.poll` at every safe point between batches - cheap, non-blocking;
* :meth:`EventPump.wait` whenever it has nothing to do for a moment (paused, finished
  but lingering, the optional per-epoch delay) - blocks up to ``timeout`` seconds while
  the frontend keeps processing input and animating.

Sharing the thread keeps PyTorch on the thread that created the data and the model, so
its intra-op thread pool runs at full speed, and the GUI never competes with training for
the interpreter lock. The default :class:`WaitingPump` (console / headless runs) simply
sleeps until a control request arrives.
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

"""``TrainingControl``: runtime requests flowing from a frontend to the training thread.

Commands (typed in the GUI terminal, for example) only ever *request* changes through this
object; the training thread applies them at safe points between batches. Anything that
must touch the model or optimizer is queued as an *action* - a callable that the training
thread runs with a :class:`TrainingTarget` - so it never races with an optimisation step.
All methods are thread-safe.
"""

from __future__ import annotations

import math
import threading
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class TrainingTarget:
    """What queued actions may touch. Handed to them on the training thread."""

    model: Any  # torch.nn.Module
    optimizer: Any  # torch.optim.Optimizer


ControlAction = Callable[[TrainingTarget], "str | None"]


class TrainingControl:
    """Pause / resume / step / stop / settings / queued actions for one session."""

    def __init__(self, total_epochs: int, *, start_paused: bool = False) -> None:
        if int(total_epochs) < 1:
            raise ValueError("total_epochs must be at least 1")
        self._condition = threading.Condition()
        self._total_epochs = int(total_epochs)
        self._paused = bool(start_paused)
        self._stop = False
        self._closed = False
        self._delay = 0.0
        self._completed = 0
        self._pause_after: int | None = None
        self._actions: deque[tuple[ControlAction, str]] = deque()
        self._revision = 0

    def _changed(self) -> None:
        # Caller holds the condition. Count the change and wake any waiting thread.
        self._revision += 1
        self._condition.notify_all()

    # ----------------------------------------------------------------- requests
    def pause(self) -> None:
        """Pause at the next batch boundary."""
        with self._condition:
            if not self._stop:
                self._paused = True
                self._pause_after = None
                self._changed()

    def resume(self) -> None:
        with self._condition:
            self._paused = False
            self._pause_after = None
            self._changed()

    def step(self, epochs: int = 1) -> None:
        """Run ``epochs`` more epochs, then pause again."""
        if int(epochs) < 1:
            raise ValueError("epochs must be at least 1")
        with self._condition:
            self._pause_after = self._completed + int(epochs)
            self._paused = False
            self._changed()

    def stop(self) -> None:
        """End the session at the next batch boundary."""
        with self._condition:
            self._stop = True
            self._paused = False
            self._changed()

    def set_total_epochs(self, total: int) -> None:
        with self._condition:
            if int(total) < max(1, self._completed):
                raise ValueError(f"total epochs must be at least {max(1, self._completed)}")
            self._total_epochs = int(total)
            self._changed()

    def set_delay(self, seconds: float) -> None:
        """Extra pause after every epoch (0 = run as fast as synchronisation allows)."""
        if not math.isfinite(float(seconds)) or float(seconds) < 0:
            raise ValueError("delay must be a finite, non-negative number of seconds")
        with self._condition:
            self._delay = float(seconds)
            self._changed()

    def submit(self, action: ControlAction, description: str = "") -> bool:
        """Queue ``action`` for the training thread. False once the session has ended."""
        with self._condition:
            if self._closed:
                return False
            self._actions.append((action, description))
            self._changed()
            return True

    # -------------------------------------------------------------------- state
    @property
    def paused(self) -> bool:
        return self._paused

    @property
    def stop_requested(self) -> bool:
        return self._stop

    @property
    def closed(self) -> bool:
        """True once the training thread no longer runs queued actions."""
        return self._closed

    @property
    def total_epochs(self) -> int:
        return self._total_epochs

    @property
    def delay(self) -> float:
        return self._delay

    @property
    def completed_epochs(self) -> int:
        return self._completed

    @property
    def revision(self) -> int:
        """Increases with every request; lets the training thread notice changes cheaply."""
        return self._revision

    @property
    def stepping_until(self) -> int | None:
        """Epoch after which a ``step`` request pauses, if one is active."""
        return self._pause_after

    # ----------------------------------------------------- training-thread side
    def take_actions(self) -> list[tuple[ControlAction, str]]:
        """Remove and return every queued action (training thread)."""
        with self._condition:
            actions = list(self._actions)
            self._actions.clear()
            return actions

    def epoch_completed(self, epoch: int) -> None:
        """Record progress; turns a finished ``step`` request into a pause."""
        with self._condition:
            self._completed = int(epoch)
            if self._pause_after is not None and self._completed >= self._pause_after:
                self._paused = True
                self._pause_after = None

    def wait(self, timeout: float) -> None:
        """Sleep until any request arrives or ``timeout`` seconds pass."""
        with self._condition:
            self._condition.wait(timeout)

    def close(self) -> list[tuple[ControlAction, str]]:
        """Called when the training thread exits; returns actions that never ran."""
        with self._condition:
            self._closed = True
            leftover = list(self._actions)
            self._actions.clear()
            self._changed()
            return leftover

"""``SessionState``: the frontend's picture of the session, built from the event stream.

A frontend applies every event it receives (on its own thread) and then reads
``SessionState`` to draw. Version counters tell views when cached drawings must be rebuilt.
Commands read it too (``status``, ``history``, ...).

Not locked: one thread applies events. The window-process frontend also keeps a mirror
in the training process, written by the training thread and read (plus two plain
attribute writes by the ``epochs`` / ``delay`` commands) by its command thread - safe
under the interpreter lock, at worst a moment stale.
"""

from __future__ import annotations

import math
import time

from ..tracking.architecture import ArchitectureInfo
from ..tracking.weights import WeightSnapshot
from .events import (
    BatchProgress,
    EpochCompleted,
    EpochRecord,
    Event,
    LogMessage,
    Phase,
    SessionFinished,
    SessionInfo,
    SessionStarted,
    StateChanged,
    TrainingPaused,
    TrainingState,
    best_epoch,
)


class SessionState:
    """Current view of one training session (frontend thread only)."""

    def __init__(self) -> None:
        # Version counters only ever increase, so cached drawings are never mistaken for
        # current ones - not even across a reset for a new session.
        self.architecture_version = 0
        self.weights_version = 0
        self.history_version = 0
        self._reset()

    def _reset(self) -> None:
        self.info: SessionInfo | None = None
        self.architecture: ArchitectureInfo | None = None
        self.weights: WeightSnapshot | None = None
        self.history: list[EpochRecord] = []
        self.state = TrainingState.PREPARING
        self.epoch = 0  # completed epochs
        self.total_epochs = 0
        self.learning_rate = math.nan
        self.batch_size: int | None = None
        self.delay = 0.0
        self.phase: Phase | None = None
        self.batch = 0
        self.total_batches: int | None = None
        self.batch_loss: float | None = None
        self.finished: SessionFinished | None = None
        self._elapsed = 0.0
        self._anchor = time.monotonic()

    # ------------------------------------------------------------------ updates
    def apply(self, event: Event) -> None:
        """Fold one event into the state."""
        if isinstance(event, SessionStarted):
            self._start(event)
        elif isinstance(event, BatchProgress):
            self.phase = event.phase
            self.batch = event.batch
            self.total_batches = event.total_batches
            self.batch_loss = event.loss
            if event.batch_size is not None and self.batch_size is None:
                self.batch_size = event.batch_size
            self._set_elapsed(event.elapsed)
        elif isinstance(event, EpochCompleted):
            self.history.append(event.record)
            self.history_version += 1
            self.epoch = event.record.epoch
            self.total_epochs = event.total_epochs
            self.learning_rate = event.learning_rate
            if event.batch_size is not None:
                self.batch_size = event.batch_size
            if event.architecture is not None:
                self.architecture = event.architecture
                self.architecture_version += 1
            self._set_weights(event.weights)
            self.batch = 0
            self.phase = None
            self._set_elapsed(event.elapsed)
        elif isinstance(event, TrainingPaused):
            self._set_weights(event.weights)
            self.state = TrainingState.PAUSED
            self._set_elapsed(event.elapsed)
        elif isinstance(event, StateChanged):
            self.state = event.state
            self.epoch = event.epoch
            self.total_epochs = event.total_epochs
            self.learning_rate = event.learning_rate
            self.delay = event.delay
            self._set_elapsed(event.elapsed)
        elif isinstance(event, SessionFinished):
            self.finished = event
            self.state = event.state
            if event.weights is not None:
                self._set_weights(event.weights)
            self._set_elapsed(event.elapsed)
        elif isinstance(event, LogMessage):
            pass  # text is shown by the frontend, not stored

    def _start(self, event: SessionStarted) -> None:
        self._reset()
        self.history_version += 1
        self.info = event.info
        self.architecture = event.info.architecture
        self.architecture_version += 1
        self.total_epochs = event.info.total_epochs
        self.learning_rate = event.info.learning_rate
        self.batch_size = event.info.batch_size
        self.total_batches = event.info.train_batches
        self.state = event.state
        self._set_weights(event.weights)
        self._set_elapsed(0.0)

    def _set_weights(self, weights: WeightSnapshot) -> None:
        self.weights = weights
        self.weights_version += 1

    def _set_elapsed(self, elapsed: float) -> None:
        self._elapsed = elapsed
        self._anchor = time.monotonic()

    # ----------------------------------------------------------------- queries
    @property
    def network_name(self) -> str:
        return self.info.network_class if self.info else ""

    def elapsed_now(self) -> float:
        """Active training time, advancing smoothly between events while running."""
        if self.state is TrainingState.RUNNING:
            return self._elapsed + (time.monotonic() - self._anchor)
        return self._elapsed

    @property
    def last_record(self) -> EpochRecord | None:
        return self.history[-1] if self.history else None

    def best_record(self) -> EpochRecord | None:
        """Epoch with the lowest finite validation loss (training loss when there is none)."""
        return best_epoch(self.history)

    def epoch_progress(self) -> float:
        """Completed fraction of all requested epochs (0-1)."""
        if self.total_epochs <= 0:
            return 0.0
        return min(1.0, self.epoch / self.total_epochs)

    def batch_progress(self) -> float | None:
        """Fraction of the current epoch phase, None when the batch count is unknown."""
        if not self.total_batches:
            return None
        return min(1.0, self.batch / self.total_batches)

    def eta(self) -> float | None:
        """Estimated seconds until all requested epochs are done."""
        if self.epoch <= 0 or self.state is not TrainingState.RUNNING:
            return None
        remaining = max(0, self.total_epochs - self.epoch)
        return self.elapsed_now() / self.epoch * remaining

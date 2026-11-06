"""``SessionFeeder``: the events a ``TrainingManager`` sends, built from a real model.

The window and the session state only ever see events (``hollywood_training.sync.events``).
The feeder produces them the way the manager does - session info from
``build_session_info``, weight snapshots from a real ``NetworkTracker`` - while the test
decides what happens: how many epochs, which losses (including NaN, inf, 0 or negative
values), pauses, log lines and the end of the session. Each event is folded into the
``SessionState`` first and then handed to every sink (e.g. ``MainWindow.apply``), which is
exactly what the frontends do.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable
from typing import Any

import torch

from hollywood_training.sync.events import (
    BatchProgress,
    EpochCompleted,
    EpochRecord,
    LogLevel,
    LogMessage,
    Phase,
    SessionFinished,
    SessionStarted,
    StateChanged,
    TrainingPaused,
    TrainingState,
)
from hollywood_training.sync.session import SessionState
from hollywood_training.tracking.tracker import NetworkTracker
from hollywood_training.training.describe import build_session_info, current_learning_rate

from .models import Setup, make_setup

Sink = Callable[[Any], None]
UNSET: Any = object()


class SessionFeeder:
    """Feeds one session's events to ``session`` and ``sinks``.

    Args:
        session: the session state to fold the events into.
        setup: the training setup (default: a fresh :func:`~htspec.models.make_setup`).
        epochs: total epochs announced by the session.
        max_neurons: neuron limit of the tracker (drawn neurons per column).
        sinks: callables receiving every event after it was applied to ``session``.
    """

    def __init__(
        self,
        session: SessionState | None = None,
        *,
        setup: Setup | None = None,
        epochs: int = 10,
        max_neurons: int = 20,
        sinks: Iterable[Sink] = (),
    ) -> None:
        self.session = session if session is not None else SessionState()
        self.setup = setup if setup is not None else make_setup(epochs=epochs)
        self.tracker = NetworkTracker(self.setup.model, max_neurons=max_neurons)
        self.total_epochs = epochs
        self.epoch = 0
        self.elapsed = 0.0
        self.records: list[EpochRecord] = []
        self.sinks: list[Sink] = list(sinks)
        self.sent: list[Any] = []

    # ---------------------------------------------------------------- plumbing
    def apply(self, event: Any) -> Any:
        self.session.apply(event)
        for sink in self.sinks:
            sink(event)
        self.sent.append(event)
        return event

    def info(self):
        return build_session_info(
            self.setup.process,
            self.tracker.architecture.describe(),
            total_epochs=self.total_epochs,
            neuron_limit=self.tracker.max_neurons,
        )

    # ------------------------------------------------------------------ events
    def start(self, state: TrainingState = TrainingState.RUNNING) -> SessionStarted:
        self.epoch = 0
        self.records = []
        return self.apply(SessionStarted(self.info(), self.tracker.snapshot(0), state))

    def train_epoch(
        self,
        *,
        train_loss: float | None = None,
        val_loss: Any = UNSET,
        metrics: dict[str, float] | None = None,
        val_metrics: dict[str, float] | None = None,
        duration: float = 0.012,
        learn: bool = True,
        total_epochs: int | None = None,
    ) -> EpochCompleted:
        """One epoch: really trains the model on the setup's data when ``learn`` (so the
        weights change), then sends ``EpochCompleted`` with the real snapshot.

        Losses default to the real mean losses; pass values to override them
        (``val_loss=None`` sends an epoch without a validation loss).
        """
        setup = self.setup
        learning_rate = current_learning_rate(setup.optimizer)
        real_train, real_val = self._run_real_epoch(learn)
        if val_loss is UNSET:
            val_loss = real_val
        record = EpochRecord(
            epoch=self.epoch + 1,
            train_loss=real_train if train_loss is None else float(train_loss),
            val_loss=None if val_loss is None else float(val_loss),
            train_metrics=dict(metrics or {}),
            val_metrics=dict(val_metrics or {}),
            learning_rate=learning_rate,
            duration=duration,
            train_batches=len(setup.train_loader),
            val_batches=len(setup.val_loader) if setup.val_loader is not None else 0,
        )
        if total_epochs is not None:
            self.total_epochs = total_epochs
        self.epoch = record.epoch
        self.elapsed += duration
        self.records.append(record)
        event = EpochCompleted(
            record=record,
            weights=self.tracker.snapshot(record.epoch),
            total_epochs=self.total_epochs,
            learning_rate=current_learning_rate(setup.optimizer),
            batch_size=setup.train_loader.batch_size,
            elapsed=self.elapsed,
        )
        return self.apply(event)

    def epochs(self, count: int, **kwargs: Any) -> list[EpochCompleted]:
        return [self.train_epoch(**kwargs) for _ in range(count)]

    def progress(
        self, batch: int, total: int | None = None, *, phase: Phase = Phase.TRAIN, loss=0.5
    ) -> BatchProgress:
        total = len(self.setup.train_loader) if total is None else total
        batch_size = self.setup.train_loader.batch_size
        event = BatchProgress(self.epoch + 1, phase, batch, total, loss, batch_size, self.elapsed)
        return self.apply(event)

    def paused(self, *, phase: Phase | None = None, batch: int = 0) -> TrainingPaused:
        weights = self.tracker.snapshot(self.epoch, baseline=False)
        in_progress = self.epoch + 1 if phase is not None else self.epoch
        return self.apply(TrainingPaused(in_progress, phase, batch, weights, self.elapsed))

    def state(self, state: TrainingState, message: str = "", *, delay: float = 0.0):
        event = StateChanged(
            state=state,
            epoch=self.epoch,
            total_epochs=self.total_epochs,
            learning_rate=current_learning_rate(self.setup.optimizer),
            delay=delay,
            elapsed=self.elapsed,
            message=message,
        )
        return self.apply(event)

    def log(self, text: str, level: LogLevel = LogLevel.INFO) -> LogMessage:
        return self.apply(LogMessage(text, level))

    def finish(
        self,
        state: TrainingState = TrainingState.FINISHED,
        *,
        reason: str | None = None,
        error: str | None = None,
    ) -> SessionFinished:
        reason = reason or f"completed {self.epoch} epochs"
        weights = self.tracker.snapshot(self.epoch, baseline=False)
        return self.apply(SessionFinished(state, self.epoch, self.elapsed, weights, reason, error))

    # ------------------------------------------------------------------ helpers
    def _run_real_epoch(self, learn: bool) -> tuple[float, float | None]:
        setup = self.setup
        model, optimizer, loss_fn = setup.model, setup.optimizer, setup.loss_fn
        losses = []
        model.train()
        for inputs, targets in setup.train_loader:
            with torch.set_grad_enabled(learn):
                loss = loss_fn(model(inputs), targets)
            if learn:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            losses.append(float(loss.detach()))
        val = None
        if setup.val_loader is not None:
            model.eval()
            with torch.no_grad():
                val_losses = [float(loss_fn(model(x), y)) for x, y in setup.val_loader]
            val = sum(val_losses) / len(val_losses)
        return (sum(losses) / len(losses) if losses else math.nan), val

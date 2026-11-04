"""``ConsoleFrontend``: the plain-text frontend used by ``run(process, gui=False)``.

For machines without a display (servers, CI, containers). It handles every event right
away on the training thread, so synchronous events are acknowledged immediately and the
training runs at full speed.
"""

from __future__ import annotations

import sys
from typing import TextIO

from . import formatting as fmt
from .sync.channel import Envelope, SyncChannel
from .sync.events import (
    EpochCompleted,
    Event,
    LogLevel,
    LogMessage,
    SessionFinished,
    SessionStarted,
    StateChanged,
)
from .sync.session import SessionState

_LEVEL_PREFIX = {
    LogLevel.INFO: "",
    LogLevel.SUCCESS: "ok: ",
    LogLevel.WARNING: "warning: ",
    LogLevel.ERROR: "error: ",
}


class ConsoleFrontend:
    """Prints a session to a text stream.

    Args:
        channel: the channel to listen on (this frontend connects itself).
        stream: where to write (default ``sys.stdout``).
        log_every: print every n-th epoch; by default about 25 lines per run.
    """

    def __init__(
        self, channel: SyncChannel, *, stream: TextIO | None = None, log_every: int | None = None
    ) -> None:
        if log_every is not None and log_every < 1:
            raise ValueError("log_every must be at least 1")
        self._channel = channel
        self._stream = stream
        self._log_every = log_every
        self.session = SessionState()
        channel.connect(self.handle)

    def handle(self, envelope: Envelope) -> None:
        try:
            self.session.apply(envelope.event)
            self._render(envelope.event)
        finally:
            if envelope.synchronous:
                self._channel.acknowledge(envelope.sequence)

    # ---------------------------------------------------------------- output
    def _write(self, text: str) -> None:
        print(text, file=self._stream or sys.stdout, flush=True)

    def _render(self, event: Event) -> None:
        if isinstance(event, SessionStarted):
            self._header(event)
        elif isinstance(event, EpochCompleted):
            if self._should_print(event):
                self._write(self._epoch_line(event))
        elif isinstance(event, StateChanged):
            if event.message:
                self._write(f"[{event.state.value}] {event.message}")
        elif isinstance(event, LogMessage):
            self._write(f"{_LEVEL_PREFIX[event.level]}{event.text}")
        elif isinstance(event, SessionFinished):
            self._footer(event)

    def _header(self, event: SessionStarted) -> None:
        info = event.info
        name = info.network_class + (f" - {info.display_name}" if info.display_name else "")
        optimizer = info.optimizer
        if info.optimizer_details:
            optimizer += f" ({info.optimizer_details})"
        self._write("=" * 72)
        self._write(f"HOLLYWOOD TRAINING :: {name}")
        self._write(
            f"epochs {info.total_epochs} | batch size {fmt.count(info.batch_size)} | "
            f"lr {fmt.learning_rate(info.learning_rate)} | device {info.device}"
        )
        self._write(f"loss {info.loss_function} | optimizer {optimizer}")
        self._write(f"normalisation {info.normalization}")
        self._write("=" * 72)

    def _should_print(self, event: EpochCompleted) -> bool:
        epoch, total = event.record.epoch, event.total_epochs
        every = self._log_every or max(1, total // 25)
        return epoch == 1 or epoch % every == 0 or epoch == total

    def _epoch_line(self, event: EpochCompleted) -> str:
        record = event.record
        width = len(str(event.total_epochs))
        parts = [
            f"epoch {record.epoch:>{width}}/{event.total_epochs}",
            f"train {fmt.loss(record.train_loss)}",
        ]
        if record.val_loss is not None:
            parts.append(f"val {fmt.loss(record.val_loss)}")
        parts += [
            f"{name} {fmt.metric(name, value)}" for name, value in record.train_metrics.items()
        ]
        parts += [
            f"val {name} {fmt.metric(name, value)}" for name, value in record.val_metrics.items()
        ]
        parts += [f"lr {fmt.learning_rate(record.learning_rate)}", f"{record.duration:.3f}s"]
        return " | ".join(parts)

    def _footer(self, event: SessionFinished) -> None:
        self._write("-" * 72)
        self._write(f"{event.state.value.upper()}: {event.reason} in {fmt.duration(event.elapsed)}")
        best = self.session.best_record()
        if best is not None:
            value = best.val_loss if best.val_loss is not None else best.train_loss
            self._write(f"best loss {fmt.loss(value)} at epoch {best.epoch}")
        if event.error:
            self._write(f"error: {event.error}")

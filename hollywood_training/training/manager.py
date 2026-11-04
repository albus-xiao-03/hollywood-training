"""``TrainingManager``: runs a :class:`TrainingProcess` epoch by epoch, in lock-step with
a frontend.

For every epoch:

1. **train** - ``model.train()``, then the user's ``train_step`` for every batch;
2. **validate** (if configured) - ``model.eval()`` under ``torch.no_grad()``;
3. step the scheduler and run the ``on_epoch_end`` hooks;
4. snapshot the weights and **deliver** :class:`EpochCompleted` through the
   :class:`SyncChannel` - blocking until the frontend has applied *and drawn* it -
   then continue immediately (no artificial delay unless the user asks for one).

Between batches the manager pumps the frontend (:class:`~hollywood_training.sync.pump.
EventPump`) and applies runtime requests from :class:`TrainingControl` (pause, stop,
queued actions such as learning-rate changes). It never touches a GUI itself.
"""

from __future__ import annotations

import math
import time
import traceback
from dataclasses import dataclass, field

import torch

from ..sync.channel import SyncChannel
from ..sync.control import TrainingControl, TrainingTarget
from ..sync.events import (
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
from ..sync.pump import EventPump, WaitingPump
from ..tracking.tracker import NetworkTracker
from ..tracking.weights import WeightSnapshot
from .describe import (
    build_session_info,
    count_batches,
    current_learning_rate,
    resolve_batch_size,
)
from .history import TrainingResult
from .process import ProcessSteps, TrainingProcess
from .steps import infer_batch_size, read_step_output
from .timing import ActiveClock


class TrainingError(RuntimeError):
    """A problem detected while running the process (e.g. a data source with no batches)."""


@dataclass(slots=True)
class _PhaseResult:
    loss: float
    metrics: dict[str, float] = field(default_factory=dict)
    batches: int = 0


class TrainingManager:
    """Run ``process``, reporting to the frontend connected to ``channel``.

    Args:
        process: the wrapped training process.
        channel: hand-off to the frontend (must have a sink connected).
        control: runtime controls; created from ``process.epochs`` when omitted.
        tracker: network tracker; created for ``process.model`` when omitted.
        linger: after the last epoch, wait (state ``FINISHED``) so more epochs can be
            requested with ``epochs <n>``, until stopped. Used by interactive frontends.
        progress_interval: minimum seconds between :class:`BatchProgress` events.
        pump: gives the frontend time between batches and while waiting; defaults to
            :class:`~hollywood_training.sync.pump.WaitingPump`.
    """

    def __init__(
        self,
        process: TrainingProcess,
        *,
        channel: SyncChannel,
        control: TrainingControl | None = None,
        tracker: NetworkTracker | None = None,
        linger: bool = False,
        progress_interval: float = 1 / 30,
        pump: EventPump | None = None,
    ) -> None:
        self._process = process
        self._channel = channel
        self._control = control if control is not None else TrainingControl(process.epochs)
        self._tracker = tracker if tracker is not None else NetworkTracker(process.model)
        self._pump: EventPump = pump if pump is not None else WaitingPump(self._control)
        self._linger = linger
        self._progress_interval = progress_interval
        self._target = TrainingTarget(process.model, process.optimizer)
        self._steps: ProcessSteps | None = None
        self._clock = ActiveClock()
        self._history: list[EpochRecord] = []
        self._state = TrainingState.PREPARING
        self._epoch = 0
        self._batch_size = resolve_batch_size(process.batch_size, process.train_data)
        self._trace_pending = True
        self._architecture_changed = False
        self._last_progress = -math.inf
        self._seen_revision = -1

    # ---------------------------------------------------------------- queries
    @property
    def process(self) -> TrainingProcess:
        return self._process

    @property
    def control(self) -> TrainingControl:
        return self._control

    @property
    def tracker(self) -> NetworkTracker:
        return self._tracker

    @property
    def state(self) -> TrainingState:
        return self._state

    @property
    def history(self) -> tuple[EpochRecord, ...]:
        return tuple(self._history)

    def partial_result(self) -> TrainingResult:
        """Result built from what has completed so far (for frontends that give up waiting)."""
        return TrainingResult(TrainingState.STOPPED, self.history, self._clock.elapsed())

    # -------------------------------------------------------------------- run
    def run(self) -> TrainingResult:
        """Run the session to its end (finished, stopped or failed) and return the result."""
        error: BaseException | None = None
        interrupted: KeyboardInterrupt | None = None
        final_state = TrainingState.STOPPED
        try:
            self._steps = self._process.resolve_steps()
            self._start()
            final_state = TrainingState.FINISHED if self._loop() else TrainingState.STOPPED
        except KeyboardInterrupt as exc:  # console runs: Ctrl+C ends the session cleanly
            interrupted = exc
            self._log(LogLevel.WARNING, "interrupted")
        except Exception as exc:
            error = exc
            final_state = TrainingState.FAILED
            details = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
            self._log(LogLevel.ERROR, details.rstrip())
        finally:
            self._clock.pause()
            for _action, description in self._control.close():
                if description:
                    self._log(LogLevel.WARNING, f"not applied (session ended): {description}")

        self._state = final_state
        weights = self._safe_snapshot()
        self._channel.deliver(
            SessionFinished(
                state=final_state,
                epochs_completed=len(self._history),
                elapsed=self._clock.elapsed(),
                weights=weights,
                reason=self._finish_reason(final_state),
                error=None if error is None else f"{type(error).__name__}: {error}",
            )
        )
        result = TrainingResult(
            final_state, self.history, self._clock.elapsed(), error=error, final_weights=weights
        )
        if interrupted is not None:
            raise interrupted
        return result

    # ------------------------------------------------------------ lifecycle
    def _start(self) -> None:
        architecture = self._tracker.architecture.describe()
        info = build_session_info(
            self._process,
            architecture,
            total_epochs=self._control.total_epochs,
            neuron_limit=self._tracker.max_neurons,
        )
        weights = self._tracker.snapshot(0)
        self._state = TrainingState.PAUSED if self._control.paused else TrainingState.RUNNING
        if not self._channel.deliver(SessionStarted(info, weights, self._state)):
            self._control.stop()
            return
        self._seen_revision = self._control.revision
        if self._state is TrainingState.RUNNING:
            self._clock.start()
        epochs = info.total_epochs
        self._log(
            LogLevel.INFO,
            f"training {info.network_class} for {epochs} epoch{'s' if epochs != 1 else ''} "
            f"on {info.device}",
        )

    def _loop(self) -> bool:
        """Epoch loop. Returns True when every requested epoch completed."""
        while True:
            # completion first: a `step` that ends on the last epoch must not pause
            # "before" an epoch that will never run
            if self._epoch >= self._control.total_epochs:
                if not (self._linger and self._await_more_epochs()):
                    return True
                continue
            if not self._checkpoint(None, 0, None):
                return False
            record = self._run_epoch(self._epoch + 1)
            if record is None:
                return False
            self._finish_epoch(record)

    def _await_more_epochs(self) -> bool:
        """Linger in FINISHED until more epochs are requested (True) or a stop (False)."""
        self._clock.pause()
        if self._control.paused:  # a pause (e.g. from `step`) has nothing left to hold back
            self._control.resume()
        self._set_state(
            TrainingState.FINISHED,
            f"all {self._control.total_epochs} epochs complete - 'epochs <n>' trains further",
        )
        while not self._control.stop_requested:
            self._apply_requests()
            if self._control.total_epochs > self._epoch:
                self._clock.start()
                self._set_state(
                    TrainingState.RUNNING, f"continuing to epoch {self._control.total_epochs}"
                )
                return True
            self._pump.wait(0.1)
        return False

    # ----------------------------------------------------------------- epochs
    def _run_epoch(self, epoch: int) -> EpochRecord | None:
        assert self._steps is not None
        started = self._clock.elapsed()
        learning_rate = current_learning_rate(self._process.optimizer)
        train = self._run_phase(Phase.TRAIN, epoch)
        if train is None:
            return None
        validation = None
        if self._steps.validation is not None:
            validation = self._run_phase(Phase.VALIDATION, epoch)
            if validation is None:
                return None
        record = EpochRecord(
            epoch=epoch,
            train_loss=train.loss,
            val_loss=validation.loss if validation else None,
            train_metrics=train.metrics,
            val_metrics=validation.metrics if validation else {},
            learning_rate=learning_rate,
            duration=self._clock.elapsed() - started,
            train_batches=train.batches,
            val_batches=validation.batches if validation else 0,
        )
        self._step_scheduler(record)
        for hook in self._steps.epoch_end:
            hook(record)
        return record

    def _finish_epoch(self, record: EpochRecord) -> None:
        self._epoch = record.epoch
        self._history.append(record)
        weights = self._tracker.snapshot(record.epoch)
        architecture = None
        if self._architecture_changed:
            architecture = self._tracker.architecture.describe()
            self._architecture_changed = False
        event = EpochCompleted(
            record=record,
            weights=weights,
            total_epochs=self._control.total_epochs,
            learning_rate=current_learning_rate(self._process.optimizer),
            batch_size=self._batch_size,
            elapsed=self._clock.elapsed(),
            architecture=architecture,
        )
        # The hand-off: wait until the frontend has applied and drawn this epoch.
        if not self._channel.deliver(event):
            self._control.stop()
            return
        self._control.epoch_completed(record.epoch)
        if self._control.delay > 0:
            self._sleep(self._control.delay)

    def _run_phase(self, phase: Phase, epoch: int) -> _PhaseResult | None:
        assert self._steps is not None
        training = phase is Phase.TRAIN
        data = self._process.train_data if training else self._process.val_data
        step = self._steps.train if training else self._steps.validation
        step_name = "train_step" if training else "validation_step"
        assert data is not None and step is not None
        total = count_batches(data)
        self._process.model.train(training)
        loss_sum = 0.0
        batches = 0
        metric_sums: dict[str, float] = {}
        metric_counts: dict[str, int] = {}
        warned = False

        with torch.enable_grad() if training else torch.no_grad():
            for index, batch in enumerate(data, start=1):
                tracing = training and self._trace_pending
                if tracing:
                    self._tracker.begin_trace()
                try:
                    output = step(batch)
                finally:
                    if tracing:
                        self._trace_pending = False
                        self._architecture_changed |= self._tracker.end_trace()
                result = read_step_output(output, step_name=step_name)
                if not math.isfinite(result.loss) and not warned:
                    warned = True
                    self._log(LogLevel.WARNING, f"non-finite {phase.value} loss in epoch {epoch}")
                loss_sum += result.loss
                batches += 1
                for name, value in result.metrics.items():
                    metric_sums[name] = metric_sums.get(name, 0.0) + value
                    metric_counts[name] = metric_counts.get(name, 0) + 1
                if self._batch_size is None:
                    self._batch_size = infer_batch_size(batch)
                self._report_progress(epoch, phase, index, total, result.loss)
                if not self._checkpoint(phase, index, epoch):
                    return None

        if batches == 0:
            source = "train_data" if training else "val_data"
            raise TrainingError(f"{source} produced no batches in epoch {epoch}")
        # each metric is averaged over the batches that reported it
        means = {name: metric_sums[name] / metric_counts[name] for name in metric_sums}
        return _PhaseResult(loss_sum / batches, means, batches)

    def _step_scheduler(self, record: EpochRecord) -> None:
        scheduler = self._process.scheduler
        if scheduler is None:
            return
        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(record.monitored_loss)
        else:
            scheduler.step()

    # --------------------------------------------------------------- control
    def _checkpoint(self, phase: Phase | None, batch: int, epoch: int | None) -> bool:
        """Safe point between batches/epochs. Returns False when the session must stop."""
        self._pump.poll()
        self._apply_requests()
        if self._control.stop_requested:
            return False
        if self._control.paused:
            self._pause(phase, batch, epoch)
        return not self._control.stop_requested

    def _pause(self, phase: Phase | None, batch: int, epoch: int | None) -> None:
        self._clock.pause()
        in_progress = epoch if epoch is not None else self._epoch
        weights = self._tracker.snapshot(self._epoch, baseline=False)
        event = TrainingPaused(in_progress, phase, batch, weights, self._clock.elapsed())
        if not self._channel.deliver(event):
            self._control.stop()
            return
        if phase is None:
            where = f"before epoch {self._epoch + 1}"
        else:
            where = f"epoch {in_progress}, {phase.value} batch {batch}"
        self._set_state(TrainingState.PAUSED, f"paused ({where}) - 'resume' or 'step' continues")
        while self._control.paused and not self._control.stop_requested:
            self._pump.wait(0.1)
            self._apply_requests()
        if not self._control.stop_requested:
            self._clock.start()
            self._set_state(TrainingState.RUNNING, "resumed")

    def _sleep(self, seconds: float) -> None:
        """The optional per-epoch delay; ends early on stop or pause."""
        deadline = time.monotonic() + seconds
        while not (self._control.stop_requested or self._control.paused):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return
            self._pump.wait(min(remaining, 0.1))
            self._apply_requests()

    def _apply_requests(self) -> None:
        """Run queued actions and publish any setting changes."""
        for action, description in self._control.take_actions():
            try:
                message = action(self._target)
            except Exception as exc:
                what = description or "action"
                self._log(LogLevel.ERROR, f"{what} failed: {type(exc).__name__}: {exc}")
            else:
                if message:
                    self._log(LogLevel.SUCCESS, message)
        if self._control.revision != self._seen_revision:
            self._seen_revision = self._control.revision
            self._publish_status()

    # ---------------------------------------------------------------- events
    def _report_progress(
        self, epoch: int, phase: Phase, index: int, total: int | None, loss: float
    ) -> None:
        now = time.perf_counter()
        if index not in (1, total) and now - self._last_progress < self._progress_interval:
            return
        self._last_progress = now
        self._channel.notify(
            BatchProgress(epoch, phase, index, total, loss, self._batch_size, self._clock.elapsed())
        )

    def _set_state(self, state: TrainingState, message: str = "") -> None:
        self._state = state
        self._publish_status(message)

    def _publish_status(self, message: str = "") -> None:
        self._channel.notify(
            StateChanged(
                state=self._state,
                epoch=self._epoch,
                total_epochs=self._control.total_epochs,
                learning_rate=current_learning_rate(self._process.optimizer),
                delay=self._control.delay,
                elapsed=self._clock.elapsed(),
                message=message,
            )
        )

    def _log(self, level: LogLevel, text: str) -> None:
        self._channel.notify(LogMessage(text, level))

    def _safe_snapshot(self) -> WeightSnapshot | None:
        try:
            return self._tracker.snapshot(self._epoch, baseline=False)
        except Exception:
            return None

    def _finish_reason(self, state: TrainingState) -> str:
        epochs = len(self._history)
        if state is TrainingState.FINISHED:
            return f"completed {epochs} epoch{'s' if epochs != 1 else ''}"
        if state is TrainingState.FAILED:
            return f"failed after {epochs} completed epoch{'s' if epochs != 1 else ''}"
        return f"stopped after {epochs} epoch{'s' if epochs != 1 else ''}"

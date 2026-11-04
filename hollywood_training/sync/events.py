"""Events sent from the training thread to a frontend, and the records they carry.

Every event is an immutable slot dataclass without references to live modules or tensors,
so it can cross threads safely. Four of them are **synchronous**: the training thread
waits until the frontend has fully applied (and drawn) them before it continues.

=====================  ===========  =================================================
event                  synchronous  sent when
=====================  ===========  =================================================
``SessionStarted``     yes          once, before the first epoch (initial weights)
``EpochCompleted``     yes          after **every** epoch (new weights, losses, ...)
``TrainingPaused``     yes          when a pause takes effect (current weights)
``SessionFinished``    yes          once, at the end (final weights, outcome)
``BatchProgress``      no           during an epoch, throttled
``StateChanged``       no           state / setting changes (pause, lr, epochs, ...)
``LogMessage``         no           text for the frontend's log / terminal
=====================  ===========  =================================================
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from enum import Enum

from ..tracking.architecture import ArchitectureInfo
from ..tracking.weights import WeightSnapshot


class TrainingState(str, Enum):
    """Lifecycle state of a managed training session."""

    PREPARING = "preparing"
    RUNNING = "running"
    PAUSED = "paused"
    FINISHED = "finished"  # all requested epochs completed (more may still be requested)
    STOPPED = "stopped"  # ended early on request
    FAILED = "failed"  # user code raised an exception

    @property
    def active(self) -> bool:
        return self in (TrainingState.PREPARING, TrainingState.RUNNING, TrainingState.PAUSED)


class Phase(str, Enum):
    TRAIN = "train"
    VALIDATION = "validation"


class LogLevel(str, Enum):
    INFO = "info"
    SUCCESS = "success"
    WARNING = "warning"
    ERROR = "error"


@dataclass(frozen=True, slots=True)
class EpochRecord:
    """Results of one completed epoch."""

    epoch: int
    train_loss: float
    val_loss: float | None
    train_metrics: Mapping[str, float] = field(default_factory=dict)
    val_metrics: Mapping[str, float] = field(default_factory=dict)
    learning_rate: float = 0.0  # learning rate used during the epoch
    duration: float = 0.0  # seconds spent training + validating
    train_batches: int = 0
    val_batches: int = 0

    @property
    def monitored_loss(self) -> float:
        """The loss that decides "best": validation loss if there is one, else training loss."""
        return self.val_loss if self.val_loss is not None else self.train_loss


def best_epoch(records: Iterable[EpochRecord]) -> EpochRecord | None:
    """The record with the lowest finite :attr:`EpochRecord.monitored_loss` (None if none)."""
    finite = [record for record in records if math.isfinite(record.monitored_loss)]
    return min(finite, key=lambda record: record.monitored_loss) if finite else None


@dataclass(frozen=True, slots=True)
class SessionInfo:
    """Static description of a session, shown by frontends."""

    network_class: str
    display_name: str | None
    architecture: ArchitectureInfo
    total_epochs: int
    batch_size: int | None
    train_batches: int | None
    val_batches: int | None
    has_validation: bool
    loss_function: str
    optimizer: str
    optimizer_details: str
    normalization: str
    scheduler: str | None
    device: str
    learning_rate: float
    neuron_limit: int


@dataclass(frozen=True, slots=True, eq=False)
class SessionStarted:
    info: SessionInfo
    weights: WeightSnapshot
    state: TrainingState


@dataclass(frozen=True, slots=True)
class BatchProgress:
    epoch: int
    phase: Phase
    batch: int
    total_batches: int | None
    loss: float | None
    batch_size: int | None
    elapsed: float


@dataclass(frozen=True, slots=True, eq=False)
class EpochCompleted:
    record: EpochRecord
    weights: WeightSnapshot
    total_epochs: int
    learning_rate: float  # learning rate for the next epoch (after any scheduler step)
    batch_size: int | None
    elapsed: float
    architecture: ArchitectureInfo | None = None  # set when the architecture was refined


@dataclass(frozen=True, slots=True, eq=False)
class TrainingPaused:
    epoch: int  # epoch in progress (completed epochs + 1) or completed epochs at a boundary
    phase: Phase | None
    batch: int
    weights: WeightSnapshot
    elapsed: float


@dataclass(frozen=True, slots=True)
class StateChanged:
    state: TrainingState
    epoch: int  # completed epochs
    total_epochs: int
    learning_rate: float
    delay: float
    elapsed: float
    message: str = ""


@dataclass(frozen=True, slots=True)
class LogMessage:
    text: str
    level: LogLevel = LogLevel.INFO


@dataclass(frozen=True, slots=True, eq=False)
class SessionFinished:
    state: TrainingState
    epochs_completed: int
    elapsed: float
    weights: WeightSnapshot | None
    reason: str
    error: str | None = None


Event = (
    SessionStarted
    | BatchProgress
    | EpochCompleted
    | TrainingPaused
    | StateChanged
    | LogMessage
    | SessionFinished
)

SYNCHRONOUS_EVENTS: tuple[type, ...] = (
    SessionStarted,
    EpochCompleted,
    TrainingPaused,
    SessionFinished,
)


def is_synchronous(event: object) -> bool:
    """True for events the training thread must wait on."""
    return isinstance(event, SYNCHRONOUS_EVENTS)

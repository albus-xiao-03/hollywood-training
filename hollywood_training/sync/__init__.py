"""Synchronisation between the training thread and a frontend.

* :mod:`.events`  - what crosses from training to the frontend
* :mod:`.channel` - how it crosses (blocking hand-off for synchronous events)
* :mod:`.control` - requests crossing back from the frontend to training
* :mod:`.pump`    - how the training loop gives the frontend time to run
* :mod:`.session` - the frontend-side state assembled from the events
"""

from .channel import Envelope, Sink, SyncChannel
from .control import ControlAction, TrainingControl, TrainingTarget
from .events import (
    SYNCHRONOUS_EVENTS,
    BatchProgress,
    EpochCompleted,
    EpochRecord,
    Event,
    LogLevel,
    LogMessage,
    Phase,
    SessionFinished,
    SessionInfo,
    SessionStarted,
    StateChanged,
    TrainingPaused,
    TrainingState,
    best_epoch,
    is_synchronous,
)
from .pump import EventPump, WaitingPump
from .session import SessionState

__all__ = [
    "SYNCHRONOUS_EVENTS",
    "BatchProgress",
    "ControlAction",
    "Envelope",
    "EpochCompleted",
    "EpochRecord",
    "Event",
    "EventPump",
    "LogLevel",
    "LogMessage",
    "Phase",
    "SessionFinished",
    "SessionInfo",
    "SessionStarted",
    "SessionState",
    "Sink",
    "StateChanged",
    "SyncChannel",
    "TrainingControl",
    "TrainingPaused",
    "TrainingState",
    "TrainingTarget",
    "WaitingPump",
    "best_epoch",
    "is_synchronous",
]

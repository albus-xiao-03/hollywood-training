"""Synchronisation between the training thread and a frontend.

* :mod:`.events`  - what crosses from training to the frontend
* :mod:`.channel` - how it crosses (blocking hand-off for synchronous events)
* :mod:`.control` - requests crossing back from the frontend to training
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
    is_synchronous,
)
from .session import SessionState

__all__ = [
    "SYNCHRONOUS_EVENTS",
    "BatchProgress",
    "ControlAction",
    "Envelope",
    "EpochCompleted",
    "EpochRecord",
    "Event",
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
    "is_synchronous",
]

"""Training management: the user-defined process and the manager that runs it."""

from ..sync.events import EpochRecord
from .describe import (
    current_learning_rate,
    describe_loss,
    describe_normalization,
    describe_optimizer,
    model_device,
)
from .history import TrainingResult
from .manager import TrainingError, TrainingManager
from .process import EpochHook, ProcessError, ProcessSteps, TrainingProcess
from .steps import (
    StepFunction,
    StepOutputError,
    StepResult,
    infer_batch_size,
    read_step_output,
    supervised_train_step,
    supervised_validation_step,
)
from .timing import ActiveClock

__all__ = [
    "ActiveClock",
    "EpochHook",
    "EpochRecord",
    "ProcessError",
    "ProcessSteps",
    "StepFunction",
    "StepOutputError",
    "StepResult",
    "TrainingError",
    "TrainingManager",
    "TrainingProcess",
    "TrainingResult",
    "current_learning_rate",
    "describe_loss",
    "describe_normalization",
    "describe_optimizer",
    "infer_batch_size",
    "model_device",
    "read_step_output",
    "supervised_train_step",
    "supervised_validation_step",
]

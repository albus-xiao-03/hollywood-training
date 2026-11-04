"""What a managed run returns: :class:`TrainingResult` with the per-epoch history."""

from __future__ import annotations

from dataclasses import dataclass

from ..sync.events import EpochRecord, TrainingState, best_epoch
from ..tracking.weights import WeightSnapshot


@dataclass(frozen=True, slots=True, eq=False)
class TrainingResult:
    """Outcome of :func:`hollywood_training.run`."""

    state: TrainingState
    history: tuple[EpochRecord, ...]
    elapsed: float  # active training time in seconds (pauses excluded)
    error: BaseException | None = None
    final_weights: WeightSnapshot | None = None

    @property
    def epochs_completed(self) -> int:
        return len(self.history)

    @property
    def train_losses(self) -> list[float]:
        return [record.train_loss for record in self.history]

    @property
    def val_losses(self) -> list[float | None]:
        return [record.val_loss for record in self.history]

    def best_epoch(self) -> EpochRecord | None:
        """Epoch with the lowest finite validation loss (training loss when not validating)."""
        return best_epoch(self.history)

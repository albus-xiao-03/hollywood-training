"""Weight snapshot data: what a frontend receives about the weights (numpy only, no torch).

Produced by :mod:`hollywood_training.tracking.snapshot` in the training process; small,
immutable and picklable, so it can be drawn by another thread or another process.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

DEFAULT_NEURON_LIMIT = 20


@dataclass(frozen=True, slots=True)
class WeightStats:
    """Summary statistics of a full weight tensor."""

    mean: float
    std: float
    minimum: float
    maximum: float
    abs_mean: float
    norm: float
    count: int


@dataclass(frozen=True, slots=True, eq=False)
class LayerWeights:
    """Displayed connection matrix of one layer at one point in time."""

    name: str
    matrix: np.ndarray  # (len(out_index), len(in_index)) float32
    bias: np.ndarray | None  # (len(out_index),) float32
    delta: np.ndarray | None  # change of `matrix` since the previous snapshot
    in_index: np.ndarray  # displayed input feature indices
    out_index: np.ndarray  # displayed output feature indices
    in_features: int
    out_features: int
    stats: WeightStats
    update_ratio: float | None  # ||delta|| / ||previous matrix||

    @property
    def truncated(self) -> bool:
        """True when only a subset of the layer's neurons is displayed."""
        return len(self.in_index) < self.in_features or len(self.out_index) < self.out_features


@dataclass(frozen=True, slots=True, eq=False)
class WeightSnapshot:
    """Displayed weights of every connection layer after ``epoch`` epochs."""

    epoch: int
    layers: tuple[LayerWeights, ...]

    def layer(self, name: str) -> LayerWeights | None:
        return next((layer for layer in self.layers if layer.name == name), None)

    def names(self) -> tuple[str, ...]:
        return tuple(layer.name for layer in self.layers)


def display_indices(size: int, limit: int) -> np.ndarray:
    """Evenly spaced feature indices, at most ``limit`` of them, always incl. first and last."""
    if size <= 0:
        return np.zeros(0, dtype=np.int64)
    if size <= limit:
        return np.arange(size, dtype=np.int64)
    return np.unique(np.linspace(0, size - 1, max(limit, 2)).round().astype(np.int64))

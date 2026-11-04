"""Synthetic datasets for the demos - generated on the fly, nothing to download."""

from __future__ import annotations

import math

import torch
from torch.utils.data import TensorDataset


def make_spirals(
    points_per_arm: int = 300,
    arms: int = 3,
    turns: float = 0.75,
    noise: float = 0.2,
    seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Interleaved spiral arms in 2-D; the label is the arm index.

    Returns ``(points, labels)`` with shapes ``(arms * points_per_arm, 2)`` and
    ``(arms * points_per_arm,)``.
    """
    generator = torch.Generator().manual_seed(seed)
    points, labels = [], []
    for arm in range(arms):
        radius = torch.linspace(0.1, 1.0, points_per_arm)
        angle = (
            arm * 2 * math.pi / arms
            + radius * turns * 2 * math.pi
            + noise * torch.randn(points_per_arm, generator=generator) * radius
        )
        points.append(torch.stack((radius * torch.cos(angle), radius * torch.sin(angle)), dim=1))
        labels.append(torch.full((points_per_arm,), arm, dtype=torch.long))
    return torch.cat(points), torch.cat(labels)


def make_sine(
    samples: int = 600, noise: float = 0.08, seed: int = 0
) -> tuple[torch.Tensor, torch.Tensor]:
    """Noisy samples of ``sin(x)`` for ``x`` in ``[-pi, pi]``, shaped ``(samples, 1)``."""
    generator = torch.Generator().manual_seed(seed)
    x = (torch.rand(samples, 1, generator=generator) * 2 - 1) * math.pi
    y = torch.sin(x) + noise * torch.randn(samples, 1, generator=generator)
    return x, y


class Standardizer:
    """z-score normalisation fitted on training data (the "normalisation strategy")."""

    def __init__(self, data: torch.Tensor) -> None:
        self.mean = data.mean(dim=0)
        self.std = data.std(dim=0).clamp_min(1e-8)

    def __call__(self, data: torch.Tensor) -> torch.Tensor:
        return (data - self.mean) / self.std


def split(
    inputs: torch.Tensor, targets: torch.Tensor, validation: float = 0.2, seed: int = 0
) -> tuple[TensorDataset, TensorDataset]:
    """Shuffle and split into training and validation datasets."""
    order = torch.randperm(len(inputs), generator=torch.Generator().manual_seed(seed))
    cut = int(len(inputs) * (1 - validation))
    train, val = order[:cut], order[cut:]
    return TensorDataset(inputs[train], targets[train]), TensorDataset(inputs[val], targets[val])


def pick_device(name: str) -> torch.device:
    """``auto`` → CUDA, then Apple MPS, then CPU."""
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

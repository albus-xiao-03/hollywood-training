"""Training steps: the default supervised steps and reading what a step returns.

A *step* is a plain function ``step(batch)`` containing ordinary PyTorch code for one batch.
It returns the batch loss - a scalar tensor or a number - or a mapping with a ``"loss"``
entry plus any extra metrics, e.g. ``{"loss": loss, "accuracy": acc}``.
"""

from __future__ import annotations

import numbers
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

import torch
from torch import nn

from .describe import model_device

StepFunction = Callable[[Any], Any]


class StepOutputError(TypeError):
    """A step returned something that is not a loss."""


@dataclass(frozen=True, slots=True)
class StepResult:
    """Loss and metrics of one batch, as plain floats."""

    loss: float
    metrics: dict[str, float] = field(default_factory=dict)


def _scalar(value: object, what: str) -> float:
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            raise StepOutputError(
                f"{what} must be a scalar, got a tensor of shape {tuple(value.shape)}"
            )
        return float(value.detach().item())
    if isinstance(value, numbers.Real) and not isinstance(value, bool):
        return float(value)
    raise StepOutputError(f"{what} must be a number or a scalar tensor, got {type(value).__name__}")


def read_step_output(value: object, *, step_name: str) -> StepResult:
    """Normalise a step's return value into a :class:`StepResult`."""
    if isinstance(value, Mapping):
        if "loss" not in value:
            raise StepOutputError(f"{step_name} returned a mapping without a 'loss' entry")
        metrics = {
            str(key): _scalar(item, f"{step_name} metric {key!r}")
            for key, item in value.items()
            if key != "loss"
        }
        return StepResult(_scalar(value["loss"], f"{step_name} loss"), metrics)
    if value is None:
        raise StepOutputError(f"{step_name} must return the batch loss, but returned None")
    return StepResult(_scalar(value, f"{step_name} result"))


def infer_batch_size(batch: object) -> int | None:
    """Leading dimension of the first tensor found in ``batch``."""
    if isinstance(batch, torch.Tensor):
        return int(batch.shape[0]) if batch.dim() > 0 else None
    if isinstance(batch, (list, tuple)) and batch:
        return infer_batch_size(batch[0])
    if isinstance(batch, Mapping) and batch:
        return infer_batch_size(next(iter(batch.values())))
    return None


def move_to_device(value: Any, device: torch.device) -> Any:
    """Move tensors (also inside lists, tuples and mappings) to ``device``."""
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, Mapping):
        return {key: move_to_device(item, device) for key, item in value.items()}
    if isinstance(value, tuple) and hasattr(value, "_fields"):  # namedtuple
        return type(value)(*(move_to_device(item, device) for item in value))
    if isinstance(value, (list, tuple)):
        return type(value)(move_to_device(item, device) for item in value)
    return value


def split_supervised_batch(batch: Any) -> tuple[Any, Any]:
    """``(inputs, targets)`` from a two-item batch (what ``TensorDataset`` yields)."""
    if isinstance(batch, (list, tuple)) and len(batch) == 2:
        return batch[0], batch[1]
    raise ValueError(
        "the default steps expect batches of (inputs, targets); "
        "register a custom train_step/validation_step for other batch layouts"
    )


def supervised_train_step(
    model: nn.Module, optimizer: torch.optim.Optimizer, loss_fn: Callable[..., torch.Tensor]
) -> StepFunction:
    """The textbook step: ``zero_grad`` → forward → loss → ``backward`` → ``step``."""

    def default_train_step(batch: Any) -> torch.Tensor:
        inputs, targets = split_supervised_batch(batch)
        device = model_device(model)
        inputs, targets = move_to_device(inputs, device), move_to_device(targets, device)
        optimizer.zero_grad()
        loss = loss_fn(model(inputs), targets)
        loss.backward()
        optimizer.step()
        return loss

    return default_train_step


def supervised_validation_step(
    model: nn.Module, loss_fn: Callable[..., torch.Tensor]
) -> StepFunction:
    """Forward pass and loss only (the manager already applies ``eval()`` and ``no_grad``)."""

    def default_validation_step(batch: Any) -> torch.Tensor:
        inputs, targets = split_supervised_batch(batch)
        device = model_device(model)
        inputs, targets = move_to_device(inputs, device), move_to_device(targets, device)
        return loss_fn(model(inputs), targets)

    return default_validation_step

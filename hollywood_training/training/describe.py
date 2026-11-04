"""Human-readable descriptions of the training components, read from public attributes.

Everything shown in the GUI's status panel (loss function, optimizer, normalisation
strategy, device, batch size, ...) is derived here from objects the user handed over -
``type(...).__name__``, ``optimizer.defaults``, ``optimizer.param_groups``,
``DataLoader.batch_size`` - or from explicit values given to ``TrainingProcess``.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Iterable
from typing import TYPE_CHECKING, Any

import torch
from torch import nn

from ..tracking.architecture import ArchitectureInfo
from ..tracking.layers import is_uninitialized

if TYPE_CHECKING:
    from ..sync.events import SessionInfo
    from .process import TrainingProcess

_OPTIMIZER_SETTINGS = (
    ("momentum", "momentum"),
    ("betas", "betas"),
    ("alpha", "alpha"),
    ("rho", "rho"),
    ("weight_decay", "wd"),
    ("dampening", "dampening"),
    ("nesterov", "nesterov"),
    ("amsgrad", "amsgrad"),
)


def _number(value: float) -> str:
    return f"{value:g}"


def model_device(model: nn.Module) -> torch.device:
    """Device of the model's first initialised parameter (or buffer); CPU otherwise."""
    for tensor in itertools.chain(model.parameters(), model.buffers()):
        if not is_uninitialized(tensor):
            return tensor.device
    return torch.device("cpu")


def describe_device(device: torch.device) -> str:
    if device.type == "cuda" and torch.cuda.is_available():
        index = device.index if device.index is not None else torch.cuda.current_device()
        return f"cuda:{index} ({torch.cuda.get_device_name(index)})"
    if device.type == "cpu":
        threads = torch.get_num_threads()
        return f"cpu ({threads} thread{'s' if threads != 1 else ''})"
    return str(device)


def describe_loss(loss_fn: Any) -> str:
    if loss_fn is None:
        return "defined in step"
    if isinstance(loss_fn, nn.Module):
        reduction = getattr(loss_fn, "reduction", None)
        name = type(loss_fn).__name__
        return f"{name} ({reduction})" if isinstance(reduction, str) else name
    return getattr(loss_fn, "__name__", type(loss_fn).__name__)


def describe_optimizer(optimizer: Any) -> tuple[str, str]:
    """``(name, details)``, e.g. ``("Adam", "betas=0.9/0.999")``."""
    defaults = getattr(optimizer, "defaults", None) or {}
    parts: list[str] = []
    for key, label in _OPTIMIZER_SETTINGS:
        value = defaults.get(key)
        if value is None or value is False:
            continue
        if value is True:
            parts.append(label)
        elif isinstance(value, (tuple, list)):
            parts.append(f"{label}={'/'.join(_number(float(v)) for v in value)}")
        elif isinstance(value, (int, float)) and value != 0:
            parts.append(f"{label}={_number(float(value))}")
    groups = len(getattr(optimizer, "param_groups", ()))
    if groups > 1:
        parts.append(f"{groups} param groups")
    return type(optimizer).__name__, ", ".join(parts)


def describe_scheduler(scheduler: Any) -> str | None:
    return None if scheduler is None else type(scheduler).__name__


def describe_normalization(architecture: ArchitectureInfo, data_normalization: str | None) -> str:
    """Data normalisation (declared by the user) plus normalisation layers found in the net."""
    parts = [f"data: {data_normalization}"] if data_normalization else []
    parts.append(f"layers: {architecture.normalization_summary()}")
    return " · ".join(parts)


def current_learning_rate(optimizer: Any) -> float:
    groups = getattr(optimizer, "param_groups", None)
    if not groups:
        return math.nan
    return float(groups[0].get("lr", math.nan))


def count_batches(data: Iterable[Any] | None) -> int | None:
    """``len(data)`` when the data source knows it (a ``DataLoader`` does)."""
    if data is None:
        return None
    try:
        return len(data)  # type: ignore[arg-type]
    except TypeError:
        return None


def resolve_batch_size(explicit: int | None, data: Iterable[Any]) -> int | None:
    if explicit is not None:
        return int(explicit)
    size = getattr(data, "batch_size", None)
    return size if isinstance(size, int) else None


def build_session_info(
    process: TrainingProcess,
    architecture: ArchitectureInfo,
    *,
    total_epochs: int,
    neuron_limit: int,
) -> SessionInfo:
    from ..sync.events import SessionInfo

    optimizer, details = describe_optimizer(process.optimizer)
    return SessionInfo(
        network_class=architecture.class_name,
        display_name=process.name or architecture.display_name,
        architecture=architecture,
        total_epochs=total_epochs,
        batch_size=resolve_batch_size(process.batch_size, process.train_data),
        train_batches=count_batches(process.train_data),
        val_batches=count_batches(process.val_data),
        has_validation=process.val_data is not None,
        loss_function=describe_loss(process.loss_fn),
        optimizer=optimizer,
        optimizer_details=details,
        normalization=describe_normalization(architecture, process.normalization),
        scheduler=describe_scheduler(process.scheduler),
        device=describe_device(model_device(process.model)),
        learning_rate=current_learning_rate(process.optimizer),
        neuron_limit=neuron_limit,
    )

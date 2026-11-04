"""The ``@track`` decorator: declare a network for architecture and weight tracking.

``track`` works on a class (as a decorator, with or without options) and on an instance
(as a plain function, for models you did not define yourself, e.g. ``nn.Sequential``)::

    @track
    class Net(nn.Module): ...

    @track(name="Spiral classifier", output_labels=("A", "B", "C"))
    class Net(nn.Module): ...

    model = track(nn.Sequential(nn.Linear(2, 8), nn.ReLU(), nn.Linear(8, 1)))

The decorated class (or instance) is returned unchanged: the declaration is stored in a
registry held by this module, so ``__init__``, ``forward`` and pickling behave exactly as
in plain PyTorch. Subclasses of a tracked class inherit its declaration.
"""

from __future__ import annotations

import weakref
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TypeVar, overload

from torch import nn

T = TypeVar("T")


@dataclass(frozen=True, slots=True)
class TrackOptions:
    """Display options declared with :func:`track`."""

    name: str | None = None
    input_labels: tuple[str, ...] | None = None
    output_labels: tuple[str, ...] | None = None


_class_options: weakref.WeakKeyDictionary[type, TrackOptions] = weakref.WeakKeyDictionary()
_instance_options: weakref.WeakKeyDictionary[nn.Module, TrackOptions] = weakref.WeakKeyDictionary()


def _labels(values: Sequence[str] | None, argument: str) -> tuple[str, ...] | None:
    if values is None:
        return None
    if isinstance(values, str):
        raise TypeError(f"{argument} must be a sequence of strings, not a single string")
    return tuple(str(value) for value in values)


def _register(target: T, options: TrackOptions) -> T:
    if isinstance(target, type):
        if not issubclass(target, nn.Module):
            raise TypeError(f"@track expects a torch.nn.Module subclass, got {target.__name__}")
        _class_options[target] = options
        return target
    if isinstance(target, nn.Module):
        _instance_options[target] = options
        return target
    raise TypeError(
        f"track() expects a torch.nn.Module subclass or instance, got {type(target).__name__}"
    )


@overload
def track(target: T, /) -> T: ...


@overload
def track(
    target: None = None,
    /,
    *,
    name: str | None = ...,
    input_labels: Sequence[str] | None = ...,
    output_labels: Sequence[str] | None = ...,
) -> Callable[[T], T]: ...


def track(
    target: T | None = None,
    /,
    *,
    name: str | None = None,
    input_labels: Sequence[str] | None = None,
    output_labels: Sequence[str] | None = None,
) -> T | Callable[[T], T]:
    """Declare a network (class or instance) for tracking.

    Args:
        target: an ``nn.Module`` subclass or instance. Omit it to pass options.
        name: optional display name shown next to the network's class name.
        input_labels: optional names of the input features (first column of neurons).
        output_labels: optional names of the outputs (last column of neurons).

    Returns:
        The target itself, unchanged, or a decorator when called with options only.
    """
    options = TrackOptions(
        name=name,
        input_labels=_labels(input_labels, "input_labels"),
        output_labels=_labels(output_labels, "output_labels"),
    )
    if target is None:
        return lambda obj: _register(obj, options)
    return _register(target, options)


def track_options(model: nn.Module) -> TrackOptions | None:
    """Options declared for ``model`` (instance first, then its class hierarchy)."""
    options = _instance_options.get(model)
    if options is not None:
        return options
    for cls in type(model).__mro__:
        options = _class_options.get(cls)
        if options is not None:
            return options
    return None


def is_tracked(model: nn.Module) -> bool:
    """True when ``model`` (or one of its classes) was declared with :func:`track`."""
    return track_options(model) is not None

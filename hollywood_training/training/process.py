"""``TrainingProcess``: a training process written in plain PyTorch, wrapped for the manager.

Write the training of one batch exactly as you would without this package, then wrap it -
by passing the functions to the constructor, or with the step decorators::

    process = TrainingProcess(model, optimizer, train_loader,
                              loss_fn=loss_fn, val_data=val_loader, epochs=50)

    @process.train_step
    def train_step(batch):
        inputs, targets = batch
        optimizer.zero_grad()
        loss = loss_fn(model(inputs), targets)
        loss.backward()
        optimizer.step()
        return loss

    @process.validation_step
    def validation_step(batch):
        inputs, targets = batch
        return loss_fn(model(inputs), targets)

    hollywood_training.run(process)

The decorators register and return the function unchanged, so it stays directly callable.
Without a ``train_step`` / ``validation_step`` the textbook supervised steps are used; they
need ``loss_fn`` and batches of ``(inputs, targets)``.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import dataclass
from typing import Any

from torch import nn

from ..commands.registry import Argument, Command, Handler
from ..sync.events import EpochRecord
from .steps import StepFunction, supervised_train_step, supervised_validation_step

EpochHook = Callable[[EpochRecord], None]


class ProcessError(ValueError):
    """A :class:`TrainingProcess` that cannot run as configured."""


@dataclass(frozen=True, slots=True)
class ProcessSteps:
    """The step functions the manager will call."""

    train: StepFunction
    validation: StepFunction | None
    epoch_end: tuple[EpochHook, ...]


def _check_reiterable(data: object, argument: str) -> None:
    if not isinstance(data, Iterable):
        raise TypeError(f"{argument} must be iterable, e.g. a torch DataLoader")
    if isinstance(data, Iterator):
        raise TypeError(
            f"{argument} is a one-shot iterator; pass a re-iterable source such as a "
            "DataLoader or a list so that every epoch sees the data"
        )


def _check_callable(function: object, what: str) -> None:
    if not callable(function):
        raise TypeError(f"{what} must be callable, got {type(function).__name__}")


class TrainingProcess:
    """Everything the manager needs to run - and display - one training process.

    Args:
        model: the network being trained.
        optimizer: its optimizer.
        train_data: re-iterable source of training batches (usually a ``DataLoader``).
        epochs: number of epochs to run (can be changed at runtime with ``epochs <n>``).
        loss_fn: the loss function; required by the default steps, shown in the GUI.
        val_data: optional re-iterable source of validation batches.
        scheduler: optional learning-rate scheduler, stepped after every epoch
            (``ReduceLROnPlateau`` receives the validation loss, or the training loss).
        train_step: ``step(batch)`` training one batch; returns its loss (or a mapping
            with ``"loss"`` and extra metrics). Same as decorating with ``@train_step``.
        validation_step: ``step(batch)`` returning the validation loss of one batch.
        on_epoch_end: hook(s) ``hook(record)`` called after every epoch.
        batch_size: shown in the GUI; read from ``train_data.batch_size`` when omitted.
        normalization: description of the data normalisation, e.g. ``"z-score"``.
        name: display name of the run (defaults to the ``@track`` name, if any).
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: Any,
        train_data: Iterable[Any],
        *,
        epochs: int,
        loss_fn: Callable[..., Any] | None = None,
        val_data: Iterable[Any] | None = None,
        scheduler: Any | None = None,
        train_step: StepFunction | None = None,
        validation_step: StepFunction | None = None,
        on_epoch_end: EpochHook | Sequence[EpochHook] | None = None,
        batch_size: int | None = None,
        normalization: str | None = None,
        name: str | None = None,
    ) -> None:
        if not isinstance(model, nn.Module):
            raise TypeError(f"model must be a torch.nn.Module, got {type(model).__name__}")
        if not hasattr(optimizer, "param_groups"):
            raise TypeError("optimizer must be a torch.optim.Optimizer (it has no param_groups)")
        _check_reiterable(train_data, "train_data")
        if val_data is not None:
            _check_reiterable(val_data, "val_data")
        if isinstance(epochs, bool) or not isinstance(epochs, int) or epochs < 1:
            raise ValueError(f"epochs must be a positive integer, got {epochs!r}")
        if batch_size is not None and (
            isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1
        ):
            raise ValueError(f"batch_size must be a positive integer, got {batch_size!r}")
        if loss_fn is not None:
            _check_callable(loss_fn, "loss_fn")

        self._model = model
        self._optimizer = optimizer
        self._train_data = train_data
        self._val_data = val_data
        self._loss_fn = loss_fn
        self._scheduler = scheduler
        self._epochs = epochs
        self._batch_size = batch_size
        self._normalization = normalization
        self._name = name
        self._train_step: StepFunction | None = None
        self._validation_step: StepFunction | None = None
        self._epoch_hooks: list[EpochHook] = []
        self._commands: list[Command] = []

        if train_step is not None:
            self.train_step(train_step)
        if validation_step is not None:
            self.validation_step(validation_step)
        if on_epoch_end is not None:
            hooks = [on_epoch_end] if callable(on_epoch_end) else list(on_epoch_end)
            for hook in hooks:
                self.on_epoch_end(hook)

    # ----------------------------------------------------------------- components
    @property
    def model(self) -> nn.Module:
        return self._model

    @property
    def optimizer(self) -> Any:
        return self._optimizer

    @property
    def train_data(self) -> Iterable[Any]:
        return self._train_data

    @property
    def val_data(self) -> Iterable[Any] | None:
        return self._val_data

    @property
    def loss_fn(self) -> Callable[..., Any] | None:
        return self._loss_fn

    @property
    def scheduler(self) -> Any | None:
        return self._scheduler

    @property
    def epochs(self) -> int:
        return self._epochs

    @property
    def batch_size(self) -> int | None:
        return self._batch_size

    @property
    def normalization(self) -> str | None:
        return self._normalization

    @property
    def name(self) -> str | None:
        return self._name

    @property
    def commands(self) -> tuple[Command, ...]:
        """Custom commands registered with :meth:`command`."""
        return tuple(self._commands)

    # ------------------------------------------------------- step registration
    def train_step(self, function: StepFunction) -> StepFunction:
        """Decorator: register ``function(batch)`` as the training step."""
        _check_callable(function, "train_step")
        self._train_step = function
        return function

    def validation_step(self, function: StepFunction) -> StepFunction:
        """Decorator: register ``function(batch)`` as the validation step."""
        _check_callable(function, "validation_step")
        self._validation_step = function
        return function

    def on_epoch_end(self, function: EpochHook) -> EpochHook:
        """Decorator: call ``function(record)`` after every epoch (any number of hooks)."""
        _check_callable(function, "on_epoch_end hook")
        self._epoch_hooks.append(function)
        return function

    def command(
        self,
        name: str,
        *,
        summary: str,
        arguments: Sequence[Argument] = (),
        aliases: Sequence[str] = (),
        details: str = "",
    ) -> Callable[[Handler], Handler]:
        """Decorator: add a custom terminal command ``handler(context, *arguments)``."""

        def register(handler: Handler) -> Handler:
            _check_callable(handler, f"command {name!r}")
            self._commands.append(
                Command(
                    name=name,
                    summary=summary,
                    handler=handler,
                    arguments=tuple(arguments),
                    aliases=tuple(aliases),
                    category="custom",
                    details=details,
                )
            )
            return handler

        return register

    # ------------------------------------------------------------- resolution
    def resolve_steps(self) -> ProcessSteps:
        """The step functions to run, falling back to the default supervised steps."""
        train = self._train_step
        if train is None:
            if self._loss_fn is None:
                raise ProcessError(
                    "no train_step was defined; register one with @process.train_step or "
                    "pass loss_fn to use the default supervised step"
                )
            train = supervised_train_step(self._model, self._optimizer, self._loss_fn)

        validation = None
        if self._val_data is not None:
            validation = self._validation_step
            if validation is None:
                if self._loss_fn is None:
                    raise ProcessError(
                        "val_data was given but no validation_step; register one with "
                        "@process.validation_step or pass loss_fn"
                    )
                validation = supervised_validation_step(self._model, self._loss_fn)
        elif self._validation_step is not None:
            raise ProcessError("a validation_step was registered but no val_data was given")
        return ProcessSteps(train, validation, tuple(self._epoch_hooks))

    def __repr__(self) -> str:
        validation = "yes" if self._val_data is not None else "no"
        return (
            f"TrainingProcess(model={type(self._model).__name__}, epochs={self._epochs}, "
            f"validation={validation})"
        )

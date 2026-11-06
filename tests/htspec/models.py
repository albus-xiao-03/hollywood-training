"""Small native PyTorch networks, data and training setups for the tests.

Everything here is ordinary PyTorch. Networks are defined *without* ``@ht.track`` unless the
name says so, so each test decides how (and whether) a network is tracked.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


# ----------------------------------------------------------------------------- networks
class TinyNet(nn.Module):
    """inputs -> hidden -> outputs classifier with a Tanh between the two Linear layers."""

    def __init__(self, hidden: int = 8, inputs: int = 2, outputs: int = 3) -> None:
        super().__init__()
        self.fc1 = nn.Linear(inputs, hidden)
        self.act = nn.Tanh()
        self.fc2 = nn.Linear(hidden, outputs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


class DeepNet(nn.Module):
    """Three Linear layers with BatchNorm, ReLU, Tanh and Dropout between them (4 inputs)."""

    def __init__(self, hidden: int = 12) -> None:
        super().__init__()
        self.fc1 = nn.Linear(4, hidden)
        self.norm = nn.BatchNorm1d(hidden)
        self.act1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden, hidden)
        self.act2 = nn.Tanh()
        self.drop = nn.Dropout(0.1)
        self.fc3 = nn.Linear(hidden, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act1(self.norm(self.fc1(x)))
        return self.fc3(self.drop(self.act2(self.fc2(x))))


class ConvNet(nn.Module):
    """1 x 8 x 8 images: Conv2d -> ReLU -> MaxPool -> Flatten -> Linear (a reshape between)."""

    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(1, 4, kernel_size=3, padding=1)
        self.act = nn.ReLU()
        self.pool = nn.MaxPool2d(2)
        self.flat = nn.Flatten()
        self.fc = nn.Linear(4 * 4 * 4, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.flat(self.pool(self.act(self.conv(x)))))


class WideNet(nn.Module):
    """A hidden layer far wider than the default 20 drawn neurons (3 inputs, 2 outputs)."""

    def __init__(self, width: int = 200) -> None:
        super().__init__()
        self.fc1 = nn.Linear(3, width)
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(width, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


class ReorderedNet(nn.Module):
    """Modules registered in a different order than they run (out, act, inp)."""

    def __init__(self) -> None:
        super().__init__()
        self.out = nn.Linear(6, 3)
        self.act = nn.ReLU()
        self.inp = nn.Linear(2, 6)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.out(self.act(self.inp(x)))


class SharedNet(nn.Module):
    """One Linear layer used twice in the forward pass."""

    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(2, 2)
        self.act = nn.Tanh()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.act(self.fc(x)))


def sequential_net() -> nn.Sequential:
    return nn.Sequential(nn.Linear(2, 6), nn.ReLU(), nn.Linear(6, 3))


# --------------------------------------------------------------------------------- data
def classification_data(
    samples: int = 64, features: int = 2, classes: int = 3, seed: int = 0
) -> TensorDataset:
    generator = torch.Generator().manual_seed(seed)
    inputs = torch.randn(samples, features, generator=generator)
    targets = torch.randint(0, classes, (samples,), generator=generator)
    return TensorDataset(inputs, targets)


def learnable_data(samples: int = 128, seed: int = 0) -> TensorDataset:
    """2-D points labelled by quadrant pairs - a TinyNet can learn it within a few epochs."""
    generator = torch.Generator().manual_seed(seed)
    inputs = torch.randn(samples, 2, generator=generator)
    targets = (inputs[:, 0] > 0).long() + (inputs[:, 1] > 0).long()
    return TensorDataset(inputs, targets)


def loader(
    samples: int = 64,
    batch_size: int = 16,
    *,
    features: int = 2,
    classes: int = 3,
    seed: int = 0,
    shuffle: bool = False,
) -> DataLoader:
    dataset = classification_data(samples, features, classes, seed)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def image_loader(samples: int = 32, batch_size: int = 8, seed: int = 0) -> DataLoader:
    generator = torch.Generator().manual_seed(seed)
    images = torch.randn(samples, 1, 8, 8, generator=generator)
    targets = torch.randint(0, 3, (samples,), generator=generator)
    return DataLoader(TensorDataset(images, targets), batch_size=batch_size)


# ----------------------------------------------------------------------------- processes
@dataclass
class Setup:
    """A complete, native PyTorch training setup plus the wrapped process."""

    model: nn.Module
    optimizer: torch.optim.Optimizer
    loss_fn: nn.Module
    train_loader: DataLoader
    val_loader: DataLoader | None
    process: Any  # hollywood_training.TrainingProcess
    calls: dict[str, list] = field(default_factory=lambda: {"train": [], "validation": []})


def make_setup(
    *,
    epochs: int = 3,
    model: nn.Module | None = None,
    hidden: int = 8,
    samples: int = 64,
    batch_size: int = 16,
    validation: bool = True,
    lr: float = 0.05,
    seed: int = 0,
    native_steps: bool = True,
    train_hook: Callable[[Any], None] | None = None,
    features: int = 2,
    **process_kwargs: Any,
) -> Setup:
    """A TinyNet (or ``model``) classifier with native train / validation steps.

    ``train_hook(batch)`` runs at the start of every training step (e.g. to record timing or
    to slow the step down); ``calls`` records every step that ran. With ``native_steps=False``
    no step is registered and the package's default supervised steps are used.
    """
    import hollywood_training as ht

    torch.manual_seed(seed)
    model = model if model is not None else TinyNet(hidden, inputs=features)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    train_loader = loader(samples, batch_size, features=features, seed=seed)
    val_loader = (
        loader(max(8, samples // 4), batch_size, features=features, seed=seed + 1)
        if validation
        else None
    )
    process = ht.TrainingProcess(
        model,
        optimizer,
        train_loader,
        loss_fn=loss_fn,
        val_data=val_loader,
        epochs=epochs,
        **process_kwargs,
    )
    setup = Setup(model, optimizer, loss_fn, train_loader, val_loader, process)
    if native_steps:

        @process.train_step
        def train_step(batch):
            if train_hook is not None:
                train_hook(batch)
            setup.calls["train"].append(1)
            inputs, targets = batch
            optimizer.zero_grad()
            loss = loss_fn(model(inputs), targets)
            loss.backward()
            optimizer.step()
            return loss

        if validation:

            @process.validation_step
            def validation_step(batch):
                setup.calls["validation"].append(1)
                inputs, targets = batch
                return loss_fn(model(inputs), targets)

    return setup


def weights_of(model: nn.Module) -> dict[str, torch.Tensor]:
    """Detached CPU copies of every parameter, by qualified name."""
    return {name: p.detach().cpu().clone() for name, p in model.named_parameters()}

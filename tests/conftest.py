"""Shared fixtures. GUI tests run headless (Qt "offscreen") unless a platform is chosen."""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

import hollywood_training as ht
from hollywood_training.sync import Envelope, SyncChannel


class SmallNet(nn.Module):
    def __init__(self, hidden: int = 6) -> None:
        super().__init__()
        self.fc1 = nn.Linear(2, hidden)
        self.act = nn.Tanh()
        self.fc2 = nn.Linear(hidden, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


def make_data(samples: int = 96, seed: int = 0) -> TensorDataset:
    generator = torch.Generator().manual_seed(seed)
    inputs = torch.randn(samples, 2, generator=generator)
    targets = (inputs[:, 0] * inputs[:, 1] > 0).long()
    return TensorDataset(inputs, targets)


@pytest.fixture
def make_process():
    """Factory for a small, fast TrainingProcess."""

    def factory(epochs: int = 3, *, validation: bool = True, **kwargs) -> ht.TrainingProcess:
        torch.manual_seed(0)
        model = kwargs.pop("model", None) or SmallNet()
        optimizer = kwargs.pop("optimizer", None) or torch.optim.SGD(model.parameters(), lr=0.1)
        data = make_data()
        return ht.TrainingProcess(
            model,
            optimizer,
            DataLoader(data, batch_size=16),
            val_data=DataLoader(data, batch_size=48) if validation else None,
            loss_fn=kwargs.pop("loss_fn", nn.CrossEntropyLoss()),
            epochs=epochs,
            **kwargs,
        )

    return factory


class RecordingSink:
    """Frontend stand-in: records envelopes, acknowledges synchronous ones immediately."""

    def __init__(self, channel: SyncChannel) -> None:
        self.channel = channel
        self.events: list = []
        channel.connect(self)

    def __call__(self, envelope: Envelope) -> None:
        self.events.append(envelope.event)
        if envelope.synchronous:
            self.channel.acknowledge(envelope.sequence)

    def of_type(self, kind: type) -> list:
        return [event for event in self.events if isinstance(event, kind)]

    def names(self) -> list[str]:
        return [type(event).__name__ for event in self.events]


@pytest.fixture
def recording_channel():
    channel = SyncChannel()
    return channel, RecordingSink(channel)

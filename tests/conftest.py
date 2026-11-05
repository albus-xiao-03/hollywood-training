"""Shared fixtures. GUI tests run headless (Qt "offscreen") unless a platform is chosen.

The block at the top keeps every temporary file of a test run inside the project's
``.temp`` folder, as CONSTRICTION.md requires: ``TMPDIR`` (``tempfile``, multiprocessing
sockets), pytest's ``tmp_path`` base directory, the XDG cache (fontconfig, Qt), oneDNN's
JIT profiling dumps, and Python bytecode caches. Child processes (the GUI window process,
demos, README code) inherit these settings. It runs before torch or Qt is imported.
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_TEMP_ROOT = _PROJECT_ROOT / ".temp"


def _inside_temp(value: str | None) -> bool:
    if not value:
        return False
    try:
        Path(value).resolve().relative_to(_TEMP_ROOT.resolve())
    except ValueError:
        return False
    return True


def _keep_in_temp(variable: str, folder: str) -> None:
    """Point ``variable`` into ``.temp/<folder>`` unless it already points into ``.temp``."""
    if not _inside_temp(os.environ.get(variable)):
        path = _TEMP_ROOT / folder
        path.mkdir(parents=True, exist_ok=True)
        os.environ[variable] = str(path)


_keep_in_temp("TMPDIR", "tmp")
_keep_in_temp("XDG_CACHE_HOME", "xdg-cache")
os.environ.setdefault("ONEDNN_JIT_PROFILE", "0")  # no /tmp/perf-<pid>.map files
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")  # children write no __pycache__
sys.dont_write_bytecode = True  # neither does this process from here on
tempfile.tempdir = None  # make tempfile re-read TMPDIR
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest  # noqa: E402
import torch  # noqa: E402
from torch import nn  # noqa: E402
from torch.utils.data import DataLoader, TensorDataset  # noqa: E402

import hollywood_training as ht  # noqa: E402
from hollywood_training.sync import Envelope, SyncChannel  # noqa: E402

MARKERS = (
    "spec(*ids): SPEC.md requirement(s) verified by the test, e.g. spec('R5', 'G1')",
    "gui: needs Qt widgets (headless with QT_QPA_PLATFORM=offscreen)",
    "e2e: starts real processes (the GUI window process, demos, README code)",
    "slow: takes more than a few seconds",
    "timing: asserts wall-clock bounds; sensitive to an overloaded machine",
    "benchmark: opt-in performance comparison, run with HT_RUN_BENCHMARKS=1",
    "gpu: needs a CUDA or MPS device (skipped otherwise)",
)


@pytest.hookimpl(tryfirst=True)
def pytest_configure(config: pytest.Config) -> None:
    # Runs before pytest's tmpdir plugin reads the option: tmp_path lives in .temp.
    if not config.option.basetemp:
        config.option.basetemp = str(_TEMP_ROOT / "pytest-basetemp")
    for line in MARKERS:
        config.addinivalue_line("markers", line)


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

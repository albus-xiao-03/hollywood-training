"""R6: training runs as fast as the synchronization allows - it never waits for anything
but the per-epoch hand-off (or an explicit user request), whatever the animations need."""

from __future__ import annotations

import statistics
import time

import numpy as np
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

import hollywood_training as ht
from hollywood_training import runner
from hollywood_training.sync import EpochCompleted, TrainingControl

from .helpers import ManagedRun, TinyNet, build_process, record_deliveries

pytestmark = pytest.mark.spec("R6")


def hand_off_gaps(bundle, deliveries, epochs: int) -> list[float]:
    """Seconds between the end of epoch e's hand-off and the first batch of epoch e+1."""
    ends = {d.epoch: d.end for d in deliveries if d.event == "EpochCompleted"}
    assert all(d.acknowledged for d in deliveries if d.event == "EpochCompleted")
    return [bundle.first_batch_time(epoch + 1) - ends[epoch] for epoch in range(1, epochs)]


def trivial_process(epochs: int) -> ht.TrainingProcess:
    torch.manual_seed(0)
    model = TinyNet()
    return ht.TrainingProcess(
        model,
        torch.optim.SGD(model.parameters(), lr=0.1),
        [(torch.randn(4, 2), torch.tensor([0, 1, 0, 1]))],
        epochs=epochs,
        train_step=lambda batch: 0.0,
    )


def test_there_is_no_artificial_delay_by_default(monkeypatch):
    created: list[TrainingControl] = []

    class Recorded(TrainingControl):
        def __init__(self, *args, **kwargs) -> None:
            super().__init__(*args, **kwargs)
            created.append(self)

    monkeypatch.setattr(runner, "TrainingControl", Recorded)
    ht.run(build_process(epochs=1).process, gui=False)
    assert len(created) == 1 and created[0].delay == 0.0
    assert TrainingControl(3).delay == 0.0


@pytest.mark.timing
def test_training_resumes_right_after_each_hand_off(monkeypatch):
    deliveries = record_deliveries(monkeypatch)
    bundle = build_process(epochs=20)
    ht.run(bundle.process, gui=False)
    gaps = hand_off_gaps(bundle, deliveries, 20)
    assert statistics.median(gaps) < 0.05 and max(gaps) < 0.5, gaps


@pytest.mark.timing
@pytest.mark.gui
def test_training_resumes_right_after_the_window_drew_the_epoch(qapp, monkeypatch):
    deliveries = record_deliveries(monkeypatch)
    bundle = build_process(epochs=15)
    ht.run(bundle.process, isolate_gui=False, close_on_finish=True)
    gaps = hand_off_gaps(bundle, deliveries, 15)
    assert statistics.median(gaps) < 0.05 and max(gaps) < 0.5, gaps


@pytest.mark.timing
def test_epochs_have_no_minimum_duration():
    started = time.monotonic()
    result = ht.run(trivial_process(300), gui=False, log_every=1000)
    elapsed = time.monotonic() - started
    assert result.epochs_completed == 300
    assert elapsed < 6.0, f"300 trivial epochs took {elapsed:.2f} s (> 20 ms each)"


@pytest.mark.timing
@pytest.mark.gui
def test_the_window_costs_no_more_than_its_drawing(qapp):
    started = time.monotonic()
    result = ht.run(trivial_process(40), isolate_gui=False, close_on_finish=True)
    elapsed = time.monotonic() - started
    assert result.epochs_completed == 40
    assert elapsed < 40 * 0.25, f"40 trivial epochs with the window took {elapsed:.2f} s"


@pytest.mark.timing
@pytest.mark.parametrize("mode", ["console", pytest.param("window", marks=pytest.mark.gui)])
def test_batches_follow_each_other_without_throttling(request, mode):
    if mode == "window":
        request.getfixturevalue("qapp")
    bundle = build_process(epochs=2, samples=640, batch=8)  # 80 batches per epoch
    options = (
        {"gui": False} if mode == "console" else {"isolate_gui": False, "close_on_finish": True}
    )
    ht.run(bundle.process, **options)
    starts = [t for phase, t, _done in bundle.calls if phase == "train"]
    gaps = np.diff(starts)
    assert float(np.median(gaps)) < 0.01, f"median gap between batches {np.median(gaps):.4f} s"


@pytest.mark.timing
def test_a_requested_delay_slows_training_and_delay_0_restores_full_speed():
    bundle = build_process(epochs=100_000)  # stopped by the test

    def epoch_times(run: ManagedRun) -> list[tuple[int, float]]:
        acked = dict(run.frontend.acks)
        return [
            (envelope.event.record.epoch, acked[envelope.sequence])
            for envelope in list(run.frontend.envelopes)
            if isinstance(envelope.event, EpochCompleted) and envelope.sequence in acked
        ]

    with ManagedRun(bundle.process) as run:
        run.wait_for(lambda: run.session.epoch >= 2, "two epochs")
        assert run.command("delay 150").ok
        slow_from = run.session.epoch + 2  # surely after the request was seen
        run.wait_for(lambda: run.session.epoch >= slow_from + 3, "slow epochs", timeout=20)
        assert run.command("delay 0").ok
        fast_from = run.session.epoch + 2
        run.wait_for(lambda: run.session.epoch >= fast_from + 5, "fast epochs", timeout=20)
    times = dict(epoch_times(run))  # after the run stopped: every acknowledgement recorded

    slow = [times[e + 1] - times[e] for e in range(slow_from, slow_from + 3)]
    fast = [times[e + 1] - times[e] for e in range(fast_from, fast_from + 5)]
    assert min(slow) >= 0.14, slow
    assert statistics.median(fast) < 0.1, fast


@pytest.mark.benchmark
@pytest.mark.timeout(900)
@pytest.mark.slow
@pytest.mark.e2e
def test_the_default_thread_policy_does_not_slow_down_cpu_heavy_training():
    """Opt-in (HT_RUN_BENCHMARKS=1): the window's default ``torch_threads="auto"`` must not
    make a CPU-heavy model slower than leaving PyTorch's threads alone (QA finding M3)."""

    def process() -> ht.TrainingProcess:
        torch.manual_seed(0)
        model = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(4), nn.Flatten(), nn.Linear(512, 10),
        )  # fmt: skip
        data = TensorDataset(torch.randn(1024, 1, 28, 28), torch.randint(0, 10, (1024,)))
        return ht.TrainingProcess(
            model,
            torch.optim.Adam(model.parameters(), lr=1e-3),
            DataLoader(data, batch_size=128),
            loss_fn=nn.CrossEntropyLoss(),
            epochs=6,
        )

    def median_epoch(setting) -> float:
        result = ht.run(process(), close_on_finish=True, torch_threads=setting)
        return statistics.median(record.duration for record in result.history[1:])

    default, untouched = median_epoch("auto"), median_epoch(None)
    assert default <= untouched * 1.10, f"auto {default:.3f} s vs None {untouched:.3f} s"

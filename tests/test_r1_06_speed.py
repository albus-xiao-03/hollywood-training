"""R1 requirement 6 - "The training process **MUST** run as fast as synchronization allows.
**DO NOT** consider the readability of the GUI animations."

"As fast as synchronisation allows" means: the only waiting is the per-epoch hand-off.
No delay by default, no pacing for the eye, no waiting for animations, no waiting per batch.

How the thresholds were chosen (they were not tuned on a measurement): an epoch's hand-off
is followed by bookkeeping, one round of the event loop (in-process) and the start of the
next ``DataLoader`` pass - a few milliseconds of work. The limits below are one to two
orders of magnitude above that, yet below any deliberate pacing a "readable" animation would
need (a fraction of a second per epoch), so they catch artificial waiting without being
sensitive to a busy machine. Medians are used where one slow moment must not decide.

* no delay by default; the user's ``delay`` command slows down and ``delay 0`` restores;
* the next epoch starts right after the hand-off: headless, in-process window, window process;
* the window acknowledges as soon as it has drawn (it does not wait for animations);
* the GUI costs only the synchronisation per epoch, in-process and isolated;
* batches never wait for the frontend (progress updates are notifications).
"""

from __future__ import annotations

import itertools
import statistics
import time

import pytest
from htspec.headless import HeadlessRun
from htspec.models import make_setup
from htspec.probe_log import LINGER_CLOSE

import hollywood_training as ht
from hollywood_training.sync.control import TrainingControl
from hollywood_training.sync.events import BatchProgress, EpochCompleted

pytestmark = pytest.mark.req("R1.6")


class BatchClock:
    """A ``train_hook`` recording every training batch's start: ``(epoch, time)``."""

    def __init__(self, batches_per_epoch: int, clock=time.perf_counter) -> None:
        self.batches = batches_per_epoch
        self.clock = clock
        self.starts: list[tuple[int, float]] = []

    def __call__(self, _batch) -> None:
        self.starts.append((len(self.starts) // self.batches + 1, self.clock()))

    def first(self) -> dict[int, float]:
        found: dict[int, float] = {}
        for epoch, moment in self.starts:
            found.setdefault(epoch, moment)
        return found


def epoch_intervals(first: dict[int, float]) -> list[float]:
    return [first[b] - first[a] for a, b in itertools.pairwise(sorted(first))]


# ------------------------------------------------------------------------ delay
def test_there_is_no_delay_by_default():
    assert TrainingControl(5).delay == 0.0
    run = HeadlessRun(make_setup(epochs=1).process)
    run.run()
    assert run.session.delay == 0.0
    assert "delay       0 ms" in run.command("status").output


def test_delay_slows_down_and_delay_zero_restores_full_speed():
    clock = BatchClock(4)
    setup = make_setup(epochs=14, train_hook=clock)
    run = HeadlessRun(setup.process).start()
    run.wait_events(EpochCompleted, 2)
    assert run.command("delay 200").ok
    run.wait_events(EpochCompleted, 7)
    assert run.command("delay 0").ok
    run.join()
    first = clock.first()
    slow = [first[e + 1] - first[e] for e in (4, 5, 6)]  # surely after "delay 200"
    fast = [first[e + 1] - first[e] for e in (10, 11, 12)]  # surely after "delay 0"
    assert min(slow) >= 0.19
    assert statistics.median(fast) < 0.1


def test_a_run_without_a_window_is_not_paced():
    setup = make_setup(epochs=100, samples=4, batch_size=4, validation=False)
    started = time.perf_counter()
    result = ht.run(setup.process, gui=False)
    assert result.epochs_completed == 100
    assert time.perf_counter() - started < 15  # 150 ms per epoch would already be pacing


# --------------------------------------------------- right after the hand-off
def test_training_continues_right_after_the_hand_off():
    clock = BatchClock(4)
    run = HeadlessRun(make_setup(epochs=10, train_hook=clock).process)
    run.run()
    first = clock.first()
    ack_of = {sequence: moment for moment, sequence in run.acks}
    gaps = [
        first[r.event.record.epoch + 1] - ack_of[r.envelope.sequence]
        for r in run.received
        if isinstance(r.event, EpochCompleted) and r.event.record.epoch < 10
    ]
    assert statistics.median(gaps) < 0.02 and max(gaps) < 0.25, gaps


def test_training_continues_right_after_the_in_process_window_has_drawn(pilot):
    clock = BatchClock(4)
    setup = make_setup(epochs=10, train_hook=clock)
    pilot.on_finish(pilot.close)
    pilot.run(setup.process)
    first = clock.first()
    drawn = {}
    for render in pilot.renders:
        if render.epoch and render.network_epoch == render.epoch:
            drawn.setdefault(render.epoch, render.ended)
    gaps = [first[epoch + 1] - drawn[epoch] for epoch in range(1, 10)]
    assert statistics.median(gaps) < 0.1, gaps


@pytest.mark.subprocess
def test_training_continues_right_after_the_window_process_acknowledged(probe):
    clock = BatchClock(4, clock=time.monotonic)
    log = probe(script={"steps": [LINGER_CLOSE]})
    ht.run(make_setup(epochs=10, train_hook=clock).process)
    first = clock.first()
    acks = sorted(record["t"] for record in log.sends("ack"))
    gaps = []
    for epoch in range(1, 10):
        ack = max(t for t in acks if t < first[epoch + 1])  # the hand-off of this epoch
        gaps.append(first[epoch + 1] - ack)
    assert statistics.median(gaps) < 0.1, gaps


@pytest.mark.subprocess
def test_the_window_process_acknowledges_as_soon_as_it_has_drawn(probe):
    log = probe(script={"steps": [LINGER_CLOSE]})
    ht.run(make_setup(epochs=8).process)
    acks = sorted(record["t"] for record in log.sends("ack"))
    waits, durations = [], []
    for render in log.renders():
        ack = min((t for t in acks if t >= render["t1"]), default=None)
        assert ack is not None
        waits.append(ack - render["t1"])
        durations.append(render["t1"] - render["t0"])
    assert statistics.median(waits) < 0.02, waits  # no waiting for animations to finish
    assert statistics.median(durations) < 0.5, durations  # a repaint, not an animation


# ------------------------------------------------------- the cost of the GUI
def _median_epoch_time(**run_options) -> float:
    clock = BatchClock(4)
    setup = make_setup(epochs=20, train_hook=clock)
    ht.run(setup.process, **run_options)
    return statistics.median(epoch_intervals(clock.first()))


def _per_epoch(console: float, window: float) -> str:
    return f"per epoch: console {console * 1000:.0f} ms, window {window * 1000:.0f} ms"


def test_the_in_process_window_costs_only_the_synchronisation(qapp):
    console = _median_epoch_time(gui=False)
    window = _median_epoch_time(isolate_gui=False, close_on_finish=True, torch_threads=None)
    assert window - console < 0.25, _per_epoch(console, window)


@pytest.mark.subprocess
def test_the_window_process_costs_only_the_synchronisation():
    console = _median_epoch_time(gui=False)
    window = _median_epoch_time(close_on_finish=True, torch_threads=None)
    assert window - console < 0.25, _per_epoch(console, window)


# ------------------------------------------------------------------- batches
def test_batches_never_wait_for_the_frontend():
    """Even a frontend that needs 0.3 s per epoch leaves the batches inside an epoch alone:
    progress reports are notifications, only the epoch hand-off waits."""
    clock = BatchClock(4)
    setup = make_setup(epochs=4, train_hook=clock)
    run = HeadlessRun(
        setup.process, ack_delay=lambda e: 0.3 if isinstance(e, EpochCompleted) else 0.0
    )
    run.run()
    progress = [r.envelope for r in run.received if isinstance(r.event, BatchProgress)]
    assert progress and not any(envelope.synchronous for envelope in progress)
    within = []
    for (epoch_a, time_a), (epoch_b, time_b) in itertools.pairwise(clock.starts):
        if epoch_a == epoch_b:
            within.append(time_b - time_a)
    assert statistics.median(within) < 0.02 and max(within) < 0.2, within

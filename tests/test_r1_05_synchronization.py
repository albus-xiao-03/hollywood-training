"""R1 requirement 5 - "**ALL** the training update **MUST** be synchronized with the GUI
update. The GUI **MUST** be updated at **EVERY** `epoch`, and complete update **BEFORE** the
next `epoch`. Computational efficiency has **NO** priority in this project."

Checked at three levels, always against the time the next epoch's first batch starts:

* the hand-off itself (``SyncChannel``): a synchronous delivery blocks until the frontend
  acknowledges it; closing releases it; notifications never wait;
* the real manager with a recording frontend: every epoch is delivered, once, in order,
  synchronously, and the next epoch starts only after the acknowledgement - also with a slow
  frontend, many fast epochs, pause / step, a stop and a failure;
* the real windows: in-process (``isolate_gui=False``) and the default window process. For
  every epoch the network view, the loss chart and the status panel are repainted with that
  epoch - and the drawn weights are the model's real weights of that epoch - and the repaint
  has completed before the next epoch's first batch. Boundaries: one epoch, many fast
  epochs, pause / step / resume, extending a finished run, closing the window mid-training,
  a failing step.
"""

from __future__ import annotations

import threading
import time

import numpy as np
import pytest
from htspec.headless import HeadlessRun
from htspec.models import make_setup, weights_of
from htspec.probe_log import LINGER_CLOSE

import hollywood_training as ht
from hollywood_training.sync.channel import SyncChannel
from hollywood_training.sync.events import (
    EpochCompleted,
    LogMessage,
    SessionFinished,
    StateChanged,
    TrainingPaused,
    TrainingState,
)

pytestmark = pytest.mark.req("R1.5")


class FirstBatches:
    """A ``train_hook`` that notes when each epoch's first training batch starts."""

    def __init__(self, batches_per_epoch: int, clock=time.perf_counter) -> None:
        self.batches = batches_per_epoch
        self.clock = clock
        self.calls = 0
        self.times: dict[int, float] = {}

    def __call__(self, _batch) -> None:
        if self.calls % self.batches == 0:
            self.times.setdefault(self.calls // self.batches + 1, self.clock())
        self.calls += 1


def epoch_delay(seconds: float):
    """``ack_delay`` for HeadlessRun: a frontend that needs ``seconds`` to draw an epoch."""
    return lambda event: seconds if isinstance(event, EpochCompleted) else 0.0


def keep_weights(setup) -> dict[int, dict]:
    copies: dict[int, dict] = {}

    @setup.process.on_epoch_end
    def keep(record):
        copies[record.epoch] = {k: v.numpy() for k, v in weights_of(setup.model).items()}

    return copies


# ---------------------------------------------------------------- the hand-off
def test_a_synchronous_delivery_waits_for_the_acknowledgement():
    channel = SyncChannel()
    received = []
    channel.connect(received.append)
    outcome = {}

    def deliver() -> None:
        started = time.perf_counter()
        outcome["acknowledged"] = channel.deliver(LogMessage("sync"))
        outcome["waited"] = time.perf_counter() - started

    sender = threading.Thread(target=deliver)
    sender.start()
    time.sleep(0.3)
    assert sender.is_alive() and received, "the sender did not wait"
    channel.acknowledge(received[0].sequence)
    sender.join(5)
    assert outcome["acknowledged"] is True and outcome["waited"] >= 0.25


def test_closing_the_channel_releases_a_waiting_delivery():
    channel = SyncChannel()
    channel.connect(lambda envelope: None)
    outcome = {}

    def deliver() -> None:
        outcome["ok"] = channel.deliver(LogMessage("x"))

    sender = threading.Thread(target=deliver)
    sender.start()
    time.sleep(0.2)
    channel.close()
    sender.join(5)
    assert not sender.is_alive() and outcome["ok"] is False
    assert channel.deliver(LogMessage("after close")) is False


def test_notifications_never_wait():
    channel = SyncChannel()
    received = []
    channel.connect(received.append)
    started = time.perf_counter()
    assert channel.notify(LogMessage("no wait")) is True
    assert time.perf_counter() - started < 0.1
    assert received and received[0].synchronous is False


# ------------------------------------------------------------ the real manager
def test_every_epoch_is_delivered_once_in_order_and_waited_for():
    run = HeadlessRun(make_setup(epochs=6).process)
    run.run()
    epochs = [r.envelope for r in run.received if isinstance(r.event, EpochCompleted)]
    assert [e.event.record.epoch for e in epochs] == [1, 2, 3, 4, 5, 6]
    assert all(e.synchronous for e in epochs)
    assert all(e.event.weights.epoch == e.event.record.epoch for e in epochs)
    acknowledged = {sequence for _time, sequence in run.acks}
    assert {e.sequence for e in epochs} <= acknowledged


@pytest.mark.parametrize("ack_delay", [0.0, 0.1])
def test_the_next_epoch_starts_only_after_the_acknowledgement(ack_delay):
    setup_batches = 4
    first = FirstBatches(setup_batches)
    setup = make_setup(epochs=5, train_hook=first)
    assert len(setup.train_loader) == setup_batches
    run = HeadlessRun(setup.process, ack_delay=epoch_delay(ack_delay))
    run.run()
    ack_time = {sequence: t for t, sequence in run.acks}
    for received in run.received:
        if isinstance(received.event, EpochCompleted) and received.event.record.epoch < 5:
            epoch = received.event.record.epoch
            assert first.times[epoch + 1] > ack_time[received.envelope.sequence], epoch


def test_a_slow_frontend_slows_training_down_but_misses_no_epoch():
    first = FirstBatches(4)
    setup = make_setup(epochs=6, train_hook=first)
    started = time.perf_counter()
    run = HeadlessRun(setup.process, ack_delay=epoch_delay(0.15))
    result = run.run()
    assert time.perf_counter() - started >= 6 * 0.15
    assert [e.record.epoch for e in run.events(EpochCompleted)] == list(range(1, 7))
    assert result.epochs_completed == 6


def test_many_fast_epochs_are_each_delivered():
    setup = make_setup(epochs=200, samples=4, batch_size=4, validation=False)
    run = HeadlessRun(setup.process)
    run.run(timeout=120)
    assert [e.record.epoch for e in run.events(EpochCompleted)] == list(range(1, 201))


def test_pause_and_step_skip_no_epoch():
    setup = make_setup(epochs=6, train_hook=lambda _b: time.sleep(0.01))  # ~40 ms epochs
    run = HeadlessRun(setup.process).start()
    run.wait_events(EpochCompleted, 2)
    assert run.command("pause").ok
    run.wait_events(TrainingPaused, 1)
    assert run.command("step 2").ok
    run.wait_events(TrainingPaused, 2)
    assert run.command("resume").ok
    result = run.join()
    assert [e.record.epoch for e in run.events(EpochCompleted)] == list(range(1, 7))
    assert result.epochs_completed == 6
    paused = [r.envelope for r in run.received if isinstance(r.event, TrainingPaused)]
    assert len(paused) == 2 and all(envelope.synchronous for envelope in paused)


def test_a_stopped_run_delivers_no_partial_epoch_but_its_end():
    setup = make_setup(epochs=100, samples=320, train_hook=lambda _b: time.sleep(0.003))
    run = HeadlessRun(setup.process).start()
    run.wait_events(EpochCompleted, 2)
    run.command("stop")
    result = run.join()
    delivered = run.events(EpochCompleted)
    assert result.state is TrainingState.STOPPED
    assert len(delivered) == result.epochs_completed < 100
    finished = [r.envelope for r in run.received if isinstance(r.event, SessionFinished)]
    assert len(finished) == 1 and finished[0].synchronous
    assert finished[0].event.epochs_completed == result.epochs_completed


def test_a_failing_step_still_delivers_the_end_of_the_session():
    setup = make_setup(epochs=5, native_steps=False)
    calls = {"n": 0}

    @setup.process.train_step
    def train(batch):
        calls["n"] += 1
        if calls["n"] > 6:  # in the second epoch
            raise RuntimeError("boom")
        setup.optimizer.zero_grad()
        loss = setup.loss_fn(setup.model(batch[0]), batch[1])
        loss.backward()
        setup.optimizer.step()
        return loss

    run = HeadlessRun(setup.process)
    result = run.run()
    assert result.state is TrainingState.FAILED
    finished = run.events(SessionFinished)
    assert len(finished) == 1 and finished[0].state is TrainingState.FAILED
    assert len(run.events(EpochCompleted)) == 1


# ------------------------------------------------------------ in-process window
def renders_by_epoch(renders) -> dict:
    """The repaint that showed each epoch first (network view and chart at that epoch)."""
    shown: dict = {}
    for render in renders:
        if render.network_epoch == render.chart_epoch == render.epoch and render.epoch:
            shown.setdefault(render.epoch, render)
    return shown


def test_every_epoch_is_drawn_completely_before_the_next_epoch_starts(pilot):
    first = FirstBatches(4)
    setup = make_setup(epochs=6, train_hook=first)
    pilot.on_finish(pilot.close)
    result = pilot.run(setup.process)
    assert result.epochs_completed == 6
    shown = renders_by_epoch(pilot.renders)
    assert sorted(shown) == list(range(1, 7))
    for epoch in range(1, 6):
        assert shown[epoch].ended < first.times[epoch + 1], epoch


@pytest.mark.req("R1.4", "R1.G1")
def test_the_drawn_weights_are_the_real_weights_of_each_epoch(pilot):
    setup = make_setup(epochs=4)
    copies = keep_weights(setup)
    pilot.on_finish(pilot.close)
    pilot.run(setup.process)
    shown = renders_by_epoch(pilot.renders)
    for epoch, render in shown.items():
        for name in ("fc1", "fc2"):
            layer = render.weights.layer(name)
            np.testing.assert_array_equal(layer.matrix, copies[epoch][f"{name}.weight"])
            np.testing.assert_array_equal(layer.bias, copies[epoch][f"{name}.bias"])


def test_the_status_panel_shows_every_epoch(pilot, paint_capture):
    pilot.on_finish(pilot.close)
    pilot.run(make_setup(epochs=5).process)
    counters = set(paint_capture.strings("StatusPanel"))
    assert {f"{n}/5" for n in range(1, 6)} <= counters


@pytest.mark.parametrize("epochs", [1, 2])
def test_the_smallest_runs_are_drawn_too(pilot, epochs):
    pilot.on_finish(pilot.close)
    result = pilot.run(make_setup(epochs=epochs).process)
    assert result.epochs_completed == epochs
    assert sorted(renders_by_epoch(pilot.renders)) == list(range(1, epochs + 1))


@pytest.mark.slow
def test_many_fast_epochs_are_each_drawn(pilot):
    setup = make_setup(epochs=150, samples=4, batch_size=4, validation=False)
    pilot.on_finish(pilot.close)
    pilot.run(setup.process)
    assert sorted(renders_by_epoch(pilot.renders)) == list(range(1, 151))


def test_pause_step_and_resume_skip_no_drawing(pilot):
    pauses = []

    def paused(window) -> None:
        pauses.append(1)
        pilot.after(0.2, pilot.line("step 2" if len(pauses) == 1 else "resume"))(window)

    pilot.on_epoch(2, pilot.line("pause"))
    pilot.on_paused(paused, once=False)  # once per pause that takes effect
    pilot.on_finish(pilot.after(0.2, pilot.close))
    result = pilot.run(make_setup(epochs=6).process)
    assert result.epochs_completed == 6
    assert sorted(renders_by_epoch(pilot.renders)) == list(range(1, 7))
    paused_renders = [r for r in pilot.renders if r.state is TrainingState.PAUSED]
    assert len(paused_renders) >= 2  # TrainingPaused is drawn synchronously too


def test_extending_a_finished_run_draws_the_new_epochs(pilot):
    def finished_at(epoch: int):
        def test(event, _window) -> bool:
            return (
                isinstance(event, StateChanged)
                and event.state is TrainingState.FINISHED
                and event.epoch == epoch
            )

        return test

    pilot.on(finished_at(2), pilot.line("epochs 4"))
    pilot.on(finished_at(4), pilot.close)
    result = pilot.run(make_setup(epochs=2).process)
    assert result.epochs_completed == 4
    assert sorted(renders_by_epoch(pilot.renders)) == [1, 2, 3, 4]


def test_closing_the_window_mid_training_stops_training_promptly(pilot):
    pilot.on_epoch(2, pilot.close)
    started = time.perf_counter()
    result = pilot.run(make_setup(epochs=1000).process)
    assert time.perf_counter() - started < 20
    assert result.state is TrainingState.STOPPED
    assert 2 <= result.epochs_completed < 1000


def test_a_failing_step_is_drawn_as_the_end_of_the_session(pilot):
    setup = make_setup(epochs=5, native_steps=False)
    calls = {"n": 0}

    @setup.process.train_step
    def train(batch):
        calls["n"] += 1
        if calls["n"] > 6:
            raise RuntimeError("boom in the second epoch")
        setup.optimizer.zero_grad()
        loss = setup.loss_fn(setup.model(batch[0]), batch[1])
        loss.backward()
        setup.optimizer.step()
        return loss

    pilot.on_finish(pilot.after(0.2, pilot.close))
    with pytest.raises(ht.TrainingFailedError):
        pilot.run(setup.process)
    assert pilot.of_type(SessionFinished)[0].state is TrainingState.FAILED
    assert pilot.renders[-1].state is TrainingState.FAILED  # drawn, not just received
    assert "FAILED" in pilot.final_log


# --------------------------------------------------------- the window process
def _isolated_renders(log) -> dict:
    shown: dict = {}
    for render in log.renders():
        if render["network_epoch"] == render["chart_epoch"] == render["epoch"] and render["epoch"]:
            shown.setdefault(render["epoch"], render)
    return shown


@pytest.mark.subprocess
def test_the_window_process_draws_every_epoch_before_the_next_epoch_starts(probe):
    first = FirstBatches(4, clock=time.monotonic)  # monotonic: comparable across processes
    setup = make_setup(epochs=6, train_hook=first)
    log = probe(script={"steps": [LINGER_CLOSE]})
    result = ht.run(setup.process)
    assert result.epochs_completed == 6
    shown = _isolated_renders(log)
    assert sorted(shown) == list(range(1, 7))
    acks = sorted(record["t"] for record in log.sends("ack"))
    for epoch in range(1, 6):
        render = shown[epoch]
        ack = next(t for t in acks if t >= render["t1"])  # the acknowledgement follows the drawing
        assert render["t1"] <= ack < first.times[epoch + 1], epoch
    assert log.errors() == []


@pytest.mark.subprocess
@pytest.mark.req("R1.4", "R1.G1")
def test_the_window_process_draws_the_real_weights_of_each_epoch(probe):
    setup = make_setup(epochs=4)
    copies = keep_weights(setup)
    log = probe(script={"steps": [LINGER_CLOSE]}, weights=True)
    ht.run(setup.process)
    shown = _isolated_renders(log)
    assert sorted(shown) == [1, 2, 3, 4]
    for epoch, render in shown.items():
        for name in ("fc1", "fc2"):
            drawn = np.array(render["weights"][name]["matrix"], dtype=np.float32)
            np.testing.assert_array_equal(drawn, copies[epoch][f"{name}.weight"])
            bias = np.array(render["weights"][name]["bias"], dtype=np.float32)
            np.testing.assert_array_equal(bias, copies[epoch][f"{name}.bias"])


@pytest.mark.subprocess
def test_closing_the_window_process_mid_training_stops_training_promptly(probe):
    probe(script={"steps": [{"on": "epoch", "value": 2, "do": "close"}]})
    started = time.perf_counter()
    result = ht.run(make_setup(epochs=1000).process)
    assert time.perf_counter() - started < 30
    assert result.state is TrainingState.STOPPED
    assert 2 <= result.epochs_completed < 1000


@pytest.mark.subprocess
def test_a_failing_step_is_drawn_by_the_window_process(probe):
    setup = make_setup(epochs=5, native_steps=False)
    calls = {"n": 0}

    @setup.process.train_step
    def train(batch):
        calls["n"] += 1
        if calls["n"] > 6:
            raise RuntimeError("boom in the second epoch")
        setup.optimizer.zero_grad()
        loss = setup.loss_fn(setup.model(batch[0]), batch[1])
        loss.backward()
        setup.optimizer.step()
        return loss

    log = probe(script={"steps": [{"on": "finished", "do": "close", "delay_ms": 300}]})
    with pytest.raises(ht.TrainingFailedError):
        ht.run(setup.process)
    finished = [e for e in log.events("SessionFinished")]
    assert finished and finished[0]["state"] == "failed"
    assert any(r["state"] == "failed" for r in log.renders())


@pytest.mark.subprocess
def test_pausing_the_window_process_skips_no_epoch(probe):
    steps = [
        {"on": "epoch", "value": 2, "do": "keys", "keys": "pause<Return>"},
        {"on": "state", "value": "paused", "do": "keys", "keys": "resume<Return>", "delay_ms": 400},
        LINGER_CLOSE,
    ]
    log = probe(script={"steps": steps})
    # ~30 ms per epoch: the run is still going when the typed "pause" arrives
    result = ht.run(make_setup(epochs=30, train_hook=lambda _b: time.sleep(0.005)).process)
    assert result.epochs_completed == 30
    assert sorted(_isolated_renders(log)) == list(range(1, 31))
    assert any(r["state"] == "paused" for r in log.renders())

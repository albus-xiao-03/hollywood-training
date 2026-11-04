"""The hand-off channel, runtime control and the frontend session state."""

from __future__ import annotations

import threading
import time

import pytest

from hollywood_training.sync import (
    EpochCompleted,
    EpochRecord,
    LogMessage,
    SessionStarted,
    SessionState,
    StateChanged,
    SyncChannel,
    TrainingControl,
    TrainingState,
    is_synchronous,
)
from hollywood_training.tracking import NetworkTracker
from hollywood_training.training.describe import build_session_info

from .conftest import SmallNet


# ------------------------------------------------------------------------ channel
def test_deliver_blocks_until_acknowledged():
    channel = SyncChannel()
    received = []
    channel.connect(received.append)
    finished = threading.Event()

    def sender():
        assert channel.deliver(LogMessage("hello")) is True
        finished.set()

    thread = threading.Thread(target=sender)
    thread.start()
    time.sleep(0.05)
    assert received and not finished.is_set()  # delivered, still waiting
    channel.acknowledge(received[0].sequence)
    assert finished.wait(2)
    thread.join()


def test_notify_does_not_wait_and_sequences_increase():
    channel = SyncChannel()
    received = []
    channel.connect(received.append)
    assert channel.notify(LogMessage("a")) and channel.notify(LogMessage("b"))
    assert [envelope.sequence for envelope in received] == [1, 2]
    assert not any(envelope.synchronous for envelope in received)


def test_close_releases_a_waiting_sender_and_drops_later_events():
    channel = SyncChannel()
    channel.connect(lambda envelope: None)  # never acknowledges
    result = []
    thread = threading.Thread(target=lambda: result.append(channel.deliver(LogMessage("x"))))
    thread.start()
    time.sleep(0.05)
    channel.close()
    thread.join(2)
    assert result == [False]
    assert channel.deliver(LogMessage("late")) is False
    assert channel.notify(LogMessage("late")) is False


def test_channel_without_frontend_is_an_error():
    with pytest.raises(RuntimeError):
        SyncChannel().notify(LogMessage("nobody listens"))


# ------------------------------------------------------------------------ control
def test_pause_resume_stop_and_revision():
    control = TrainingControl(5)
    revision = control.revision
    control.pause()
    assert control.paused and control.revision > revision
    control.resume()
    assert not control.paused
    control.pause()
    control.stop()
    assert control.stop_requested and not control.paused  # stop wakes a paused session


def test_step_pauses_after_the_requested_epochs():
    control = TrainingControl(10, start_paused=True)
    control.epoch_completed(2)
    control.step(3)
    assert not control.paused and control.stepping_until == 5
    control.epoch_completed(4)
    assert not control.paused
    control.epoch_completed(5)
    assert control.paused and control.stepping_until is None


def test_settings_validation():
    control = TrainingControl(3)
    with pytest.raises(ValueError):
        TrainingControl(0)
    control.epoch_completed(2)
    with pytest.raises(ValueError):
        control.set_total_epochs(1)
    control.set_total_epochs(8)
    assert control.total_epochs == 8
    with pytest.raises(ValueError):
        control.set_delay(-1)
    with pytest.raises(ValueError):
        control.step(0)


def test_actions_queue_and_close():
    control = TrainingControl(3)
    assert control.submit(lambda target: "one", "first")
    assert [description for _, description in control.take_actions()] == ["first"]
    assert control.take_actions() == []
    control.submit(lambda target: None, "never ran")
    assert [d for _, d in control.close()] == ["never ran"]
    assert control.closed and not control.submit(lambda target: None, "too late")


def test_sleep_returns_early_on_stop():
    control = TrainingControl(3)
    threading.Timer(0.05, control.stop).start()
    started = time.monotonic()
    control.sleep(5)
    assert time.monotonic() - started < 1


# ---------------------------------------------------------------- session state
def _started(epochs: int = 4) -> SessionStarted:
    model = SmallNet()
    tracker = NetworkTracker(model)
    import torch

    class Process:  # the attributes build_session_info reads
        optimizer = torch.optim.SGD(model.parameters(), lr=0.5)
        train_data = (1, 2, 3)
        val_data = None
        loss_fn = None
        scheduler = None
        batch_size = 8
        normalization = None
        name = None

    Process.model = model
    info = build_session_info(
        Process(), tracker.architecture.describe(), total_epochs=epochs, neuron_limit=20
    )
    return SessionStarted(info, tracker.snapshot(0), TrainingState.RUNNING)


def _epoch(epoch: int, val: float, weights) -> EpochCompleted:
    record = EpochRecord(epoch, train_loss=val + 0.1, val_loss=val, learning_rate=0.5)
    return EpochCompleted(record, weights, 4, 0.5, 8, elapsed=epoch * 1.0)


def test_session_state_folds_events():
    state = SessionState()
    started = _started()
    state.apply(started)
    assert state.network_name == "SmallNet" and state.total_epochs == 4
    assert state.batch_size == 8 and state.total_batches == 3 and state.learning_rate == 0.5
    versions = (state.architecture_version, state.weights_version)

    state.apply(_epoch(1, 0.8, started.weights))
    state.apply(_epoch(2, 0.5, started.weights))
    state.apply(_epoch(3, 0.6, started.weights))
    assert state.epoch == 3 and len(state.history) == 3
    assert state.best_record().epoch == 2
    assert state.weights_version == versions[1] + 3
    assert state.epoch_progress() == pytest.approx(0.75)

    state.apply(StateChanged(TrainingState.PAUSED, 3, 6, 0.1, 0.0, 3.0, "paused"))
    assert state.state is TrainingState.PAUSED and state.total_epochs == 6
    frozen = state.elapsed_now()
    time.sleep(0.02)
    assert state.elapsed_now() == frozen  # timer stands still while paused

    state.apply(started)  # a new session resets data but never rewinds versions
    assert state.history == [] and state.architecture_version > versions[0]


def test_synchronous_event_classification():
    assert is_synchronous(_started())
    assert not is_synchronous(LogMessage("x"))

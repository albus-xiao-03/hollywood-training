"""R5: every training update is synchronized with the GUI - the GUI is updated at EVERY
epoch and completes that update BEFORE the next epoch starts."""

from __future__ import annotations

import pickle
import queue
import threading
import time

import numpy as np
import pytest
import torch

import hollywood_training as ht
from hollywood_training.monitor import UsageSample
from hollywood_training.sync import (
    BatchProgress,
    Envelope,
    EpochCompleted,
    LogLevel,
    LogMessage,
    Phase,
    SessionFinished,
    SessionStarted,
    StateChanged,
    SyncChannel,
    TrainingPaused,
    TrainingState,
    is_synchronous,
)
from hollywood_training.training import TrainingManager

from .helpers import (
    ManagedRun,
    active_window,
    build_process,
    epoch_completed,
    session_started,
    type_command,
)

pytestmark = pytest.mark.spec("R5")


# ------------------------------------------------------------------ the hand-off
def test_delivery_blocks_until_the_frontend_acknowledges():
    channel = SyncChannel()
    inbox: queue.Queue = queue.Queue()
    channel.connect(inbox.put)
    returned = threading.Event()
    outcome: list[bool] = []

    def training_thread() -> None:
        outcome.append(channel.deliver(LogMessage("epoch done")))
        returned.set()

    threading.Thread(target=training_thread, daemon=True).start()
    envelope = inbox.get(timeout=2)
    assert envelope.synchronous
    assert not returned.wait(0.3), "deliver() returned before the frontend acknowledged"
    channel.acknowledge(envelope.sequence)
    assert returned.wait(2) and outcome == [True]


def test_a_frontend_that_acknowledges_inside_its_sink_does_not_deadlock():
    channel = SyncChannel()
    channel.connect(lambda envelope: channel.acknowledge(envelope.sequence))
    assert channel.deliver(LogMessage("drawn immediately")) is True


def test_events_are_numbered_and_arrive_in_order():
    channel = SyncChannel()
    seen = []

    def sink(envelope: Envelope) -> None:
        seen.append((envelope.sequence, envelope.synchronous, envelope.event.text))
        if envelope.synchronous:
            channel.acknowledge(envelope.sequence)

    channel.connect(sink)
    assert channel.notify(LogMessage("a"))
    assert channel.deliver(LogMessage("b"))
    assert channel.notify(LogMessage("c"))
    assert seen == [(1, False, "a"), (2, True, "b"), (3, False, "c")]


def test_an_unrelated_acknowledgement_does_not_release_a_pending_event():
    channel = SyncChannel()
    inbox: queue.Queue = queue.Queue()
    channel.connect(inbox.put)
    returned = threading.Event()
    threading.Thread(
        target=lambda: (channel.deliver(LogMessage("x")), returned.set()), daemon=True
    ).start()
    envelope = inbox.get(timeout=2)
    channel.acknowledge(envelope.sequence + 100)
    assert not returned.wait(0.3)
    channel.acknowledge(envelope.sequence)
    assert returned.wait(2)


def test_closing_the_channel_releases_a_waiting_sender():
    channel = SyncChannel()
    channel.connect(lambda envelope: None)  # never acknowledges: the window vanished
    outcome: list[bool] = []
    thread = threading.Thread(target=lambda: outcome.append(channel.deliver(LogMessage("x"))))
    thread.start()
    time.sleep(0.1)
    channel.close()
    thread.join(2)
    assert outcome == [False] and channel.closed
    assert channel.deliver(LogMessage("late")) is False
    assert channel.notify(LogMessage("late")) is False


def test_a_channel_without_a_frontend_refuses_to_send():
    with pytest.raises(RuntimeError):
        SyncChannel().deliver(LogMessage("nobody is listening"))


def test_epoch_updates_are_synchronous_events():
    bundle = build_process(epochs=1)
    tracker, started = session_started(bundle.process)
    weights = tracker.snapshot(0)
    assert is_synchronous(epoch_completed(tracker, 1, total=1, train_loss=0.5))
    # documented contract (wiki Sync-Events): start, pause and finish are drawn too
    assert is_synchronous(started)
    assert is_synchronous(TrainingPaused(1, Phase.TRAIN, 2, weights, 0.1))
    assert is_synchronous(SessionFinished(TrainingState.FINISHED, 1, 0.1, weights, "done"))
    for event in (
        LogMessage("x"),
        StateChanged(TrainingState.RUNNING, 0, 1, 0.1, 0.0, 0.0),
        BatchProgress(1, Phase.TRAIN, 1, 2, 0.5, 16, 0.0),
    ):
        assert not is_synchronous(event)


def test_every_event_can_cross_to_the_window_process():
    """The default window runs in another process: events travel pickled."""
    bundle = build_process(epochs=2)
    tracker, started = session_started(bundle.process)
    events = [
        started,
        epoch_completed(
            tracker, 1, total=2, train_loss=0.5, val_loss=0.4, train_metrics={"acc": 0.7}
        ),
        TrainingPaused(2, Phase.VALIDATION, 1, tracker.snapshot(1, baseline=False), 1.5),
        StateChanged(TrainingState.PAUSED, 1, 2, 0.1, 0.25, 1.5, "paused"),
        BatchProgress(2, Phase.TRAIN, 3, 6, 0.42, 16, 1.7),
        LogMessage("careful", LogLevel.WARNING),
        SessionFinished(TrainingState.FINISHED, 2, 2.0, tracker.snapshot(2), "completed"),
    ]
    for sequence, event in enumerate(events, start=1):
        copy = pickle.loads(pickle.dumps(Envelope(sequence, event, is_synchronous(event))))
        assert type(copy.event) is type(event) and copy.sequence == sequence
        original_weights = getattr(event, "weights", None)
        if original_weights is not None:
            for mine, theirs in zip(
                original_weights.layers, copy.event.weights.layers, strict=True
            ):
                np.testing.assert_array_equal(mine.matrix, theirs.matrix)
                np.testing.assert_array_equal(mine.out_index, theirs.out_index)
    usage = UsageSample(1.0, 12.5, 30.0, 8, 4 * 2**30, 16 * 2**30, 300 * 2**20)
    assert pickle.loads(pickle.dumps(usage)) == usage


# ----------------------------------------------------------- the managed process
def test_every_epoch_reaches_the_frontend_synchronously_and_in_order():
    bundle = build_process(epochs=5)
    with ManagedRun(bundle.process) as run:
        result = run.join()
    names = run.frontend.names()
    assert names[0] == "SessionStarted" and names[-1] == "SessionFinished"
    epochs = [e for e in run.frontend.envelopes if isinstance(e.event, EpochCompleted)]
    assert [e.event.record.epoch for e in epochs] == [1, 2, 3, 4, 5]
    assert all(envelope.synchronous for envelope in epochs)
    acknowledged = {sequence for sequence, _time in run.frontend.acks}
    assert {envelope.sequence for envelope in epochs} <= acknowledged
    assert [record.epoch for record in result.history] == [1, 2, 3, 4, 5]
    assert run.session.history == list(result.history)  # the frontend saw every update


def test_the_next_epoch_starts_only_after_the_frontend_finished_its_update():
    bundle = build_process(epochs=4)
    channel = SyncChannel()
    inbox: queue.Queue = queue.Queue()
    channel.connect(inbox.put)
    drawn: dict[int, tuple[float, bool]] = {}

    def frontend() -> None:  # a GUI thread that needs 50 ms to draw an epoch
        while True:
            envelope = inbox.get(timeout=30)
            event = envelope.event
            if isinstance(event, EpochCompleted):
                before = bundle.model.fc1.weight.detach().clone()
                time.sleep(0.05)
                unchanged = torch.equal(before, bundle.model.fc1.weight.detach())
                drawn[event.record.epoch] = (time.monotonic(), unchanged)
            if envelope.synchronous:
                channel.acknowledge(envelope.sequence)
            if isinstance(event, SessionFinished):
                return

    thread = threading.Thread(target=frontend, daemon=True)
    thread.start()
    result = TrainingManager(bundle.process, channel=channel).run()
    thread.join(10)
    assert result.epochs_completed == 4 and sorted(drawn) == [1, 2, 3, 4]
    for epoch in (1, 2, 3):
        finished_drawing, unchanged = drawn[epoch]
        assert unchanged, f"the model changed while epoch {epoch} was being drawn"
        assert bundle.first_batch_time(epoch + 1) > finished_drawing


def test_a_slow_frontend_never_causes_skipped_epochs():
    bundle = build_process(epochs=8)
    started = time.monotonic()
    with ManagedRun(bundle.process, ack_delay=0.03) as run:
        result = run.join()
    elapsed = time.monotonic() - started
    assert [e.record.epoch for e in run.frontend.of_type(EpochCompleted)] == list(range(1, 9))
    assert result.epochs_completed == 8
    assert elapsed >= 8 * 0.03  # training waited for every drawing


def test_the_frontend_gets_the_real_weights_of_each_epoch():
    bundle = build_process(epochs=3)
    with ManagedRun(bundle.process) as run:
        run.join()
    for event in run.frontend.of_type(EpochCompleted):
        real = bundle.weights[event.record.epoch]
        np.testing.assert_array_equal(event.weights.layer("fc1").matrix, real["fc1.weight"].numpy())
        np.testing.assert_array_equal(event.weights.layer("fc2").matrix, real["fc2.weight"].numpy())


def test_pausing_and_finishing_are_handed_over_synchronously():
    bundle = build_process(epochs=2)
    with ManagedRun(bundle.process, start_paused=True) as run:
        run.wait_for(lambda: run.frontend.of_type(TrainingPaused), "the initial pause")
        assert bundle.calls == []  # nothing trained while paused
        run.control.resume()
        run.join()
    for envelope in run.frontend.envelopes:
        if isinstance(envelope.event, (SessionStarted, TrainingPaused, SessionFinished)):
            assert envelope.synchronous


def test_a_frontend_that_vanishes_cannot_hang_the_training():
    channel = SyncChannel()
    channel.connect(lambda envelope: None)  # never draws, never acknowledges
    timer = threading.Timer(0.3, channel.close)
    timer.start()
    result = TrainingManager(build_process(epochs=50).process, channel=channel).run()
    timer.join()
    assert result.state is TrainingState.STOPPED and result.epochs_completed == 0


# ------------------------------------------------------------- in-process window
@pytest.mark.gui
@pytest.mark.spec("G1", "G2", "G3")
def test_the_window_has_drawn_each_epoch_before_the_next_epoch_starts(qapp, monkeypatch):
    from hollywood_training.gui.widgets import status_panel

    status_painted: list[int] = []
    original_paint_content = status_panel.StatusPanel.paint_content

    def paint_content(self, painter, rect):
        original_paint_content(self, painter, rect)
        status_painted.append(self.window().session.epoch)

    monkeypatch.setattr(status_panel.StatusPanel, "paint_content", paint_content)

    bundle = build_process(epochs=4, native_steps=False)
    model, optimizer, loss_fn = bundle.model, bundle.optimizer, bundle.loss_fn
    seen: dict[int, dict] = {}

    @bundle.process.train_step
    def train_step(batch):
        epoch = bundle.completed[0] + 1
        if epoch not in seen:  # first batch of this epoch
            window = active_window()
            seen[epoch] = {
                "network": window.network.last_painted_epoch,
                "chart": window.chart.last_painted_epoch,
                "session": window.session.epoch,
                "status": max(status_painted, default=None),
                "weights": {
                    layer.name: layer.matrix.copy() for layer in window.session.weights.layers
                },
            }
        inputs, targets = batch
        optimizer.zero_grad()
        loss = loss_fn(model(inputs), targets)
        loss.backward()
        optimizer.step()
        return loss

    result = ht.run(bundle.process, isolate_gui=False, close_on_finish=True)
    assert result.epochs_completed == 4 and sorted(seen) == [1, 2, 3, 4]
    assert seen[1]["network"] == 0 and seen[1]["session"] == 0  # initial weights shown
    for epoch in (2, 3, 4):
        previous = epoch - 1
        observed = seen[epoch]
        assert observed["network"] == previous, f"network view not drawn for epoch {previous}"
        assert observed["chart"] == previous, f"loss chart not drawn for epoch {previous}"
        assert observed["status"] == previous, f"status panel not drawn for epoch {previous}"
        assert observed["session"] == previous
        real = bundle.weights[previous]
        np.testing.assert_array_equal(observed["weights"]["fc1"], real["fc1.weight"].numpy())


# ------------------------------------------------- the window process, in-process
@pytest.mark.gui
def test_the_window_process_acknowledges_only_after_drawing(window_process):
    harness = window_process
    assert harness.receive() == ("ready",)
    bundle = build_process(epochs=3)
    tracker, started = session_started(bundle.process)
    harness.send("event", Envelope(1, started, True))
    assert harness.receive() == ("ack", 1)
    sequence = 2
    for epoch in (1, 2, 3):
        harness.send("event", Envelope(sequence, LogMessage(f"note {epoch}"), False))
        event = epoch_completed(tracker, epoch, total=3, train_loss=1 / epoch, val_loss=2 / epoch)
        harness.send("event", Envelope(sequence + 1, event, True))
        assert harness.receive() == ("ack", sequence + 1)  # async events get no ack
        sequence += 2
    acks = [(message[1], network, chart) for message, network, chart in harness.sent
            if message[0] == "ack"]  # fmt: skip
    assert acks == [(1, 0, None), (3, 1, 1), (5, 2, 2), (7, 3, 3)]
    assert "note 3" in harness.window.terminal.text()


@pytest.mark.gui
def test_the_window_process_acknowledges_a_pause_after_drawing_it(window_process):
    harness = window_process
    assert harness.receive() == ("ready",)
    bundle = build_process(epochs=3)
    tracker, started = session_started(bundle.process, state=TrainingState.PAUSED)
    harness.send("event", Envelope(1, started, True))
    assert harness.receive() == ("ack", 1)
    paused = TrainingPaused(1, Phase.TRAIN, 2, tracker.snapshot(0, baseline=False), 0.2)
    harness.send("event", Envelope(2, paused, True))
    assert harness.receive() == ("ack", 2)
    assert harness.window.session.state is TrainingState.PAUSED
    finished = SessionFinished(TrainingState.STOPPED, 0, 0.2, tracker.snapshot(0), "stopped")
    harness.send("event", Envelope(3, finished, True))
    assert harness.receive() == ("ack", 3)
    assert "STOPPED" in harness.window.terminal.text()


@pytest.mark.gui
@pytest.mark.spec("G5")
def test_lines_typed_in_the_window_process_go_to_the_training_process(window_process):
    harness = window_process
    assert harness.receive() == ("ready",)
    type_command(harness.window, "status")
    assert harness.receive() == ("command", "status")

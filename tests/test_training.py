"""TrainingProcess, steps, descriptions and the TrainingManager loop."""

from __future__ import annotations

import queue
import threading

import numpy as np
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader

import hollywood_training as ht
from hollywood_training.sync import (
    BatchProgress,
    EpochCompleted,
    LogLevel,
    LogMessage,
    SessionFinished,
    SessionStarted,
    StateChanged,
    SyncChannel,
    TrainingControl,
    TrainingPaused,
    TrainingState,
)
from hollywood_training.training import (
    ProcessError,
    StepOutputError,
    TrainingManager,
    describe_loss,
    describe_optimizer,
    read_step_output,
)
from hollywood_training.training.describe import (
    count_batches,
    describe_normalization,
    resolve_batch_size,
)

from .conftest import SmallNet, make_data


# --------------------------------------------------------------------- process
def test_process_validates_its_components():
    model = SmallNet()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    data = [(torch.zeros(1, 2), torch.zeros(1, dtype=torch.long))]
    with pytest.raises(TypeError):
        ht.TrainingProcess(object(), optimizer, data, epochs=1)
    with pytest.raises(TypeError):
        ht.TrainingProcess(model, object(), data, epochs=1)
    with pytest.raises(TypeError, match="one-shot iterator"):
        ht.TrainingProcess(model, optimizer, iter(data), epochs=1)
    with pytest.raises(ValueError):
        ht.TrainingProcess(model, optimizer, data, epochs=0)
    with pytest.raises(ValueError):
        ht.TrainingProcess(model, optimizer, data, epochs=1, batch_size=0)


def test_step_decorators_register_and_return_the_function(make_process):
    process = make_process()

    @process.train_step
    def train_step(batch):
        return 1.0

    @process.on_epoch_end
    def hook(record):
        pass

    assert train_step(None) == 1.0  # still a plain, callable function
    steps = process.resolve_steps()
    assert steps.train is train_step and steps.epoch_end == (hook,)
    assert steps.validation is not None  # default validation step from loss_fn


def test_resolve_steps_reports_missing_pieces():
    model = SmallNet()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    data = [(torch.zeros(1, 2), torch.zeros(1, dtype=torch.long))]
    with pytest.raises(ProcessError, match="train_step"):
        ht.TrainingProcess(model, optimizer, data, epochs=1).resolve_steps()
    with pytest.raises(ProcessError, match="validation_step"):
        ht.TrainingProcess(
            model, optimizer, data, epochs=1, train_step=lambda b: 0.0, val_data=data
        ).resolve_steps()
    with pytest.raises(ProcessError, match="no val_data"):
        ht.TrainingProcess(
            model, optimizer, data, epochs=1, train_step=lambda b: 0.0,
            validation_step=lambda b: 0.0,
        ).resolve_steps()  # fmt: skip


def test_read_step_output_variants():
    assert read_step_output(torch.tensor(0.5), step_name="s").loss == 0.5
    assert read_step_output(2, step_name="s").loss == 2.0
    assert read_step_output(np.float32(1.5), step_name="s").loss == 1.5
    result = read_step_output({"loss": torch.tensor(1.0), "acc": 0.75}, step_name="s")
    assert result.loss == 1.0 and result.metrics == {"acc": 0.75}
    for bad in (None, {"acc": 1.0}, torch.ones(3), "1.0", True):
        with pytest.raises(StepOutputError):
            read_step_output(bad, step_name="s")


# ----------------------------------------------------------------- descriptions
def test_descriptions_come_from_public_attributes():
    model = nn.Sequential(nn.Linear(2, 4), nn.BatchNorm1d(4), nn.ReLU(), nn.Linear(4, 1))
    sgd = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, nesterov=True)
    assert describe_optimizer(sgd) == ("SGD", "momentum=0.9, nesterov")
    adam = torch.optim.Adam(model.parameters(), weight_decay=1e-4)
    assert describe_optimizer(adam) == ("Adam", "betas=0.9/0.999, wd=0.0001")
    assert describe_loss(nn.MSELoss(reduction="sum")) == "MSELoss (sum)"
    assert describe_loss(torch.nn.functional.l1_loss) == "l1_loss"
    assert describe_loss(None) == "defined in step"
    architecture = ht.inspect_network(model).describe()
    assert describe_normalization(architecture, "z-score") == "data: z-score · layers: BatchNorm1d"
    loader = DataLoader(make_data(), batch_size=16)
    assert count_batches(loader) == 6 and count_batches(iter([])) is None
    assert resolve_batch_size(None, loader) == 16 and resolve_batch_size(4, loader) == 4


# ----------------------------------------------------------------------- manager
def test_manager_event_stream(make_process, recording_channel):
    channel, sink = recording_channel
    result = TrainingManager(make_process(epochs=3), channel=channel).run()
    assert result.state is TrainingState.FINISHED and result.epochs_completed == 3
    names = sink.names()
    assert names[0] == "SessionStarted" and names[-1] == "SessionFinished"
    epochs = sink.of_type(EpochCompleted)
    assert [event.record.epoch for event in epochs] == [1, 2, 3]
    assert all(event.weights.epoch == event.record.epoch for event in epochs)
    assert epochs[-1].record.val_loss is not None
    assert sink.of_type(SessionFinished)[0].reason == "completed 3 epochs"
    assert result.train_losses[-1] < result.train_losses[0]


def test_frontend_handles_each_epoch_while_training_is_blocked(make_process):
    """The core synchronisation guarantee, checked from a separate frontend thread."""
    process = make_process(epochs=4)
    channel = SyncChannel()
    inbox: queue.Queue = queue.Queue()
    channel.connect(inbox.put)
    observations = []

    def frontend():
        while True:
            envelope = inbox.get()
            event = envelope.event
            if isinstance(event, EpochCompleted):
                live = process.model.fc1.weight.detach().numpy().copy()
                threading.Event().wait(0.02)  # "render" slowly
                after = process.model.fc1.weight.detach().numpy().copy()
                drawn = event.weights.layer("fc1").matrix
                observations.append(np.allclose(live, after) and np.allclose(live, drawn))
            if envelope.synchronous:
                channel.acknowledge(envelope.sequence)
            if isinstance(event, SessionFinished):
                return

    thread = threading.Thread(target=frontend)
    thread.start()
    result = TrainingManager(process, channel=channel).run()
    thread.join(5)
    assert result.epochs_completed == 4
    assert observations == [True] * 4


def test_validation_runs_in_eval_mode_without_gradients(make_process, recording_channel):
    process = make_process(epochs=1)
    seen = {}

    @process.train_step
    def train_step(batch):
        seen["train"] = (process.model.training, torch.is_grad_enabled())
        return 1.0

    @process.validation_step
    def validation_step(batch):
        seen["validation"] = (process.model.training, torch.is_grad_enabled())
        return 1.0

    channel, _ = recording_channel
    TrainingManager(process, channel=channel).run()
    assert seen == {"train": (True, True), "validation": (False, False)}


def test_start_paused_then_resume(make_process, recording_channel):
    channel, sink = recording_channel
    control = TrainingControl(2, start_paused=True)
    timer = threading.Timer(0.1, control.resume)
    timer.start()
    result = TrainingManager(make_process(epochs=2), channel=channel, control=control).run()
    assert result.epochs_completed == 2
    assert sink.names().index("TrainingPaused") < sink.names().index("EpochCompleted")
    states = [event.state for event in sink.of_type(StateChanged)]
    assert TrainingState.PAUSED in states and TrainingState.RUNNING in states


def test_stop_mid_epoch(make_process, recording_channel):
    channel, sink = recording_channel
    process = make_process(epochs=50)
    control = TrainingControl(50)

    @process.train_step
    def train_step(batch):
        if len(sink.of_type(EpochCompleted)) == 2:
            control.stop()
        return 0.5

    result = TrainingManager(process, channel=channel, control=control).run()
    assert result.state is TrainingState.STOPPED and result.epochs_completed == 2
    finished = sink.of_type(SessionFinished)[0]
    assert finished.weights is not None and finished.reason == "stopped after 2 epochs"


def test_queued_action_changes_learning_rate(make_process, recording_channel):
    channel, sink = recording_channel
    process = make_process(epochs=2)
    control = TrainingControl(2)

    def set_lr(target):
        target.optimizer.param_groups[0]["lr"] = 0.01
        return "lr changed"

    control.submit(set_lr, "set lr")
    result = TrainingManager(process, channel=channel, control=control).run()
    assert result.history[0].learning_rate == 0.01
    messages = sink.of_type(LogMessage)
    assert any(m.text == "lr changed" and m.level is LogLevel.SUCCESS for m in messages)


def test_step_command_pauses_after_requested_epochs(make_process, recording_channel):
    channel, sink = recording_channel
    control = TrainingControl(6, start_paused=True)
    paused_at = []

    original = sink.__call__

    def watch(envelope):
        if isinstance(envelope.event, TrainingPaused):
            paused_at.append(envelope.event.epoch)
            follow_up = control.stop if len(paused_at) > 1 else (lambda: control.step(2))
            threading.Timer(0.02, follow_up).start()
        original(envelope)

    channel.connect(watch)
    result = TrainingManager(make_process(epochs=6), channel=channel, control=control).run()
    assert result.epochs_completed == 2 and paused_at == [0, 2]


def test_linger_waits_for_more_epochs(make_process, recording_channel):
    channel, sink = recording_channel
    control = TrainingControl(2)

    def on_state(envelope):
        event = envelope.event
        if isinstance(event, StateChanged) and event.state is TrainingState.FINISHED:
            if event.epoch == 2:  # first finish: ask for one more epoch
                control.set_total_epochs(3)
            elif event.epoch == 3:  # finished again: close the session
                control.stop()
        sink(envelope)

    channel.connect(on_state)
    manager = TrainingManager(make_process(epochs=2), channel=channel, control=control, linger=True)
    result = manager.run()
    assert result.state is TrainingState.FINISHED and result.epochs_completed == 3


def test_errors_in_user_code_fail_the_session(make_process, recording_channel):
    channel, sink = recording_channel
    process = make_process(epochs=3)

    @process.train_step
    def train_step(batch):
        raise ZeroDivisionError("boom")

    result = TrainingManager(process, channel=channel).run()
    assert result.state is TrainingState.FAILED and isinstance(result.error, ZeroDivisionError)
    errors = [m.text for m in sink.of_type(LogMessage) if m.level is LogLevel.ERROR]
    assert errors and "ZeroDivisionError: boom" in errors[0] and "Traceback" in errors[0]
    assert sink.of_type(SessionFinished)[0].error == "ZeroDivisionError: boom"


def test_empty_training_data_is_reported(recording_channel):
    model = SmallNet()
    process = ht.TrainingProcess(
        model, torch.optim.SGD(model.parameters(), lr=0.1), [], loss_fn=nn.MSELoss(), epochs=2
    )
    channel, _ = recording_channel
    result = TrainingManager(process, channel=channel).run()
    assert result.state is TrainingState.FAILED and "no batches" in str(result.error)


def test_reduce_on_plateau_receives_the_validation_loss(make_process, recording_channel):
    process = make_process(epochs=2)
    received = []

    class Plateau(torch.optim.lr_scheduler.ReduceLROnPlateau):
        def step(self, metrics, *args, **kwargs):
            received.append(metrics)
            return super().step(metrics, *args, **kwargs)

    process = make_process(epochs=2, scheduler=None)
    process._scheduler = Plateau(process.optimizer)
    channel, _ = recording_channel
    result = TrainingManager(process, channel=channel).run()
    assert received == pytest.approx(result.val_losses)


def test_refined_architecture_is_sent_with_the_first_epoch(recording_channel):
    class Reused(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.act = nn.ReLU()
            self.fc1 = nn.Linear(2, 4)
            self.fc2 = nn.Linear(4, 2)

        def forward(self, x):
            return self.fc2(self.act(self.fc1(x)))

    model = Reused()
    process = ht.TrainingProcess(
        model, torch.optim.SGD(model.parameters(), lr=0.1),
        DataLoader(make_data(), batch_size=32), loss_fn=nn.CrossEntropyLoss(), epochs=2,
    )  # fmt: skip
    channel, sink = recording_channel
    TrainingManager(process, channel=channel).run()
    started = sink.of_type(SessionStarted)[0].info.architecture
    assert started.input_operations  # static order put the ReLU before fc1
    first, second = sink.of_type(EpochCompleted)
    assert first.architecture is not None and second.architecture is None
    assert not first.architecture.input_operations
    assert first.architecture.layers[0].activation == "ReLU"


def test_progress_is_throttled(make_process, recording_channel):
    channel, sink = recording_channel
    TrainingManager(make_process(epochs=1), channel=channel, progress_interval=1e9).run()
    batches = [(e.phase.value, e.batch) for e in sink.of_type(BatchProgress)]
    assert batches == [("train", 1), ("train", 6), ("validation", 1), ("validation", 2)]

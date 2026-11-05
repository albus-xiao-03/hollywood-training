"""Failure scenarios across components: errors in user code, interrupted runs, closed
windows, invalid options, missing displays and extreme models. A managed run must end
cleanly, report the cause and never hang."""

from __future__ import annotations

import math
import sys
import threading

import pytest
import torch
from torch import nn

import hollywood_training as ht
from hollywood_training.sync import TrainingState as State

from .helpers import (
    active_window,
    build_process,
    type_command,
    window_process_children,
    xor_data,
)


class BoomError(Exception):
    pass


class FlakyData:
    """Re-iterable data whose second pass breaks after two batches."""

    def __init__(self) -> None:
        self.passes = 0
        self.batches = [tuple(xor_data(16, seed).tensors) for seed in range(4)]

    def __iter__(self):
        self.passes += 1
        for index, batch in enumerate(self.batches):
            if self.passes == 2 and index == 2:
                raise BoomError("data source broke")
            yield batch


class CountingScheduler:
    def __init__(self) -> None:
        self.calls = 0

    def step(self) -> None:
        self.calls += 1
        if self.calls == 2:
            raise BoomError("scheduler broke")


def failing_process(where: str) -> ht.TrainingProcess:
    bundle = build_process(epochs=3)
    process = bundle.process
    if where == "train_step":
        inner = process.resolve_steps().train

        def train_step(batch):
            if bundle.completed[0] == 1:
                raise BoomError("train step broke")
            return inner(batch)

        process.train_step(train_step)
    elif where == "validation_step":

        def validation_step(batch):
            if bundle.completed[0] == 1:
                raise BoomError("validation step broke")
            return 1.0

        process.validation_step(validation_step)
    elif where == "on_epoch_end":

        @process.on_epoch_end
        def hook(record):
            if record.epoch == 2:
                raise BoomError("hook broke")

    elif where == "scheduler":
        process = ht.TrainingProcess(
            bundle.model, bundle.optimizer, bundle.train_loader, loss_fn=bundle.loss_fn,
            epochs=3, scheduler=CountingScheduler(),
        )  # fmt: skip
    elif where == "loss_fn":
        state = {"epochs": 0}

        def loss_fn(outputs, targets):
            if state["epochs"] == 1:
                raise BoomError("loss function broke")
            return nn.functional.cross_entropy(outputs, targets)

        process = ht.TrainingProcess(
            bundle.model, bundle.optimizer, bundle.train_loader, loss_fn=loss_fn, epochs=3,
            on_epoch_end=lambda record: state.update(epochs=record.epoch),
        )  # fmt: skip
    elif where == "data":
        process = ht.TrainingProcess(
            bundle.model, bundle.optimizer, FlakyData(), loss_fn=bundle.loss_fn, epochs=3
        )
    return process


@pytest.mark.spec("R2")
@pytest.mark.parametrize(
    "where", ["train_step", "validation_step", "on_epoch_end", "scheduler", "loss_fn", "data"]
)
def test_an_error_in_user_code_ends_the_run_and_carries_the_cause(where, capsys):
    with pytest.raises(ht.TrainingFailedError) as info:
        ht.run(failing_process(where), gui=False)
    error = info.value
    assert isinstance(error.__cause__, BoomError) and "broke" in str(error)
    assert error.result.state is State.FAILED
    assert error.result.epochs_completed == 1  # what completed before is kept
    assert "broke" in capsys.readouterr().out  # and the user is told


@pytest.mark.spec("R2", "R5")
def test_ctrl_c_in_a_console_run_ends_it_and_propagates():
    bundle = build_process(epochs=5, native_steps=False)
    inner = bundle.process.resolve_steps().train
    threads = threading.active_count()

    def train_step(batch):
        if bundle.completed[0] == 2:
            raise KeyboardInterrupt
        return inner(batch)

    bundle.process.train_step(train_step)
    with pytest.raises(KeyboardInterrupt):
        ht.run(bundle.process, gui=False)
    assert threading.active_count() == threads


@pytest.mark.gui
@pytest.mark.spec("R5", "G5")
def test_a_failure_is_shown_in_the_window_which_stays_browsable(drive):
    bundle = build_process(epochs=4, native_steps=False)
    inner = bundle.process.resolve_steps().train

    def train_step(batch):
        if bundle.completed[0] == 2:
            raise BoomError("exploded in epoch 3")
        return inner(batch)

    bundle.process.train_step(train_step)
    shown: list[str] = []
    driver = drive(
        [
            ("failure shown", lambda w: "FAILED" in w.terminal.text(),
             lambda w: (shown.append(w.terminal.text()), type_command(w, "history"))),
            ("history after failure", lambda w: "train" in w.terminal.text().split("history")[-1],
             lambda w: type_command(w, "quit")),
        ],
        timeout=60,
    )  # fmt: skip
    with pytest.raises(ht.TrainingFailedError) as info:
        ht.run(bundle.process, isolate_gui=False)
    driver.check()
    assert isinstance(info.value.__cause__, BoomError)
    assert info.value.result.epochs_completed == 2
    assert "exploded in epoch 3" in shown[0] and "BoomError" in shown[0]


@pytest.mark.gui
@pytest.mark.spec("R5")
def test_closing_the_window_stops_training_promptly(drive):
    bundle = build_process(epochs=1000, step_delay=0.001)
    driver = drive(
        [("two epochs", lambda w: w.session.epoch >= 2, lambda w: w.close())], timeout=60
    )
    result = ht.run(bundle.process, isolate_gui=False)
    driver.check()
    assert result.state is State.STOPPED and 2 <= result.epochs_completed < 1000


@pytest.mark.gui
@pytest.mark.spec("R2")
@pytest.mark.parametrize(
    "options",
    [
        {"torch_threads": "bogus"},
        {"torch_threads": 0},
        {"torch_threads": True},
        {"torch_threads": -2},
        {"log_every": 0},
        {"log_every": 1.5},
        {"gui": False, "start_paused": True},
    ],
    ids=str,
)
def test_invalid_run_options_are_rejected_before_anything_starts(qapp, options):
    bundle = build_process(epochs=2)
    with pytest.raises(ValueError):
        ht.run(bundle.process, close_on_finish=True, **options)
    assert bundle.calls == []  # nothing trained
    assert active_window() is None and window_process_children() == []


@pytest.mark.gui
@pytest.mark.spec("R4")
@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="X11 / Wayland display check")
@pytest.mark.parametrize("isolate_gui", [True, False])
def test_without_a_display_the_run_fails_clearly(monkeypatch, isolate_gui):
    for variable in ("QT_QPA_PLATFORM", "DISPLAY", "WAYLAND_DISPLAY"):
        monkeypatch.delenv(variable, raising=False)
    with pytest.raises(RuntimeError, match="display"):
        ht.run(build_process(epochs=1).process, isolate_gui=isolate_gui, close_on_finish=True)
    assert window_process_children() == []


@pytest.mark.gui
@pytest.mark.spec("R7")
def test_a_custom_command_clashing_with_a_built_in_is_rejected_before_the_window_opens(qapp):
    bundle = build_process(epochs=1)
    bundle.process.command("pause", summary="my own pause")(lambda context: "mine")
    with pytest.raises(ValueError):
        ht.run(bundle.process, isolate_gui=False, close_on_finish=True)
    assert active_window() is None and bundle.calls == []


@pytest.mark.gui
@pytest.mark.spec("R7", "G5")
def test_a_broken_custom_command_does_not_disturb_training(drive):
    bundle = build_process(epochs=2)

    @bundle.process.command("broken", summary="always fails")
    def broken(context):
        raise RuntimeError("handler bug")

    driver = drive(
        [
            ("paused", lambda w: w.session.state is State.PAUSED,
             lambda w: type_command(w, "broken")),
            ("error shown", lambda w: "handler bug" in w.terminal.text(),
             lambda w: type_command(w, "resume")),
        ],
        timeout=60,
    )  # fmt: skip
    result = ht.run(bundle.process, isolate_gui=False, start_paused=True, close_on_finish=True)
    driver.check()
    assert result.state is State.FINISHED and result.epochs_completed == 2


@pytest.mark.gui
@pytest.mark.spec("R1", "G1")
def test_huge_layers_are_sampled_and_drawn(qapp):
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(4, 1500), nn.ReLU(), nn.Linear(1500, 1500), nn.ReLU(),
                          nn.Linear(1500, 3))  # fmt: skip
    process = ht.TrainingProcess(
        model,
        torch.optim.SGD(model.parameters(), lr=0.01),
        [(torch.randn(8, 4), torch.randint(0, 3, (8,))) for _ in range(2)],
        loss_fn=nn.CrossEntropyLoss(),
        epochs=2,
    )
    result = ht.run(process, isolate_gui=False, close_on_finish=True)
    middle = result.final_weights.layer("2")
    assert middle.matrix.shape == (20, 20)  # drawn: the default 20 neurons per column
    assert middle.stats.count == 1500 * 1500  # described: the whole layer


@pytest.mark.gui
@pytest.mark.spec("G3")
def test_non_finite_losses_do_not_break_the_window(qapp):
    values = iter([math.nan] * 6 + [math.inf] * 6 + [0.5] * 6)
    bundle = build_process(epochs=3, validation=False, native_steps=False)
    bundle.process.train_step(lambda batch: next(values))
    result = ht.run(bundle.process, isolate_gui=False, close_on_finish=True)
    assert result.state is State.FINISHED and result.epochs_completed == 3


@pytest.mark.gui
@pytest.mark.spec("R3")
def test_several_runs_in_one_program_leave_nothing_behind(qapp):
    threads = threading.active_count()
    for options in ({"isolate_gui": False}, {"isolate_gui": False}, {"gui": False}):
        result = ht.run(build_process(epochs=2).process, close_on_finish=True, **options)
        assert result.epochs_completed == 2
    assert active_window() is None
    assert threading.active_count() <= threads
    assert torch.get_num_threads() >= 1


@pytest.mark.spec("R2")
def test_a_model_without_parameters_to_draw_still_trains():
    torch.manual_seed(0)
    model = nn.Sequential(nn.PReLU())
    process = ht.TrainingProcess(
        model,
        torch.optim.SGD(model.parameters(), lr=0.1),
        [(torch.randn(4, 3), torch.randn(4, 3))],
        loss_fn=nn.MSELoss(),
        epochs=2,
    )
    result = ht.run(process, gui=False)
    assert result.state is State.FINISHED and result.final_weights.layers == ()

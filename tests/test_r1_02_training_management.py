"""R1 requirement 2 - "The package **MUST** provide a method to automatically manage the
training process. It **MUST** allow users to defind the training process in the native
PyTorch-style way. Then, wrap the defined training process into a Python object. And, pass
the wrap object to the management method the package provides. The package may provide
several `decorator` or `class` or other similar Python object to distinguish the different
steps or components of the training process."

* the three steps: native step functions -> ``TrainingProcess`` (decorators or constructor
  arguments, or the default supervised steps) -> ``ht.run``, which runs every epoch;
* what a step may return (and what it may not), train / validation modes, averaging,
  epoch hooks, schedulers, data sources;
* incomplete or invalid processes and options are refused before anything starts;
* errors in the user's code reach the caller with their cause;
* the console report, the returned ``TrainingResult``;
* managed GUI runs: closing by themselves, lingering for more epochs, starting paused,
  repeated runs, the ``torch_threads`` option.
"""

from __future__ import annotations

import os

import numpy as np
import psutil
import pytest
import torch
from htspec.models import TinyNet, learnable_data, loader, make_setup
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

import hollywood_training as ht
from hollywood_training.training.steps import StepOutputError

pytestmark = pytest.mark.req("R1.2")


def native_parts(seed: int = 0, samples: int = 128):
    torch.manual_seed(seed)
    model = TinyNet()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.05)
    loss_fn = nn.CrossEntropyLoss()
    data = DataLoader(learnable_data(samples, seed), batch_size=16)
    return model, optimizer, loss_fn, data


# --------------------------------------------------------------- the three steps
def test_a_native_step_wrapped_and_handed_to_run_trains_the_network():
    model, optimizer, loss_fn, data = native_parts()
    process = ht.TrainingProcess(model, optimizer, data, loss_fn=loss_fn, epochs=8)

    @process.train_step
    def train_step(batch):
        inputs, targets = batch
        optimizer.zero_grad()
        loss = loss_fn(model(inputs), targets)
        loss.backward()
        optimizer.step()
        return loss

    result = ht.run(process, gui=False)
    assert result.state is ht.TrainingState.FINISHED
    assert [r.epoch for r in result.history] == list(range(1, 9))
    assert result.train_losses[-1] < result.train_losses[0] * 0.8  # it really learned


def test_the_step_decorators_return_the_functions_unchanged():
    process = make_setup(native_steps=False).process

    def train(batch):
        return 0.0

    def validate(batch):
        return 0.0

    def hook(record):
        pass

    def handler(context):
        return "ok"

    assert process.train_step(train) is train
    assert process.validation_step(validate) is validate
    assert process.on_epoch_end(hook) is hook
    assert process.command("probe-it", summary="test")(handler) is handler


def test_steps_and_hooks_can_be_given_to_the_constructor():
    model, optimizer, loss_fn, data = native_parts()
    calls = {"train": 0, "val": 0, "hooks": []}

    def train(batch):
        calls["train"] += 1
        inputs, targets = batch
        optimizer.zero_grad()
        loss = loss_fn(model(inputs), targets)
        loss.backward()
        optimizer.step()
        return loss

    def validate(batch):
        calls["val"] += 1
        return loss_fn(model(batch[0]), batch[1])

    process = ht.TrainingProcess(
        model, optimizer, data, val_data=data, epochs=3, train_step=train,
        validation_step=validate,
        on_epoch_end=[lambda r: calls["hooks"].append(("a", r.epoch)),
                      lambda r: calls["hooks"].append(("b", r.epoch))],
    )  # fmt: skip
    ht.run(process, gui=False)
    assert calls["train"] == calls["val"] == 3 * len(data)
    assert calls["hooks"] == [("a", 1), ("b", 1), ("a", 2), ("b", 2), ("a", 3), ("b", 3)]


def test_the_default_supervised_steps_train_and_validate():
    model, optimizer, loss_fn, data = native_parts()
    process = ht.TrainingProcess(model, optimizer, data, val_data=data, loss_fn=loss_fn, epochs=8)
    result = ht.run(process, gui=False)
    assert result.train_losses[-1] < result.train_losses[0] * 0.8
    assert all(value is not None for value in result.val_losses)


# ------------------------------------------------------------------ step outputs
ACCEPTED = {
    "scalar tensor": (lambda: torch.tensor(0.5, requires_grad=True), 0.5),
    "one-element tensor": (lambda: torch.tensor([0.5]), 0.5),
    "float": (lambda: 0.5, 0.5),
    "int": (lambda: 2, 2.0),
    "numpy float": (lambda: np.float32(0.25), 0.25),
    "mapping with metrics": (lambda: {"loss": torch.tensor(0.5), "accuracy": 0.75}, 0.5),
}


@pytest.mark.parametrize("kind", list(ACCEPTED))
def test_accepted_step_results(kind):
    make_value, expected = ACCEPTED[kind]
    setup = make_setup(native_steps=False, validation=False, epochs=2)

    @setup.process.train_step
    def step(_batch):
        return make_value()

    result = ht.run(setup.process, gui=False)
    assert result.train_losses == [pytest.approx(expected)] * 2
    if kind == "mapping with metrics":
        assert result.history[0].train_metrics == {"accuracy": pytest.approx(0.75)}


REFUSED = {
    "None": lambda: None,
    "text": lambda: "0.5",
    "bool": lambda: True,
    "vector": lambda: torch.ones(2),
    "mapping without loss": lambda: {"accuracy": 1.0},
    "metric that is not a number": lambda: {"loss": 0.5, "note": "fine"},
}


@pytest.mark.parametrize("kind", list(REFUSED))
def test_refused_step_results_fail_the_run_with_a_clear_cause(kind):
    setup = make_setup(native_steps=False, validation=False, epochs=2)
    setup.process.train_step(lambda _batch: REFUSED[kind]())
    with pytest.raises(ht.TrainingFailedError) as caught:
        ht.run(setup.process, gui=False)
    assert isinstance(caught.value.__cause__, StepOutputError)
    assert caught.value.result.epochs_completed == 0


def test_validation_runs_in_eval_mode_without_gradients():
    setup = make_setup(native_steps=False, epochs=2)
    seen = {"train": set(), "val": set()}
    model, optimizer, loss_fn = setup.model, setup.optimizer, setup.loss_fn

    @setup.process.train_step
    def train(batch):
        seen["train"].add((model.training, torch.is_grad_enabled()))
        optimizer.zero_grad()
        loss = loss_fn(model(batch[0]), batch[1])
        loss.backward()
        optimizer.step()
        return loss

    @setup.process.validation_step
    def validate(batch):
        seen["val"].add((model.training, torch.is_grad_enabled()))
        return loss_fn(model(batch[0]), batch[1])

    ht.run(setup.process, gui=False)
    assert seen == {"train": {(True, True)}, "val": {(False, False)}}


def test_losses_and_metrics_are_averaged_over_the_batches():
    data = [(torch.zeros(2, 2), torch.zeros(2, dtype=torch.long))] * 4
    model = TinyNet()
    process = ht.TrainingProcess(model, torch.optim.SGD(model.parameters(), 0.1), data,
                                 val_data=data[:2], epochs=2)  # fmt: skip
    counter = {"train": 0, "val": 0}

    @process.train_step
    def train(_batch):
        counter["train"] += 1
        k = (counter["train"] - 1) % 4  # 0, 1, 2, 3 in every epoch
        return {"loss": float(k), "double": 2.0 * k}

    @process.validation_step
    def validate(_batch):
        counter["val"] += 1
        return {"loss": 10.0 * (counter["val"] % 2), "hits": 1.0}

    result = ht.run(process, gui=False)
    for record in result.history:
        assert record.train_loss == pytest.approx(1.5)
        assert record.train_metrics == {"double": pytest.approx(3.0)}
        assert record.val_loss == pytest.approx(5.0)
        assert record.val_metrics == {"hits": pytest.approx(1.0)}
        assert (record.train_batches, record.val_batches) == (4, 2)


def test_a_scheduler_is_stepped_once_per_epoch():
    setup = make_setup(epochs=5, lr=0.08)
    scheduler = torch.optim.lr_scheduler.StepLR(setup.optimizer, step_size=2, gamma=0.5)
    process = ht.TrainingProcess(
        setup.model, setup.optimizer, setup.train_loader, loss_fn=setup.loss_fn, epochs=5,
        scheduler=scheduler,
    )  # fmt: skip
    result = ht.run(process, gui=False)
    rates = [record.learning_rate for record in result.history]
    assert rates == pytest.approx([0.08, 0.08, 0.04, 0.04, 0.02])


@pytest.mark.parametrize("validation", [True, False])
def test_reduce_on_plateau_receives_the_monitored_loss(validation):
    setup = make_setup(epochs=3, validation=validation)
    received = []

    class Recording(torch.optim.lr_scheduler.ReduceLROnPlateau):
        def step(self, metrics, *args, **kwargs):
            received.append(float(metrics))
            return super().step(metrics, *args, **kwargs)

    process = ht.TrainingProcess(
        setup.model, setup.optimizer, setup.train_loader, loss_fn=setup.loss_fn, epochs=3,
        val_data=setup.val_loader, scheduler=Recording(setup.optimizer),
    )  # fmt: skip
    result = ht.run(process, gui=False)
    expected = [r.val_loss if validation else r.train_loss for r in result.history]
    assert received == pytest.approx(expected)


# ------------------------------------------------------------------ data sources
@pytest.mark.parametrize("kind", ["DataLoader", "list", "tuple"])
def test_re_iterable_data_sources(kind):
    batches = [(torch.randn(4, 2), torch.randint(0, 3, (4,))) for _ in range(3)]
    source = {"DataLoader": loader(12, 4), "list": batches, "tuple": tuple(batches)}[kind]
    model = TinyNet()
    process = ht.TrainingProcess(model, torch.optim.SGD(model.parameters(), 0.1), source,
                                 loss_fn=nn.CrossEntropyLoss(), epochs=2)  # fmt: skip
    result = ht.run(process, gui=False)
    assert [record.train_batches for record in result.history] == [3, 3]


@pytest.mark.parametrize("kind", ["generator", "iterator", "not iterable"])
def test_one_shot_or_non_iterable_data_is_refused(kind):
    batches = [(torch.randn(4, 2), torch.randint(0, 3, (4,)))]
    source = {"generator": (b for b in batches), "iterator": iter(batches), "not iterable": 5}
    model = TinyNet()
    with pytest.raises(TypeError):
        ht.TrainingProcess(model, torch.optim.SGD(model.parameters(), 0.1), source[kind],
                           loss_fn=nn.CrossEntropyLoss(), epochs=1)  # fmt: skip


def test_an_empty_data_source_fails_clearly():
    empty = DataLoader(TensorDataset(torch.empty(0, 2), torch.empty(0, dtype=torch.long)))
    model = TinyNet()
    process = ht.TrainingProcess(model, torch.optim.SGD(model.parameters(), 0.1), empty,
                                 loss_fn=nn.CrossEntropyLoss(), epochs=1)  # fmt: skip
    with pytest.raises(ht.TrainingFailedError) as caught:
        ht.run(process, gui=False)
    assert "no batches" in str(caught.value.__cause__)


# ------------------------------------------------------------------- validation
def _process(**overrides):
    model = TinyNet()
    arguments = {
        "model": model, "optimizer": torch.optim.SGD(model.parameters(), 0.1),
        "train_data": loader(), "loss_fn": nn.CrossEntropyLoss(), "epochs": 2,
    }  # fmt: skip
    arguments.update(overrides)
    model_arg = arguments.pop("model")
    optimizer = arguments.pop("optimizer")
    data = arguments.pop("train_data")
    return ht.TrainingProcess(model_arg, optimizer, data, **arguments)


@pytest.mark.parametrize(
    ("overrides", "error"),
    [
        ({"epochs": 0}, ValueError), ({"epochs": -1}, ValueError), ({"epochs": True}, ValueError),
        ({"epochs": 1.5}, ValueError), ({"epochs": "3"}, ValueError),
        ({"batch_size": 0}, ValueError), ({"batch_size": -2}, ValueError),
        ({"batch_size": True}, ValueError), ({"batch_size": 2.5}, ValueError),
        ({"model": "net"}, TypeError), ({"optimizer": object()}, TypeError),
        ({"loss_fn": 5}, TypeError), ({"train_step": 5}, TypeError),
        ({"on_epoch_end": 5}, TypeError),
    ],
)  # fmt: skip
def test_invalid_process_arguments_are_refused(overrides, error):
    with pytest.raises(error):
        _process(**overrides)


@pytest.mark.parametrize("gui", [False, True])
def test_a_process_without_a_training_step_is_refused_before_anything_starts(gui):
    model = TinyNet()
    process = ht.TrainingProcess(model, torch.optim.SGD(model.parameters(), 0.1), loader(),
                                 epochs=2)  # fmt: skip
    children = len(psutil.Process().children())
    with pytest.raises(ht.ProcessError):
        ht.run(process, gui=gui)
    assert len(psutil.Process().children()) == children  # no window process was started


def test_a_validation_step_needs_validation_data():
    process = _process()
    process.validation_step(lambda batch: 0.0)
    with pytest.raises(ht.ProcessError):
        ht.run(process, gui=False)


def test_validation_data_needs_a_validation_step_or_a_loss_function():
    model = TinyNet()
    process = ht.TrainingProcess(model, torch.optim.SGD(model.parameters(), 0.1), loader(),
                                 val_data=loader(), epochs=1)  # fmt: skip
    process.train_step(lambda batch: 0.0)
    with pytest.raises(ht.ProcessError):
        ht.run(process, gui=False)


@pytest.mark.parametrize(
    ("options", "error"),
    [
        ({"torch_threads": 0}, ValueError), ({"torch_threads": -1}, ValueError),
        ({"torch_threads": "many"}, ValueError), ({"torch_threads": True}, ValueError),
        ({"torch_threads": 1.5}, ValueError), ({"log_every": 0}, ValueError),
        ({"log_every": -1}, ValueError), ({"log_every": True}, ValueError),
        ({"log_every": "2"}, ValueError), ({"start_paused": True, "gui": False}, ValueError),
        ({"max_neurons": 1, "gui": False}, ValueError),
    ],
)  # fmt: skip
def test_invalid_run_options_are_refused(options, error):
    with pytest.raises(error):
        ht.run(make_setup().process, **options)


def test_run_needs_a_training_process():
    with pytest.raises(TypeError):
        ht.run(lambda: None)


# ------------------------------------------------------------------------ errors
@pytest.mark.parametrize("where", ["train", "validation", "hook"])
def test_errors_in_the_users_code_reach_the_caller_with_their_cause(where):
    setup = make_setup(epochs=4, native_steps=False)
    model, optimizer, loss_fn = setup.model, setup.optimizer, setup.loss_fn
    state = {"epoch": 0}

    class BoomError(Exception):
        pass

    @setup.process.train_step
    def train(batch):
        if where == "train" and state["epoch"] == 1:
            raise BoomError("in the training step")
        optimizer.zero_grad()
        loss = loss_fn(model(batch[0]), batch[1])
        loss.backward()
        optimizer.step()
        return loss

    @setup.process.validation_step
    def validate(batch):
        if where == "validation" and state["epoch"] == 1:
            raise BoomError("in the validation step")
        return loss_fn(model(batch[0]), batch[1])

    @setup.process.on_epoch_end
    def hook(record):
        state["epoch"] = record.epoch
        if where == "hook" and record.epoch == 2:
            raise BoomError("in the hook")

    with pytest.raises(ht.TrainingFailedError) as caught:
        ht.run(setup.process, gui=False)
    assert isinstance(caught.value.__cause__, BoomError)
    assert caught.value.result.state is ht.TrainingState.FAILED
    assert caught.value.result.epochs_completed == 1


# ---------------------------------------------------------- report and result
def test_the_console_frontend_reports_the_run(capsys):
    setup = make_setup(epochs=5, normalization="z-score")
    ht.run(setup.process, gui=False, log_every=2)
    out = capsys.readouterr().out
    assert "HOLLYWOOD TRAINING :: TinyNet" in out
    assert "epochs 5 | batch size 16 | lr 0.05" in out
    assert "loss CrossEntropyLoss (mean) | optimizer SGD" in out
    assert "normalisation data: z-score" in out
    lines = [line for line in out.splitlines() if line.startswith("epoch ")]
    epochs = [line.split("|")[0].split()[1] for line in lines]
    assert epochs == ["1/5", "2/5", "4/5", "5/5"]  # first, every second, last
    assert "FINISHED: completed 5 epochs in" in out
    assert "best loss" in out


def test_the_result_describes_the_whole_run():
    setup = make_setup(epochs=4)
    result = ht.run(setup.process, gui=False)
    assert isinstance(result, ht.TrainingResult)
    assert result.state is ht.TrainingState.FINISHED and result.error is None
    assert result.epochs_completed == 4
    assert len(result.train_losses) == len(result.val_losses) == 4
    assert result.elapsed > 0
    best = min(result.history, key=lambda r: r.val_loss)
    assert result.best_epoch() is best


# ---------------------------------------------------------------------- GUI runs
def test_an_in_process_window_run_closes_by_itself_when_asked(qapp):
    setup = make_setup(epochs=3)
    result = ht.run(setup.process, isolate_gui=False, close_on_finish=True)
    assert result.state is ht.TrainingState.FINISHED and result.epochs_completed == 3


@pytest.mark.subprocess
def test_an_isolated_window_run_closes_by_itself_when_asked():
    setup = make_setup(epochs=3)
    result = ht.run(setup.process, close_on_finish=True)
    assert result.state is ht.TrainingState.FINISHED and result.epochs_completed == 3


def test_a_window_run_lingers_for_more_epochs_until_the_window_is_closed(pilot):
    seen = {}

    def still_open(window) -> None:
        seen["visible"] = window.isVisible()
        seen["state"] = window.session.state
        window.close()

    pilot.on_finish(pilot.after(0.6, still_open))
    result = pilot.run(make_setup(epochs=2).process)
    assert seen == {"visible": True, "state": ht.TrainingState.FINISHED}
    assert result.state is ht.TrainingState.FINISHED


def test_start_paused_waits_for_resume(pilot):
    setup = make_setup(epochs=3)
    seen = {}

    def look(window) -> None:
        seen["state"] = window.session.state
        seen["steps before resume"] = len(setup.calls["train"])

    pilot.on_start(pilot.after(0.5, pilot.chain(look, pilot.line("resume"))))
    pilot.on_finish(pilot.close)
    result = pilot.run(setup.process, start_paused=True)
    assert seen == {"state": ht.TrainingState.PAUSED, "steps before resume": 0}
    assert result.epochs_completed == 3


def test_several_runs_in_one_process(qapp):
    for _ in range(2):
        assert ht.run(make_setup(epochs=2).process, gui=False).epochs_completed == 2
    for _ in range(2):
        result = ht.run(make_setup(epochs=2).process, isolate_gui=False, close_on_finish=True)
        assert result.epochs_completed == 2


@pytest.mark.parametrize("setting", ["auto", None, 1])
def test_the_torch_threads_option(qapp, setting):
    before = torch.get_num_threads()
    during = []
    setup = make_setup(epochs=1, train_hook=lambda _b: during.append(torch.get_num_threads()))
    ht.run(setup.process, isolate_gui=False, close_on_finish=True, torch_threads=setting)
    cores = len(os.sched_getaffinity(0))
    expected = {"auto": min(before, max(1, cores // 2)), None: before, 1: 1}[setting]
    assert set(during) == {expected}
    assert torch.get_num_threads() == before  # restored

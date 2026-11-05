"""R2: training is defined in native PyTorch style, wrapped into an object and handed to
the package's management method, which then manages the whole training process."""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader

import hollywood_training as ht
from hollywood_training.sync import EpochCompleted

from .helpers import ManagedRun, TinyNet, build_process, xor_data

pytestmark = pytest.mark.spec("R2")


def plain_process(epochs: int = 2, **options) -> tuple[ht.TrainingProcess, nn.Module, object]:
    torch.manual_seed(0)
    model = TinyNet()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    loader = DataLoader(xor_data(64), batch_size=16)  # 4 batches
    options.setdefault("loss_fn", nn.CrossEntropyLoss())
    return ht.TrainingProcess(model, optimizer, loader, epochs=epochs, **options), model, optimizer


# ---------------------------------------------------------------- wrapping steps
def test_decorated_native_steps_are_managed_epoch_by_epoch():
    bundle = build_process(epochs=3)
    records = []
    bundle.process.on_epoch_end(records.append)
    result = ht.run(bundle.process, gui=False)

    batches, val_batches = len(bundle.train_loader), len(bundle.val_loader)
    phases = [phase for phase, _t, _done in bundle.calls]
    assert phases.count("train") == 3 * batches
    assert phases.count("validation") == 3 * val_batches
    # every epoch: all training batches, then all validation batches
    assert phases == (["train"] * batches + ["validation"] * val_batches) * 3
    assert [record.epoch for record in records] == [1, 2, 3]
    assert result.state is ht.TrainingState.FINISHED
    assert [record.epoch for record in result.history] == [1, 2, 3]


def test_constructor_arguments_are_equivalent_to_the_decorators():
    seen = {"train": 0, "validation": 0, "hooks": []}
    torch.manual_seed(0)
    model = TinyNet()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    loss_fn = nn.CrossEntropyLoss()

    def train_step(batch):
        seen["train"] += 1
        inputs, targets = batch
        optimizer.zero_grad()
        loss = loss_fn(model(inputs), targets)
        loss.backward()
        optimizer.step()
        return loss

    def validation_step(batch):
        seen["validation"] += 1
        inputs, targets = batch
        return loss_fn(model(inputs), targets)

    process = ht.TrainingProcess(
        model,
        optimizer,
        DataLoader(xor_data(32), batch_size=8),
        val_data=DataLoader(xor_data(16, 1), batch_size=8),
        epochs=2,
        train_step=train_step,
        validation_step=validation_step,
        on_epoch_end=[lambda r: seen["hooks"].append(("a", r.epoch)),
                      lambda r: seen["hooks"].append(("b", r.epoch))],
    )  # fmt: skip
    result = ht.run(process, gui=False)
    assert result.epochs_completed == 2
    assert seen["train"] == 2 * 4 and seen["validation"] == 2 * 2
    assert seen["hooks"] == [("a", 1), ("b", 1), ("a", 2), ("b", 2)]


def test_default_supervised_steps_need_only_a_loss_function():
    torch.manual_seed(0)
    model = TinyNet(hidden=16)
    process = ht.TrainingProcess(
        model,
        torch.optim.Adam(model.parameters(), lr=0.05),
        DataLoader(xor_data(256), batch_size=32, shuffle=True),
        val_data=DataLoader(xor_data(64, 1), batch_size=64),
        loss_fn=nn.CrossEntropyLoss(),
        epochs=30,
    )
    result = ht.run(process, gui=False)
    assert result.state is ht.TrainingState.FINISHED
    assert result.train_losses[-1] < result.train_losses[0] * 0.8  # it really trains
    assert all(loss is not None for loss in result.val_losses)


def test_result_holds_the_complete_history():
    bundle = build_process(epochs=4)
    result = ht.run(bundle.process, gui=False)
    assert result.epochs_completed == 4
    assert len(result.train_losses) == len(result.val_losses) == 4
    assert result.elapsed > 0
    assert result.error is None
    assert result.final_weights is not None and result.final_weights.epoch == 4
    best = result.best_epoch()
    assert best.val_loss == min(result.val_losses)
    for record in result.history:
        assert record.train_batches == len(bundle.train_loader)
        assert record.val_batches == len(bundle.val_loader)
        assert record.learning_rate == pytest.approx(0.1)
        assert record.duration > 0


def test_epoch_losses_are_the_mean_of_the_batch_losses():
    losses = iter([1.0, 2.0, 3.0, 6.0, 0.5, 0.5, 0.5, 0.5])
    val_losses = iter([4.0, 2.0, 1.0, 1.0])
    process, _model, _ = plain_process(epochs=2, val_data=[1, 2])

    process.train_step(lambda batch: next(losses))
    process.validation_step(lambda batch: next(val_losses))
    result = ht.run(process, gui=False)
    assert result.train_losses == pytest.approx([3.0, 0.5])
    assert result.val_losses == pytest.approx([3.0, 1.0])


def test_metrics_returned_by_steps_are_averaged_per_epoch():
    accuracies = iter([0.5, 1.0, 0.0, 0.5])
    process, _model, _ = plain_process(epochs=1, val_data=[1])

    @process.train_step
    def train_step(batch):
        return {"loss": torch.tensor(1.0), "accuracy": torch.tensor(next(accuracies))}

    @process.validation_step
    def validation_step(batch):
        return {"loss": 2.0, "accuracy": 0.25}

    record = ht.run(process, gui=False).history[0]
    assert record.train_metrics == {"accuracy": pytest.approx(0.5)}
    assert record.val_metrics == {"accuracy": pytest.approx(0.25)}


@pytest.mark.parametrize(
    "value",
    [torch.tensor(0.5), torch.tensor([0.5]), 0.5, 1, np.float32(0.5), np.float64(0.5)],
    ids=["tensor", "one-element-tensor", "float", "int", "numpy32", "numpy64"],
)
def test_steps_may_return_any_scalar_loss(value):
    process, _model, _ = plain_process(epochs=1)
    process.train_step(lambda batch: value)
    result = ht.run(process, gui=False)
    assert result.train_losses == [pytest.approx(float(value))]


@pytest.mark.parametrize(
    "value",
    [None, torch.ones(3), "0.5", True, {"accuracy": 1.0}, [0.5], object()],
    ids=["none", "vector", "string", "bool", "no-loss-key", "list", "object"],
)
def test_steps_returning_no_usable_loss_fail_clearly(value):
    process, _model, _ = plain_process(epochs=2)
    process.train_step(lambda batch: value)
    with pytest.raises(ht.TrainingFailedError) as info:
        ht.run(process, gui=False)
    assert isinstance(info.value.__cause__, TypeError)
    assert "train_step" in str(info.value.__cause__)
    assert info.value.result.state is ht.TrainingState.FAILED
    assert info.value.result.epochs_completed == 0


# --------------------------------------------------------------- schedules & modes
def test_the_scheduler_is_stepped_once_per_epoch():
    process, _model, optimizer = plain_process(epochs=3)
    process = ht.TrainingProcess(
        process.model,
        optimizer,
        process.train_data,
        loss_fn=process.loss_fn,
        epochs=3,
        scheduler=torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5),
    )
    result = ht.run(process, gui=False)
    assert [r.learning_rate for r in result.history] == pytest.approx([0.1, 0.05, 0.025])
    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.0125)


@pytest.mark.parametrize("validation", [True, False])
def test_plateau_scheduler_receives_the_monitored_loss(validation):
    received = []

    class Plateau(torch.optim.lr_scheduler.ReduceLROnPlateau):
        def step(self, metrics, *args, **kwargs):
            received.append(float(metrics))
            return super().step(metrics, *args, **kwargs)

    torch.manual_seed(0)
    model = TinyNet()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    bundle = build_process(
        epochs=3,
        validation=validation,
        model=model,
        optimizer=optimizer,
        scheduler=Plateau(optimizer),
    )
    result = ht.run(bundle.process, gui=False)
    expected = result.val_losses if validation else result.train_losses
    assert received == pytest.approx(expected)


def test_training_and_validation_run_in_the_right_modes():
    process, model, _ = plain_process(
        epochs=2, val_data=[(torch.randn(4, 2), torch.ones(4).long())]
    )
    seen = []

    @process.train_step
    def train_step(batch):
        seen.append(("train", model.training, torch.is_grad_enabled()))
        return 1.0

    @process.validation_step
    def validation_step(batch):
        seen.append(("validation", model.training, torch.is_grad_enabled()))
        return 1.0

    ht.run(process, gui=False)
    assert {entry for entry in seen if entry[0] == "train"} == {("train", True, True)}
    assert {entry for entry in seen if entry[0] == "validation"} == {("validation", False, False)}


# --------------------------------------------------------------------- data sources
def test_plain_lists_are_valid_data_sources_and_the_batch_size_is_inferred():
    batches = [(torch.randn(5, 2), torch.randint(0, 2, (5,))) for _ in range(3)]
    torch.manual_seed(0)
    model = TinyNet()
    process = ht.TrainingProcess(
        model,
        torch.optim.SGD(model.parameters(), lr=0.1),
        batches,
        loss_fn=nn.CrossEntropyLoss(),
        epochs=2,
    )
    with ManagedRun(process) as run:
        result = run.join()
    assert result.epochs_completed == 2 and result.history[0].train_batches == 3
    assert run.frontend.of_type(EpochCompleted)[-1].batch_size == 5


def test_dictionary_batches_with_a_custom_step():
    batches = [{"x": torch.randn(4, 2), "y": torch.randint(0, 2, (4,))} for _ in range(2)]
    torch.manual_seed(0)
    model = TinyNet()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    loss_fn = nn.CrossEntropyLoss()

    def train_step(batch):
        optimizer.zero_grad()
        loss = loss_fn(model(batch["x"]), batch["y"])
        loss.backward()
        optimizer.step()
        return loss

    process = ht.TrainingProcess(model, optimizer, batches, epochs=2, train_step=train_step)
    assert ht.run(process, gui=False).epochs_completed == 2


def test_the_smallest_possible_run():
    torch.manual_seed(0)
    model = TinyNet()
    process = ht.TrainingProcess(
        model,
        torch.optim.SGD(model.parameters(), lr=0.1),
        [(torch.randn(1, 2), torch.tensor([1]))],
        loss_fn=nn.CrossEntropyLoss(),
        epochs=1,
    )
    result = ht.run(process, gui=False)
    assert result.state is ht.TrainingState.FINISHED
    assert result.epochs_completed == 1 and result.history[0].train_batches == 1
    assert result.val_losses == [None]


@pytest.mark.parametrize("which", ["train_data", "val_data"])
def test_data_sources_without_batches_fail_clearly(which):
    torch.manual_seed(0)
    model = TinyNet()
    data = {"train_data": [(torch.randn(2, 2), torch.tensor([0, 1]))], "val_data": None}
    data[which] = []
    process = ht.TrainingProcess(
        model,
        torch.optim.SGD(model.parameters(), lr=0.1),
        data["train_data"],
        val_data=data["val_data"],
        loss_fn=nn.CrossEntropyLoss(),
        epochs=2,
    )
    with pytest.raises(ht.TrainingFailedError) as info:
        ht.run(process, gui=False)
    assert which in str(info.value) and "no batches" in str(info.value)


# ------------------------------------------------------------ invalid definitions
def test_a_process_without_any_training_step_is_rejected_before_starting():
    process, _model, _ = plain_process(loss_fn=None)
    with pytest.raises(ht.ProcessError, match="train_step"):
        ht.run(process, gui=False)
    with pytest.raises(ht.ProcessError):  # also before any window would open
        ht.run(process, close_on_finish=True)


def test_inconsistent_validation_definitions_are_rejected():
    process, _model, _ = plain_process(loss_fn=None, val_data=[1])
    process.train_step(lambda batch: 1.0)
    with pytest.raises(ht.ProcessError, match="validation_step"):
        ht.run(process, gui=False)

    process, _model, _ = plain_process()
    process.validation_step(lambda batch: 1.0)
    with pytest.raises(ht.ProcessError, match="val_data"):
        ht.run(process, gui=False)


def _generator():
    yield (torch.randn(2, 2), torch.tensor([0, 1]))


@pytest.mark.parametrize(
    ("change", "error"),
    [
        pytest.param({"model": object()}, TypeError, id="model-not-a-module"),
        pytest.param({"optimizer": object()}, TypeError, id="optimizer-without-param-groups"),
        pytest.param({"train_data": 42}, TypeError, id="train-data-not-iterable"),
        pytest.param({"train_data": _generator()}, TypeError, id="train-data-generator"),
        pytest.param({"train_data": iter([1, 2])}, TypeError, id="train-data-iterator"),
        pytest.param({"val_data": _generator()}, TypeError, id="val-data-generator"),
        pytest.param({"epochs": 0}, ValueError, id="epochs-0"),
        pytest.param({"epochs": -3}, ValueError, id="epochs-negative"),
        pytest.param({"epochs": 2.5}, ValueError, id="epochs-float"),
        pytest.param({"epochs": True}, ValueError, id="epochs-bool"),
        pytest.param({"epochs": "3"}, ValueError, id="epochs-string"),
        pytest.param({"batch_size": 0}, ValueError, id="batch-size-0"),
        pytest.param({"batch_size": True}, ValueError, id="batch-size-bool"),
        pytest.param({"batch_size": 1.5}, ValueError, id="batch-size-float"),
        pytest.param({"loss_fn": "mse"}, TypeError, id="loss-fn-not-callable"),
        pytest.param({"train_step": "step"}, TypeError, id="train-step-not-callable"),
        pytest.param({"validation_step": 3}, TypeError, id="validation-step-not-callable"),
        pytest.param({"on_epoch_end": [print, None]}, TypeError, id="hook-not-callable"),
    ],
)
def test_invalid_process_definitions_are_rejected_immediately(change, error):
    torch.manual_seed(0)
    model = TinyNet()
    arguments = {
        "model": model,
        "optimizer": torch.optim.SGD(model.parameters(), lr=0.1),
        "train_data": [(torch.randn(2, 2), torch.tensor([0, 1]))],
        "epochs": 1,
        "loss_fn": nn.CrossEntropyLoss(),
    }
    arguments.update(change)
    with pytest.raises(error):
        ht.TrainingProcess(
            arguments.pop("model"),
            arguments.pop("optimizer"),
            arguments.pop("train_data"),
            **arguments,
        )


@pytest.mark.parametrize("decorator", ["train_step", "validation_step", "on_epoch_end"])
def test_decorators_reject_non_callables(decorator):
    process, _model, _ = plain_process()
    with pytest.raises(TypeError):
        getattr(process, decorator)("not callable")


@pytest.mark.parametrize("candidate", [None, TinyNet(), {"epochs": 3}, "process"])
def test_run_accepts_only_wrapped_processes(candidate):
    with pytest.raises(TypeError):
        ht.run(candidate, gui=False)


# ------------------------------------------------------------------- the wrapper
def test_the_wrapper_exposes_what_it_was_given():
    torch.manual_seed(0)
    model = TinyNet()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    loader = DataLoader(xor_data(32), batch_size=8)
    loss_fn = nn.CrossEntropyLoss()
    process = ht.TrainingProcess(
        model, optimizer, loader, epochs=7, loss_fn=loss_fn, batch_size=8,
        normalization="z-score", name="my run",
    )  # fmt: skip
    assert process.model is model and process.optimizer is optimizer
    assert process.train_data is loader and process.loss_fn is loss_fn
    assert process.epochs == 7 and process.batch_size == 8
    assert process.normalization == "z-score" and process.name == "my run"
    assert process.val_data is None and process.scheduler is None
    assert "TinyNet" in repr(process)


def test_custom_commands_are_part_of_the_wrapped_process():
    process, _model, _ = plain_process()

    @process.command("double", summary="double a number", arguments=[ht.Argument("n", float)])
    def double(context, n):
        return f"{2 * n:g}"

    assert double(None, 2.0) == "4"  # the handler is returned unchanged
    (command,) = process.commands
    assert command.name == "double" and command.category == "custom"


def test_consecutive_runs_in_one_program():
    first = ht.run(build_process(epochs=2).process, gui=False)
    second = ht.run(build_process(epochs=3).process, gui=False)
    assert (first.epochs_completed, second.epochs_completed) == (2, 3)


def test_non_finite_losses_are_reported_but_do_not_stop_training():
    values = iter([math.nan, 1.0, math.inf, 1.0] * 10)
    process, _model, _ = plain_process(epochs=2)
    process.train_step(lambda batch: next(values))
    with ManagedRun(process) as run:
        result = run.join()
    assert result.state is ht.TrainingState.FINISHED and result.epochs_completed == 2
    warnings = [m.text for m in run.frontend.events if getattr(m, "level", None) is not None]
    assert any("non-finite" in text for text in warnings)


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
def test_training_on_a_gpu_with_the_default_steps():
    model = TinyNet().cuda()
    process = ht.TrainingProcess(
        model,
        torch.optim.SGD(model.parameters(), lr=0.1),
        DataLoader(xor_data(32), batch_size=8),
        loss_fn=nn.CrossEntropyLoss(),
        epochs=2,
    )
    result = ht.run(process, gui=False)
    assert result.epochs_completed == 2
    assert result.final_weights.layer("fc1").matrix.shape == (6, 2)

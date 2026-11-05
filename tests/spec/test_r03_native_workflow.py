"""R3: tracking (R1) and management (R2) must not significantly change the native
PyTorch workflow: same classes, same objects, same numbers, no leftovers."""

from __future__ import annotations

import copy
import io
import json
import pickle
import textwrap

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader

import hollywood_training as ht
from hollywood_training.tracking import NetworkTracker

from .helpers import RichNet, TinyNet, build_process, run_python, xor_data

pytestmark = pytest.mark.spec("R3")


@ht.track(name="picklable", output_labels=("a", "b"))
class PicklableNet(TinyNet):
    """Module-level so that pickle can find it."""


class TwinNet(TinyNet):
    """The same network without the decorator."""


def all_hooks(model: nn.Module) -> dict[str, int]:
    """Number of hooks of every kind on every submodule (torch's own bookkeeping)."""
    kinds = (
        "_forward_hooks",
        "_forward_pre_hooks",
        "_backward_hooks",
        "_backward_pre_hooks",
        "_state_dict_hooks",
        "_load_state_dict_pre_hooks",
    )
    return {
        f"{name}:{kind}": len(getattr(module, kind, {}))
        for name, module in model.named_modules()
        for kind in kinds
    }


def global_hooks() -> dict[str, int]:
    from torch.nn.modules import module as module_internals

    names = (
        "_global_forward_hooks",
        "_global_forward_pre_hooks",
        "_global_backward_hooks",
        "_global_backward_pre_hooks",
    )
    return {name: len(getattr(module_internals, name, {})) for name in names}


# ------------------------------------------------------------------ the decorator
def test_track_returns_the_class_itself_unchanged():
    class Net(TinyNet):
        def extra(self) -> int:
            return 7

    before = dict(vars(Net))
    mro = Net.__mro__
    assert ht.track(Net) is Net
    assert ht.track(name="x")(Net) is Net
    assert dict(vars(Net)) == before  # nothing added, removed or replaced
    assert Net.__mro__ == mro
    assert Net().extra() == 7 and type(Net()) is Net


def test_tracked_instances_behave_exactly_like_untracked_ones():
    torch.manual_seed(1)
    tracked = PicklableNet()
    torch.manual_seed(1)
    plain = TwinNet()
    x = torch.randn(5, 2)
    torch.testing.assert_close(tracked(x), plain(x))
    assert tracked.state_dict().keys() == plain.state_dict().keys()
    for key, value in tracked.state_dict().items():
        torch.testing.assert_close(value, plain.state_dict()[key])
    assert [n for n, _ in tracked.named_modules()] == [n for n, _ in plain.named_modules()]


def test_tracked_instances_pickle_copy_and_save_like_plain_modules():
    model = PicklableNet()
    x = torch.randn(3, 2)
    restored = pickle.loads(pickle.dumps(model))
    torch.testing.assert_close(restored(x), model(x))
    torch.testing.assert_close(copy.deepcopy(model)(x), model(x))
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    buffer.seek(0)
    twin = TwinNet()
    twin.load_state_dict(torch.load(buffer, weights_only=True))
    torch.testing.assert_close(twin(x), model(x))
    assert ht.is_tracked(restored)  # the declaration survives with the class


def test_tracking_an_instance_or_inspecting_it_adds_nothing_to_it():
    model = nn.Sequential(nn.Linear(2, 3), nn.ReLU(), nn.Linear(3, 1))
    attributes = set(vars(model))
    hooks = all_hooks(model)
    ht.track(model, name="seq")
    NetworkTracker(model).snapshot(0)
    ht.inspect_network(model)
    assert set(vars(model)) == attributes
    assert all_hooks(model) == hooks


# ---------------------------------------------------------- the step decorators
def test_step_decorators_return_the_users_functions_unchanged():
    process = build_process(native_steps=False).process

    def train_step(batch):
        return 1.0

    def validation_step(batch):
        return 2.0

    def hook(record):
        return None

    for decorator, function in (
        (process.train_step, train_step),
        (process.validation_step, validation_step),
        (process.on_epoch_end, hook),
    ):
        state = dict(vars(function))
        code = function.__code__
        assert decorator(function) is function
        assert dict(vars(function)) == state and function.__code__ is code
        assert not hasattr(function, "__wrapped__")
    assert train_step(None) == 1.0 and validation_step(None) == 2.0  # still plain functions


# --------------------------------------------------- managed == native training
def _native_run(epochs: int) -> tuple[dict, list[float], list[float]]:
    """The textbook PyTorch loop: what a user writes without this package."""
    torch.manual_seed(3)
    model = RichNet()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9)
    loss_fn = nn.CrossEntropyLoss()
    train = DataLoader(xor_data(96, 5), batch_size=16, shuffle=True)
    val = DataLoader(xor_data(40, 6), batch_size=20)
    train_losses, val_losses = [], []
    for _ in range(epochs):
        model.train()
        losses = []
        for inputs, targets in train:
            optimizer.zero_grad()
            loss = loss_fn(model(inputs), targets)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        train_losses.append(sum(losses) / len(losses))
        model.eval()
        with torch.no_grad():
            losses = [loss_fn(model(inputs), targets).item() for inputs, targets in val]
        val_losses.append(sum(losses) / len(losses))
    return model.state_dict(), train_losses, val_losses


def _managed_run(epochs: int, **run_options) -> tuple[dict, list[float], list[float]]:
    """The same loop, written as steps and handed to ht.run()."""
    torch.manual_seed(3)
    model = RichNet()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9)
    loss_fn = nn.CrossEntropyLoss()
    process = ht.TrainingProcess(
        model,
        optimizer,
        DataLoader(xor_data(96, 5), batch_size=16, shuffle=True),
        val_data=DataLoader(xor_data(40, 6), batch_size=20),
        loss_fn=loss_fn,
        epochs=epochs,
    )

    @process.train_step
    def train_step(batch):
        inputs, targets = batch
        optimizer.zero_grad()
        loss = loss_fn(model(inputs), targets)
        loss.backward()
        optimizer.step()
        return loss

    @process.validation_step
    def validation_step(batch):
        inputs, targets = batch
        return loss_fn(model(inputs), targets)

    result = ht.run(process, **run_options)
    return model.state_dict(), result.train_losses, result.val_losses


def _assert_same_training(native, managed):
    native_state, native_train, native_val = native
    managed_state, managed_train, managed_val = managed
    assert managed_train == pytest.approx(native_train, rel=1e-5, abs=1e-6)
    assert managed_val == pytest.approx(native_val, rel=1e-5, abs=1e-6)
    assert managed_state.keys() == native_state.keys()
    for key, value in native_state.items():
        torch.testing.assert_close(managed_state[key], value, rtol=1e-5, atol=1e-6)


def test_managed_training_is_numerically_the_native_training():
    """Same seed, data, model, optimizer: identical losses and final weights, including
    dropout randomness, shuffling and batch-norm statistics (no hidden RNG use)."""
    _assert_same_training(_native_run(4), _managed_run(4, gui=False))


@pytest.mark.gui
def test_managed_training_with_the_window_is_numerically_the_native_training(qapp):
    _assert_same_training(_native_run(3), _managed_run(3, isolate_gui=False, close_on_finish=True))


# ------------------------------------------------------------------- leftovers
def test_run_leaves_no_hooks_behind():
    bundle = build_process(epochs=2)
    hooks = all_hooks(bundle.model)
    global_before = global_hooks()
    ht.run(bundle.process, gui=False)
    assert all_hooks(bundle.model) == hooks
    assert global_hooks() == global_before


@pytest.mark.gui
def test_run_with_the_window_leaves_no_hooks_or_settings_behind(qapp):
    bundle = build_process(epochs=2)
    hooks = all_hooks(bundle.model)
    threads, grad, dtype = (
        torch.get_num_threads(),
        torch.is_grad_enabled(),
        torch.get_default_dtype(),
    )
    ht.run(bundle.process, isolate_gui=False, close_on_finish=True)
    assert all_hooks(bundle.model) == hooks
    assert torch.get_num_threads() == threads
    assert torch.is_grad_enabled() == grad and torch.get_default_dtype() == dtype


@pytest.mark.parametrize("setting", ["auto", 1, None])
@pytest.mark.gui
def test_the_thread_setting_is_restored_after_the_window_closes(qapp, setting):
    before = torch.get_num_threads()
    ht.run(build_process(epochs=1).process, isolate_gui=False, close_on_finish=True,
           torch_threads=setting)  # fmt: skip
    assert torch.get_num_threads() == before


def test_console_runs_do_not_touch_global_torch_settings():
    before = (torch.get_num_threads(), torch.is_grad_enabled(), torch.get_default_dtype())
    ht.run(build_process(epochs=2).process, gui=False)
    after = (torch.get_num_threads(), torch.is_grad_enabled(), torch.get_default_dtype())
    assert after == before


def test_run_works_on_the_users_own_objects():
    bundle = build_process(epochs=2)
    groups = [sorted(group) for group in bundle.optimizer.param_groups]
    parameters = [id(p) for p in bundle.model.parameters()]
    ht.run(bundle.process, gui=False)
    assert bundle.process.model is bundle.model
    assert bundle.process.optimizer is bundle.optimizer
    assert bundle.process.train_data is bundle.train_loader
    assert [sorted(group) for group in bundle.optimizer.param_groups] == groups
    assert [id(p) for p in bundle.model.parameters()] == parameters
    assert all(
        p.dtype == torch.float32 and p.device.type == "cpu" for p in bundle.model.parameters()
    )


def test_steps_receive_the_users_batches_untouched():
    batches = [(torch.randn(4, 2), torch.tensor([0, 1, 0, 1])) for _ in range(3)]
    received = []
    torch.manual_seed(0)
    model = TinyNet()

    def train_step(*args, **kwargs):
        received.append((args, kwargs))
        return 1.0

    process = ht.TrainingProcess(
        model, torch.optim.SGD(model.parameters(), lr=0.1), batches, epochs=2,
        train_step=train_step,
    )  # fmt: skip
    ht.run(process, gui=False)
    assert len(received) == 6
    for index, (args, kwargs) in enumerate(received):
        assert kwargs == {} and len(args) == 1
        assert args[0] is batches[index % 3]  # the very object the user's data yields


def test_double_precision_training_stays_double():
    torch.manual_seed(0)
    model = TinyNet().double()
    process = ht.TrainingProcess(
        model,
        torch.optim.SGD(model.parameters(), lr=0.1),
        [(torch.randn(4, 2, dtype=torch.float64), torch.tensor([0, 1, 1, 0]))],
        loss_fn=nn.CrossEntropyLoss(),
        epochs=2,
    )
    assert ht.run(process, gui=False).epochs_completed == 2
    assert all(p.dtype == torch.float64 for p in model.parameters())


_IMPORT_PROBE = textwrap.dedent(
    """
    import json, sys
    import torch, torch.optim, torch.utils.data

    targets = {
        "Module": torch.nn.Module, "Optimizer": torch.optim.Optimizer,
        "DataLoader": torch.utils.data.DataLoader, "Tensor": torch.Tensor,
        "Parameter": torch.nn.Parameter,
    }
    def fingerprint():
        return {name: {k: id(v) for k, v in vars(cls).items()} for name, cls in targets.items()}
    before = fingerprint()
    state = (torch.get_num_threads(), torch.is_grad_enabled(), str(torch.get_default_dtype()))
    trace = sys.gettrace(), sys.getprofile()

    import hollywood_training as ht
    for name in ("track", "run", "TrainingProcess", "NetworkTracker", "inspect_network"):
        getattr(ht, name)
    after = fingerprint()
    print(json.dumps({
        "changed": sorted(
            f"{name}.{key}" for name in targets
            for key in set(before[name]) | set(after[name])
            if before[name].get(key) != after[name].get(key)
        ),
        "state_same": state == (torch.get_num_threads(), torch.is_grad_enabled(),
                                str(torch.get_default_dtype())),
        "trace_same": trace == (sys.gettrace(), sys.getprofile()),
        "qt_loaded": sorted(m for m in sys.modules if m.startswith("PySide6")),
    }))
    """
)


def test_importing_the_package_changes_nothing_in_pytorch(tmp_path):
    script = tmp_path / "probe.py"
    script.write_text(_IMPORT_PROBE, encoding="utf-8")
    completed = run_python([str(script)], timeout=120)
    assert completed.returncode == 0, completed.stderr
    report = json.loads(completed.stdout.strip().splitlines()[-1])
    assert report["changed"] == []  # no monkey-patching of torch classes
    assert report["state_same"] and report["trace_same"]
    assert report["qt_loaded"] == []  # importing never starts or loads a GUI toolkit

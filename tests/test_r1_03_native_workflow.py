"""R1 requirement 3 - "**BOTH** requirements `1` and `2` **MUST NOT** significantly change the
native PyTorch workflow."

* tracking (requirement 1) leaves the network alone: the decorated class is the class
  itself, unchanged; no attribute, parameter, buffer or hook is added - not by tracking and
  not by a managed run; the network still pickles, deep-copies, saves / loads and scripts;
* training management (requirement 2) leaves the training alone: the step decorators return
  the user's function, which stays directly callable; the process uses the user's own
  objects; a managed run gives **the same result as the plain PyTorch loop** it wraps (same
  seed, shuffled data, native or default steps, console or window), consumes no random
  numbers of its own and restores PyTorch's thread setting.
"""

from __future__ import annotations

import copy
import io
import pickle

import pytest
import torch
from htspec.models import DeepNet, TinyNet, learnable_data
from torch import nn
from torch.utils.data import DataLoader

import hollywood_training as ht

pytestmark = pytest.mark.req("R1.3")


@ht.track(name="tracked for the workflow tests")
class TrackedNet(nn.Module):
    """Module level, so that pickling finds it by name."""

    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(2, 6)
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(6, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


HOOK_DICTS = (
    "_forward_hooks", "_forward_pre_hooks", "_backward_hooks", "_backward_pre_hooks",
    "_forward_hooks_with_kwargs", "_forward_pre_hooks_with_kwargs",
    "_forward_hooks_always_called", "_state_dict_hooks", "_state_dict_pre_hooks",
    "_load_state_dict_pre_hooks", "_load_state_dict_post_hooks",
)  # fmt: skip


def module_fingerprint(model: nn.Module) -> dict:
    """Attributes, parameters, buffers and registered hooks of every submodule."""
    print_ = {}
    for name, module in model.named_modules():
        hooks = {key: len(getattr(module, key, {}) or {}) for key in HOOK_DICTS}
        print_[name] = (sorted(vars(module)), hooks, module.training)
    print_["parameters"] = [
        (n, tuple(p.shape), p.requires_grad) for n, p in model.named_parameters()
    ]
    print_["buffers"] = [(n, tuple(b.shape)) for n, b in model.named_buffers()]
    print_["state_dict"] = list(model.state_dict())
    return print_


def global_hooks() -> dict:
    module = torch.nn.modules.module
    names = [n for n in dir(module) if n.startswith("_global_") and n.endswith("hooks")]
    return {name: len(getattr(module, name)) for name in names}


# ------------------------------------------------------------------- the network
def test_the_decorated_class_is_unchanged():
    class Plain(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.fc = nn.Linear(2, 2)

        def forward(self, x):
            return self.fc(x)

    namespace, bases, mro = dict(vars(Plain)), Plain.__bases__, Plain.__mro__
    assert ht.track(Plain) is Plain
    assert dict(vars(Plain)) == namespace and Plain.__bases__ == bases and Plain.__mro__ == mro
    model = Plain()
    assert type(model) is Plain and isinstance(model, nn.Module)


def test_tracking_adds_nothing_to_the_network():
    model = TinyNet()
    before = module_fingerprint(model)
    ht.track(model)
    ht.inspect_network(model)
    assert module_fingerprint(model) == before


def test_a_managed_run_leaves_no_trace_in_the_network():
    torch.manual_seed(0)
    model = DeepNet()
    model.train()
    before = module_fingerprint(model)
    hooks = global_hooks()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
    data = DataLoader(learnable_data(64), batch_size=16)
    data = [(x.repeat(1, 2), y % 2) for x, y in data]  # DeepNet: 4 inputs, 2 classes
    process = ht.TrainingProcess(model, optimizer, data, loss_fn=nn.CrossEntropyLoss(), epochs=2)
    ht.run(process, gui=False)
    model.train()  # the training mode is the user's to set again, as after any loop
    assert module_fingerprint(model) == before
    assert global_hooks() == hooks


def test_the_network_still_pickles_copies_saves_and_loads(tmp_path):
    model = TrackedNet()
    clone = pickle.loads(pickle.dumps(model))
    assert type(clone) is TrackedNet
    deep = copy.deepcopy(model)
    x = torch.randn(5, 2)
    assert torch.equal(clone(x), model(x)) and torch.equal(deep(x), model(x))
    path = tmp_path / "model.pt"
    torch.save(model.state_dict(), path)
    fresh = TrackedNet()
    fresh.load_state_dict(torch.load(path))
    assert torch.equal(fresh(x), model(x))
    buffer = io.BytesIO()
    torch.save(model, buffer)
    buffer.seek(0)
    loaded = torch.load(buffer, weights_only=False)
    assert type(loaded) is TrackedNet and torch.equal(loaded(x), model(x))


def test_the_network_still_scripts_like_an_untracked_one():
    class Twin(nn.Module):  # the same network, never tracked
        def __init__(self) -> None:
            super().__init__()
            self.fc1 = nn.Linear(2, 6)
            self.act = nn.ReLU()
            self.fc2 = nn.Linear(6, 3)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.fc2(self.act(self.fc1(x)))

    try:
        torch.jit.script(Twin())
    except Exception as exc:  # TorchScript itself unavailable in this PyTorch build
        pytest.skip(f"TorchScript does not work here at all: {exc}")
    scripted = torch.jit.script(TrackedNet())
    x = torch.randn(3, 2)
    assert scripted(x).shape == (3, 3)


# ------------------------------------------------------------------ the training
def test_the_decorated_step_is_the_users_function_and_still_trains_a_batch():
    torch.manual_seed(0)
    model = TinyNet()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    loss_fn = nn.CrossEntropyLoss()
    data = DataLoader(learnable_data(32), batch_size=8)
    process = ht.TrainingProcess(model, optimizer, data, loss_fn=loss_fn, epochs=1)

    def train_step(batch):
        optimizer.zero_grad()
        loss = loss_fn(model(batch[0]), batch[1])
        loss.backward()
        optimizer.step()
        return loss

    assert process.train_step(train_step) is train_step
    before = model.fc1.weight.detach().clone()
    loss = train_step(next(iter(data)))  # called directly, outside of any manager
    assert isinstance(loss, torch.Tensor)
    assert not torch.equal(model.fc1.weight, before)


def test_the_process_uses_the_users_own_objects():
    model = TinyNet()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    loss_fn = nn.CrossEntropyLoss()
    data = DataLoader(learnable_data(16), batch_size=8)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1)
    process = ht.TrainingProcess(
        model, optimizer, data, loss_fn=loss_fn, val_data=data, epochs=1, scheduler=scheduler
    )
    assert process.model is model and process.optimizer is optimizer
    assert process.train_data is data and process.val_data is data
    assert process.loss_fn is loss_fn and process.scheduler is scheduler


def plain_loop(epochs: int, *, validation: bool, seed: int = 0):
    """The training a user writes without the package (shuffled data, train + eval)."""
    torch.manual_seed(seed)
    model = DeepNet()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9)
    loss_fn = nn.CrossEntropyLoss()
    train = DataLoader(_four_feature_data(96, seed), batch_size=16, shuffle=True)
    val = DataLoader(_four_feature_data(32, seed + 1), batch_size=16)
    for _ in range(epochs):
        model.train()
        for inputs, targets in train:
            optimizer.zero_grad()
            loss = loss_fn(model(inputs), targets)
            loss.backward()
            optimizer.step()
        if validation:
            model.eval()
            with torch.no_grad():
                for inputs, targets in val:
                    loss_fn(model(inputs), targets)
    return model, torch.get_rng_state()


def _four_feature_data(samples: int, seed: int):
    generator = torch.Generator().manual_seed(seed)
    inputs = torch.randn(samples, 4, generator=generator)
    targets = (inputs[:, 0] + inputs[:, 1] > 0).long()
    return torch.utils.data.TensorDataset(inputs, targets)


def managed(epochs: int, *, validation: bool, native: bool, seed: int = 0, **run_options):
    torch.manual_seed(seed)
    model = DeepNet()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9)
    loss_fn = nn.CrossEntropyLoss()
    train = DataLoader(_four_feature_data(96, seed), batch_size=16, shuffle=True)
    val = DataLoader(_four_feature_data(32, seed + 1), batch_size=16) if validation else None
    process = ht.TrainingProcess(
        model, optimizer, train, loss_fn=loss_fn, val_data=val, epochs=epochs
    )
    if native:

        @process.train_step
        def train_step(batch):
            inputs, targets = batch
            optimizer.zero_grad()
            loss = loss_fn(model(inputs), targets)
            loss.backward()
            optimizer.step()
            return loss

        if validation:

            @process.validation_step
            def validation_step(batch):
                return loss_fn(model(batch[0]), batch[1])

    ht.run(process, **run_options)
    return model, torch.get_rng_state()


def assert_same_weights(a: nn.Module, b: nn.Module, *, exact: bool) -> None:
    for (name, p), (_name, q) in zip(a.state_dict().items(), b.state_dict().items(), strict=True):
        if exact:
            assert torch.equal(p, q), name
        else:
            assert torch.allclose(p.float(), q.float(), rtol=1e-4, atol=1e-6), name


@pytest.mark.parametrize("native", [True, False], ids=["native-steps", "default-steps"])
@pytest.mark.parametrize("validation", [True, False], ids=["with-validation", "no-validation"])
def test_a_managed_console_run_gives_exactly_the_plain_loops_result(native, validation):
    expected, expected_rng = plain_loop(4, validation=validation)
    model, rng = managed(4, validation=validation, native=native, gui=False)
    assert_same_weights(model, expected, exact=True)
    assert torch.equal(rng, expected_rng)  # the package drew no random numbers of its own


def test_a_managed_in_process_window_run_gives_the_plain_loops_result(qapp):
    expected, expected_rng = plain_loop(3, validation=True)
    model, rng = managed(
        3, validation=True, native=True, isolate_gui=False, close_on_finish=True,
        torch_threads=None,
    )  # fmt: skip
    assert_same_weights(model, expected, exact=True)
    assert torch.equal(rng, expected_rng)


@pytest.mark.subprocess
def test_a_managed_isolated_window_run_gives_the_plain_loops_result():
    expected, _rng = plain_loop(3, validation=True)
    model, _rng = managed(3, validation=True, native=True, close_on_finish=True)
    assert_same_weights(model, expected, exact=False)  # the window run may use fewer threads


def test_pytorchs_thread_setting_is_restored_after_a_window_run(qapp):
    before = torch.get_num_threads()
    managed(1, validation=False, native=True, isolate_gui=False, close_on_finish=True)
    assert torch.get_num_threads() == before

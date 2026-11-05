"""R1: the package automatically tracks the architecture of the defined network,
including the weights of each layer (a class decorator / slot dataclasses preferred)."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest
import torch
from torch import nn

import hollywood_training as ht
from hollywood_training.sync import EpochCompleted, LogMessage
from hollywood_training.tracking import (
    LayerRole,
    NetworkArchitecture,
    NetworkTracker,
    TrackedLayer,
    track_options,
)

from .helpers import ManagedRun, TinyNet, build_process

pytestmark = pytest.mark.spec("R1")


def weight_matrix(module: nn.Module) -> np.ndarray:
    return module.weight.detach().float().numpy()


# --------------------------------------------------------------------- declaring
def test_bare_class_decorator_tracks_every_instance():
    @ht.track
    class Net(TinyNet):
        pass

    assert ht.is_tracked(Net()) and ht.is_tracked(Net(hidden=3))
    assert not ht.is_tracked(TinyNet())  # the base class is not affected


def test_decorator_options_reach_the_tracked_architecture():
    @ht.track(name="XOR solver", input_labels=["x", "y"], output_labels=("no", "yes"))
    class Net(TinyNet):
        pass

    architecture = NetworkTracker(Net()).architecture
    assert architecture.class_name == "Net"
    assert architecture.display_name == "XOR solver"
    assert architecture.input_labels == ("x", "y")
    assert architecture.output_labels == ("no", "yes")


def test_subclasses_inherit_the_declaration():
    @ht.track(name="base")
    class Base(TinyNet):
        pass

    class Child(Base):
        pass

    assert ht.is_tracked(Child()) and track_options(Child()).name == "base"


def test_instance_form_tracks_models_the_user_did_not_define():
    model = nn.Sequential(nn.Linear(2, 3), nn.ReLU(), nn.Linear(3, 1))
    assert ht.track(model, name="seq") is model
    assert ht.is_tracked(model)
    assert not ht.is_tracked(nn.Sequential(nn.Linear(2, 3)))
    assert NetworkTracker(model).architecture.display_name == "seq"


def test_untracked_models_are_tracked_automatically_by_run():
    bundle = build_process(epochs=2)  # TinyNet carries no decorator
    assert not ht.is_tracked(bundle.model)
    result = ht.run(bundle.process, gui=False)
    assert result.final_weights is not None
    assert result.final_weights.names() == ("fc1", "fc2")


@pytest.mark.parametrize(
    "target",
    [int, object(), "net", 3.0, None.__class__],
    ids=["builtin-class", "object", "string", "float", "NoneType"],
)
def test_track_rejects_things_that_are_not_networks(target):
    with pytest.raises(TypeError):
        ht.track(target)


@pytest.mark.parametrize("argument", ["input_labels", "output_labels"])
def test_labels_must_be_a_sequence_not_a_single_string(argument):
    with pytest.raises(TypeError):
        ht.track(**{argument: "xy"})


def test_tracker_rejects_invalid_arguments():
    with pytest.raises(TypeError):
        NetworkTracker("not a module")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        NetworkTracker(TinyNet(), max_neurons=1)


# ------------------------------------------------------------------ architecture
def test_every_connection_layer_and_its_size_is_found():
    model = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 8), nn.Tanh(), nn.Linear(8, 3))
    layers = NetworkTracker(model).architecture.layers
    assert [(layer.name, layer.in_features, layer.out_features) for layer in layers] == [
        ("0", 4, 8),
        ("2", 8, 8),
        ("4", 8, 3),
    ]


def test_activation_functions_are_attached_to_the_layer_they_follow():
    info = NetworkTracker(TinyNet()).architecture.describe()
    assert [layer.activation for layer in info.layers] == ["Tanh", None]


def test_convolution_pooling_and_reshape_layers():
    model = nn.Sequential(
        nn.Conv2d(1, 4, 3), nn.ReLU(), nn.MaxPool2d(2), nn.Flatten(), nn.Linear(4 * 3 * 3, 5)
    )
    architecture = NetworkTracker(model).architecture
    conv, linear = architecture.layers
    assert (conv.in_features, conv.out_features) == (1, 4)
    assert (linear.in_features, linear.out_features) == (36, 5)
    assert [op.role for op in conv.operations] == [
        LayerRole.ACTIVATION,
        LayerRole.POOLING,
        LayerRole.RESHAPE,
    ]


def test_embedding_layers_are_connection_layers():
    model = nn.Sequential(nn.Embedding(10, 4), nn.Flatten(), nn.Linear(4 * 3, 2))
    first = NetworkTracker(model).architecture.layers[0]
    assert (first.in_features, first.out_features) == (10, 4)


def test_normalization_and_regularization_are_recorded():
    model = nn.Sequential(nn.Linear(2, 4), nn.BatchNorm1d(4), nn.ReLU(), nn.Dropout(0.3))
    info = NetworkTracker(model).architecture.describe()
    roles = [op.role for op in info.layers[0].operations]
    assert roles == [LayerRole.NORMALIZATION, LayerRole.ACTIVATION, LayerRole.REGULARIZATION]
    assert "BatchNorm1d" in info.normalization_summary()


def test_parameter_counts_distinguish_frozen_parameters():
    model = TinyNet()
    model.fc1.weight.requires_grad_(False)
    architecture = NetworkTracker(model).architecture
    total = sum(p.numel() for p in model.parameters())
    assert architecture.total_parameters == total
    assert architecture.trainable_parameters == total - model.fc1.weight.numel()


def test_layer_order_follows_execution_not_registration():
    class Shuffled(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.head = nn.Linear(4, 2)  # registered first, runs last
            self.body = nn.Linear(3, 4)
            self.act = nn.ReLU()

        def forward(self, x):
            return self.head(self.act(self.body(x)))

    model = Shuffled()
    process = ht.TrainingProcess(
        model,
        torch.optim.SGD(model.parameters(), lr=0.1),
        [(torch.randn(4, 3), torch.randn(4, 2))],
        epochs=1,
        loss_fn=nn.MSELoss(),
    )
    with ManagedRun(process) as run:
        run.join()
    shown = run.session.architecture  # what the frontend ends up drawing
    assert [layer.name for layer in shown.layers] == ["body", "head"]
    assert shown.layers[0].activation == "ReLU"


def test_lazy_layers_are_resolved_by_the_first_batch():
    model = nn.Sequential(nn.LazyLinear(5), nn.ReLU(), nn.Linear(5, 2))
    process = ht.TrainingProcess(
        model,
        torch.optim.SGD(model[2].parameters(), lr=0.1),  # lazy parameters come later
        [(torch.randn(4, 7), torch.randint(0, 2, (4,)))],
        epochs=2,
        loss_fn=nn.CrossEntropyLoss(),
    )
    with ManagedRun(process) as run:
        run.join()
    first = run.session.architecture.layers[0]
    assert (first.in_features, first.out_features) == (7, 5)
    epochs = run.frontend.of_type(EpochCompleted)
    assert epochs[-1].weights.names() == ("0", "2")


def test_a_module_used_twice_is_one_layer():
    class Recurrent(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.inp = nn.Linear(3, 4)
            self.shared = nn.Linear(4, 4)

        def forward(self, x):
            return self.shared(torch.tanh(self.shared(self.inp(x))))

    tracker = NetworkTracker(Recurrent())
    tracker.begin_trace()
    tracker.model(torch.randn(2, 3))
    tracker.end_trace()
    assert [layer.name for layer in tracker.architecture.layers] == ["inp", "shared"]


def test_wrappers_are_looked_through():
    @ht.track(name="inner")
    class Inner(TinyNet):
        pass

    architecture = NetworkTracker(nn.DataParallel(Inner())).architecture
    assert architecture.class_name == "Inner"
    assert architecture.display_name == "inner"
    assert [layer.name for layer in architecture.layers] == ["fc1", "fc2"]


def test_compiled_models_are_tracked_as_the_user_network():
    """``torch.compile`` is part of the native PyTorch 2 workflow; the tracked network must
    still be the user's class with its own layer names (QA finding M4)."""

    @ht.track(name="compiled")
    class Net(TinyNet):
        pass

    try:
        compiled = torch.compile(Net(), backend="eager")
    except Exception as exc:  # pragma: no cover - platform without torch.compile
        pytest.skip(f"torch.compile unavailable: {exc}")
    architecture = NetworkTracker(compiled).architecture
    assert architecture.class_name == "Net"
    assert architecture.display_name == "compiled"
    assert [layer.name for layer in architecture.layers] == ["fc1", "fc2"]


def test_a_network_without_connection_layers_is_handled():
    tracker = NetworkTracker(nn.Sequential(nn.ReLU(), nn.Tanh()))
    assert tracker.architecture.layers == []
    assert tracker.snapshot(0).layers == ()


def _accounted_parameters(model: nn.Module, architecture: NetworkArchitecture) -> set[int]:
    modules = dict(model.named_modules())
    owners = [layer.module for layer in architecture.layers]
    owners += [op.module for layer in architecture.layers for op in layer.operations]
    owners += [op.module for op in architecture.input_operations]
    owners += [modules[name] for name in architecture.unsupported if name in modules]
    return {id(p) for owner in owners for p in owner.parameters()}


@pytest.mark.parametrize(
    "factory",
    [
        lambda: TinyNet(),
        lambda: nn.Sequential(nn.Conv1d(2, 3, 3), nn.BatchNorm1d(3), nn.PReLU(), nn.Flatten()),
        lambda: nn.Sequential(nn.Embedding(9, 3), nn.Flatten(), nn.Linear(6, 2)),
        lambda: nn.Sequential(nn.Linear(4, 4), nn.LayerNorm(4), nn.GRU(4, 3)),
        lambda: nn.TransformerEncoderLayer(d_model=8, nhead=2, dim_feedforward=16),
    ],
    ids=["mlp", "conv-bn-prelu", "embedding", "recurrent", "transformer"],
)
def test_no_layer_weights_are_silently_dropped(factory):
    """Every parameter is either tracked (layer or operation) or explicitly reported as
    not drawn - never silently missing (QA finding L1)."""
    model = factory()
    architecture = NetworkTracker(model).architecture
    missing = [
        name
        for name, parameter in model.named_parameters()
        if id(parameter) not in _accounted_parameters(model, architecture)
    ]
    assert missing == []


def test_unsupported_layers_are_reported_to_the_user():
    model = nn.Sequential(nn.Linear(4, 4), nn.LSTM(4, 3))
    assert NetworkTracker(model).architecture.describe().unsupported == ("1",)


# ----------------------------------------------------------------------- weights
def test_snapshot_holds_the_real_weights_and_biases_of_every_layer():
    model = TinyNet()
    snapshot = NetworkTracker(model).snapshot(0)
    assert snapshot.names() == ("fc1", "fc2")
    for name in ("fc1", "fc2"):
        layer, module = snapshot.layer(name), getattr(model, name)
        np.testing.assert_allclose(layer.matrix, weight_matrix(module))
        np.testing.assert_allclose(layer.bias, module.bias.detach().numpy())
        assert (layer.in_features, layer.out_features) == (module.in_features, module.out_features)


def test_layers_without_bias():
    model = nn.Sequential(nn.Linear(3, 2, bias=False))
    layer = NetworkTracker(model).snapshot(0).layers[0]
    assert layer.bias is None
    np.testing.assert_allclose(layer.matrix, weight_matrix(model[0]))


def test_snapshots_follow_the_real_training_at_every_epoch():
    bundle = build_process(epochs=4)
    with ManagedRun(bundle.process) as run:
        run.join()
    epochs = run.frontend.of_type(EpochCompleted)
    assert [event.weights.epoch for event in epochs] == [1, 2, 3, 4]
    for event in epochs:
        real = bundle.weights[event.record.epoch]
        np.testing.assert_allclose(event.weights.layer("fc1").matrix, real["fc1.weight"].numpy())
        np.testing.assert_allclose(event.weights.layer("fc2").bias, real["fc2.bias"].numpy())
    first, last = epochs[0].weights.layer("fc1").matrix, epochs[-1].weights.layer("fc1").matrix
    assert not np.allclose(first, last)  # training really changed what is tracked


def test_each_snapshot_records_the_change_since_the_previous_epoch():
    model = TinyNet()
    tracker = NetworkTracker(model)
    before = tracker.snapshot(0).layer("fc1").matrix.copy()
    with torch.no_grad():
        model.fc1.weight.add_(0.5)
    after = tracker.snapshot(1).layer("fc1")
    np.testing.assert_allclose(after.delta, after.matrix - before, atol=1e-6)
    assert after.update_ratio > 0


def test_wide_layers_are_sampled_but_statistics_cover_the_full_tensor():
    model = nn.Sequential(nn.Linear(3, 100), nn.ReLU(), nn.Linear(100, 2))
    tracker = NetworkTracker(model, max_neurons=10)
    hidden, output = tracker.snapshot(0).layers
    assert hidden.matrix.shape == (10, 3) and output.matrix.shape == (2, 10)
    assert hidden.out_index[0] == 0 and hidden.out_index[-1] == 99
    assert np.array_equal(hidden.out_index, output.in_index)  # adjacent columns line up
    full = model[0].weight.detach()
    np.testing.assert_allclose(hidden.matrix, full.numpy()[hidden.out_index])
    assert hidden.stats.count == 300
    assert hidden.stats.mean == pytest.approx(float(full.mean()), abs=1e-6)
    assert hidden.stats.maximum == pytest.approx(float(full.max()))
    assert hidden.truncated and output.truncated


def test_the_smallest_neuron_limit_keeps_the_first_and_last_neuron():
    model = nn.Sequential(nn.Linear(5, 7))
    layer = NetworkTracker(model, max_neurons=2).snapshot(0).layers[0]
    assert layer.in_index.tolist() == [0, 4] and layer.out_index.tolist() == [0, 6]


def test_convolution_connections_are_kernel_means_of_the_real_weights():
    model = nn.Sequential(nn.Conv2d(2, 3, 3))
    layer = NetworkTracker(model).snapshot(0).layers[0]
    expected = model[0].weight.detach().mean(dim=(2, 3)).numpy()
    np.testing.assert_allclose(layer.matrix, expected, rtol=1e-5, atol=1e-6)


def test_snapshots_are_immutable_copies():
    model = TinyNet()
    layer = NetworkTracker(model).snapshot(0).layer("fc1")
    copy = layer.matrix.copy()
    with torch.no_grad():
        model.fc1.weight.mul_(-3.0)
    np.testing.assert_array_equal(layer.matrix, copy)
    with pytest.raises(ValueError):
        layer.matrix[0, 0] = 1.0


def test_double_precision_models_are_tracked():
    model = TinyNet().double()
    layer = NetworkTracker(model).snapshot(0).layer("fc1")
    np.testing.assert_allclose(layer.matrix, model.fc1.weight.detach().numpy(), rtol=1e-6)


def test_errors_in_the_traced_forward_pass_are_reported_unchanged():
    """A model whose forward pass fails on the first batch reports the error; tracking
    must not mask it with an error of its own."""

    class Broken(TinyNet):
        def forward(self, x):
            raise ValueError("broken forward")

    model = Broken()
    process = ht.TrainingProcess(
        model,
        torch.optim.SGD(model.parameters(), lr=0.1),
        [(torch.randn(2, 2), torch.tensor([0, 1]))],
        epochs=1,
        loss_fn=nn.CrossEntropyLoss(),
    )
    with ManagedRun(process) as run:
        result = run.join()
    assert isinstance(result.error, ValueError) and "broken forward" in str(result.error)
    assert any("broken forward" in m.text for m in run.frontend.of_type(LogMessage))


# ------------------------------------------------------- preferred mechanisms
def test_architecture_is_held_in_slot_dataclasses():
    """SPEC prefers slot dataclasses holding the linear components and activations."""
    for cls in (NetworkArchitecture, TrackedLayer):
        assert dataclasses.is_dataclass(cls) and hasattr(cls, "__slots__")
    layer = NetworkTracker(TinyNet()).architecture.layers[0]
    assert isinstance(layer.module, nn.Linear)  # the linear component itself
    assert isinstance(layer.operations[0].module, nn.Tanh)  # its activation function

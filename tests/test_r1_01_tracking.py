"""R1 requirement 1 - "The package **MUST** provide a method to automatically track the
architecture of the defined network, including the weights of each layers. The method
**CAN** be **ANY** Python objects or mechanisms, but a `decorator` applys to the class of
the network or a `slot dataclass` designed to contain the network including the linear
components and activation functions may be preferred."

* the declaration: ``@ht.track`` on a class (bare and with options), ``ht.track(instance)``,
  inheritance, argument checks - and networks passed to the manager without any declaration
  are tracked automatically;
* the architecture: every connection layer in execution order with its in/out sizes and
  parameter count, the activation / normalisation / dropout / ... operations attached to the
  right layer, the order traced from a real forward pass, shared and lazy layers, wrappers,
  nested containers, parameter-free networks;
* the weights: snapshots hold the *real* weights (Linear, Conv incl. grouped, ConvTranspose,
  Embedding), statistics of the full tensors, evenly sampled wide layers, read-only
  independent copies, the change since the previous epoch;
* automatic per-epoch tracking in a managed run;
* boundary: every weight matrix of common ``torch.nn`` layer types must be tracked.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest
import torch
from htspec.headless import HeadlessRun
from htspec.models import (
    DeepNet,
    ReorderedNet,
    SharedNet,
    TinyNet,
    WideNet,
    make_setup,
    weights_of,
)
from torch import nn

import hollywood_training as ht
from hollywood_training.sync.events import EpochCompleted, SessionStarted
from hollywood_training.tracking import LayerRole, display_indices
from hollywood_training.tracking.tracker import NetworkTracker

pytestmark = pytest.mark.req("R1.1")


def as_numpy(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().to(torch.float32).cpu().numpy()


# --------------------------------------------------------------------- declaration
def test_the_class_decorator_returns_the_class_itself():
    class Net(TinyNet):
        pass

    assert ht.track(Net) is Net
    assert ht.is_tracked(Net())


def test_the_decorator_takes_display_options():
    @ht.track(name="My network", input_labels=("a", "b"), output_labels=["x", "y", "z"])
    class Labelled(TinyNet):
        pass

    info = ht.inspect_network(Labelled()).describe()
    assert info.class_name == "Labelled"
    assert info.display_name == "My network"
    assert info.input_labels == ("a", "b")
    assert info.output_labels == ("x", "y", "z")


def test_tracking_an_instance_returns_the_instance():
    model = nn.Sequential(nn.Linear(2, 3), nn.ReLU(), nn.Linear(3, 1))
    assert ht.track(model, name="sequential") is model
    assert ht.is_tracked(model)
    assert ht.inspect_network(model).display_name == "sequential"
    assert not ht.is_tracked(nn.Sequential(nn.Linear(2, 3)))  # only that instance


def test_subclasses_inherit_the_declaration():
    @ht.track(name="base")
    class Base(TinyNet):
        pass

    class Child(Base):
        pass

    assert ht.is_tracked(Child())
    assert ht.inspect_network(Child()).display_name == "base"


def test_labels_must_be_a_sequence_of_strings_not_one_string():
    with pytest.raises(TypeError):
        ht.track(input_labels="xy")
    with pytest.raises(TypeError):
        ht.track(TinyNet(), output_labels="abc")


@pytest.mark.parametrize("target", [object(), print, 3, "net", type("NotAModule", (), {})])
def test_only_networks_can_be_tracked(target):
    with pytest.raises(TypeError):
        ht.track(target)


def test_a_network_passed_without_any_declaration_is_tracked_automatically():
    setup = make_setup(epochs=1)
    assert not ht.is_tracked(setup.model)
    run = HeadlessRun(setup.process)
    run.run()
    started = run.events(SessionStarted)[0]
    assert [layer.name for layer in started.info.architecture.layers] == ["fc1", "fc2"]
    assert started.weights.names() == ("fc1", "fc2")


# -------------------------------------------------------------------- architecture
def test_every_connection_layer_is_listed_with_its_sizes_and_parameters():
    model = DeepNet(hidden=12)
    info = NetworkTracker(model).architecture.describe()
    assert info.class_name == "DeepNet"
    assert [layer.name for layer in info.layers] == ["fc1", "fc2", "fc3"]
    sizes = [(layer.in_features, layer.out_features) for layer in info.layers]
    assert sizes == [(4, 12), (12, 12), (12, 2)]
    for layer in info.layers:
        module = getattr(model, layer.name)
        assert layer.parameters == sum(p.numel() for p in module.parameters())
        assert layer.type_name == "Linear"


def test_operations_are_attached_to_the_layer_whose_output_they_process():
    info = NetworkTracker(DeepNet()).architecture.describe()
    operations = {
        layer.name: [(op.name, op.role, op.type_name) for op in layer.operations]
        for layer in info.layers
    }
    assert operations == {
        "fc1": [
            ("norm", LayerRole.NORMALIZATION, "BatchNorm1d"),
            ("act1", LayerRole.ACTIVATION, "ReLU"),
        ],
        "fc2": [
            ("act2", LayerRole.ACTIVATION, "Tanh"),
            ("drop", LayerRole.REGULARIZATION, "Dropout"),
        ],
        "fc3": [],
    }
    assert info.normalization_layers() == ("BatchNorm1d",)


def test_parameter_counts_cover_the_whole_network_and_frozen_parts():
    model = DeepNet()
    for parameter in model.fc1.parameters():
        parameter.requires_grad_(False)
    info = NetworkTracker(model).architecture.describe()
    assert info.total_parameters == sum(p.numel() for p in model.parameters())
    assert info.trainable_parameters == sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    assert info.trainable_parameters < info.total_parameters


def test_the_execution_order_of_a_real_forward_pass_is_adopted():
    """Modules registered in another order than they run (out, act, inp)."""
    setup = make_setup(model=ReorderedNet(), epochs=1)
    static = NetworkTracker(setup.model).architecture.describe()
    assert [layer.name for layer in static.layers] == ["out", "inp"]
    run = HeadlessRun(setup.process)
    run.run()
    refined = run.events(EpochCompleted)[0].architecture
    assert refined is not None and refined.traced
    assert [layer.name for layer in refined.layers] == ["inp", "out"]
    assert [op.name for op in refined.layers[0].operations] == ["act"]
    assert run.session.architecture.layers[0].name == "inp"


def test_a_layer_used_twice_is_tracked_once():
    model = SharedNet()
    tracker = NetworkTracker(model)
    tracker.begin_trace()
    model(torch.randn(3, 2))
    tracker.end_trace()
    layers = tracker.architecture.describe().layers
    assert [layer.name for layer in layers] == ["fc"]
    assert [op.name for op in layers[0].operations] == ["act"]


def test_lazy_layers_are_resolved_by_the_first_forward_pass():
    model = nn.Sequential(nn.LazyLinear(5), nn.ReLU(), nn.Linear(5, 2))
    tracker = NetworkTracker(model)
    first = tracker.architecture.describe().layers[0]
    assert (first.in_features, first.out_features) == (None, None)
    assert "uninitialised" in first.summary
    assert "0" not in tracker.snapshot(0).names()
    tracker.begin_trace()
    model(torch.randn(4, 3))
    assert tracker.end_trace() is True
    first = tracker.architecture.describe().layers[0]
    assert (first.in_features, first.out_features) == (3, 5)
    weights = tracker.snapshot(0).layer("0")
    assert weights is not None and weights.matrix.shape == (5, 3)
    np.testing.assert_array_equal(weights.matrix, as_numpy(model[0].weight))


def test_wrappers_are_looked_through():
    inner = TinyNet()
    info = NetworkTracker(nn.DataParallel(inner)).architecture.describe()
    assert info.class_name == "TinyNet"
    assert [layer.name for layer in info.layers] == ["fc1", "fc2"]


def test_nested_containers_and_identity_modules():
    model = nn.Sequential(
        nn.Linear(2, 4), nn.Identity(), nn.Sequential(nn.ReLU(), nn.Linear(4, 3)), nn.Softmax(1)
    )
    info = NetworkTracker(model).architecture.describe()
    assert [layer.name for layer in info.layers] == ["0", "2.1"]
    assert [op.name for op in info.layers[0].operations] == ["2.0"]  # Identity is skipped
    assert [op.type_name for op in info.layers[1].operations] == ["Softmax"]


def test_networks_without_connection_layers():
    tracker = NetworkTracker(nn.Sequential(nn.ReLU(), nn.Tanh()))
    info = tracker.architecture.describe()
    assert info.layers == ()
    assert [op.type_name for op in info.input_operations] == ["ReLU", "Tanh"]
    assert info.total_parameters == 0
    assert tracker.snapshot(0).layers == ()


# -------------------------------------------------------------------------- weights
def test_snapshots_hold_the_real_weights_and_biases():
    model = TinyNet()
    snapshot = NetworkTracker(model).snapshot(0)
    for name in ("fc1", "fc2"):
        layer = snapshot.layer(name)
        module = getattr(model, name)
        np.testing.assert_array_equal(layer.matrix, as_numpy(module.weight))
        np.testing.assert_array_equal(layer.bias, as_numpy(module.bias))
        assert list(layer.out_index) == list(range(module.out_features))
        assert list(layer.in_index) == list(range(module.in_features))
        assert (layer.out_features, layer.in_features) == tuple(module.weight.shape)


def test_layers_without_bias_have_no_bias():
    model = nn.Sequential(nn.Linear(3, 2, bias=False))
    assert NetworkTracker(model).snapshot(0).layer("0").bias is None


@pytest.mark.parametrize("groups", [1, 2])
def test_convolution_weights_are_the_kernel_means(groups):
    conv = nn.Conv2d(4, 6, kernel_size=3, groups=groups)
    layer = NetworkTracker(nn.Sequential(conv)).snapshot(0).layer("0")
    assert layer.matrix.shape == (6, 4)
    weight = conv.weight.detach()  # (out, in / groups, k, k)
    in_per_group, out_per_group = 4 // groups, 6 // groups
    for j in range(6):
        for i in range(4):
            same_group = j // out_per_group == i // in_per_group
            expected = float(weight[j, i % in_per_group].mean()) if same_group else 0.0
            assert layer.matrix[j, i] == pytest.approx(expected, abs=1e-6), (j, i)


def test_transposed_convolution_weights_are_turned_back_to_out_by_in():
    conv = nn.ConvTranspose2d(3, 2, kernel_size=3)
    layer = NetworkTracker(nn.Sequential(conv)).snapshot(0).layer("0")
    expected = as_numpy(conv.weight.mean(dim=(2, 3)).T)  # weight is (in, out, k, k)
    np.testing.assert_allclose(layer.matrix, expected, atol=1e-6)


def test_embedding_weights_are_the_transposed_table():
    embedding = nn.Embedding(10, 4)
    layer = NetworkTracker(nn.Sequential(embedding)).snapshot(0).layer("0")
    assert (layer.in_features, layer.out_features) == (10, 4)
    np.testing.assert_array_equal(layer.matrix, as_numpy(embedding.weight).T)


def test_statistics_cover_the_full_weight_tensor():
    model = WideNet(200)
    stats = NetworkTracker(model, max_neurons=20).snapshot(0).layer("fc1").stats
    weight = model.fc1.weight.detach()
    assert stats.count == weight.numel() == 600
    assert stats.mean == pytest.approx(float(weight.mean()), abs=1e-6)
    assert stats.std == pytest.approx(float(weight.std(correction=0)), abs=1e-6)
    assert stats.minimum == pytest.approx(float(weight.min()))
    assert stats.maximum == pytest.approx(float(weight.max()))
    assert stats.abs_mean == pytest.approx(float(weight.abs().mean()), abs=1e-6)
    assert stats.norm == pytest.approx(float(weight.norm()), rel=1e-5)


def test_wide_layers_are_sampled_evenly_and_consistently():
    model = WideNet(200)
    snapshot = NetworkTracker(model, max_neurons=20).snapshot(0)
    fc1, fc2 = snapshot.layer("fc1"), snapshot.layer("fc2")
    expected = display_indices(200, 20)
    assert list(fc1.out_index) == list(expected)
    assert len(expected) == 20 and expected[0] == 0 and expected[-1] == 199
    assert all(b > a for a, b in itertools.pairwise(expected))
    assert list(fc2.in_index) == list(fc1.out_index)  # adjacent layers line up
    assert fc1.truncated and fc2.truncated
    np.testing.assert_array_equal(fc1.matrix, as_numpy(model.fc1.weight)[expected])


def test_the_neuron_limit_boundaries():
    model = WideNet(200)
    layer = NetworkTracker(model, max_neurons=2).snapshot(0).layer("fc1")
    assert list(layer.out_index) == [0, 199]
    with pytest.raises(ValueError):
        NetworkTracker(model, max_neurons=1)


def test_snapshots_are_read_only_copies_that_later_training_cannot_change():
    model = TinyNet()
    layer = NetworkTracker(model).snapshot(0).layer("fc1")
    kept = layer.matrix.copy()
    assert not layer.matrix.flags.writeable
    with pytest.raises(ValueError):
        layer.matrix[0, 0] = 1.0
    with torch.no_grad():
        model.fc1.weight.add_(1.0)
    np.testing.assert_array_equal(layer.matrix, kept)


def test_each_snapshot_carries_the_change_since_the_previous_epoch():
    model = TinyNet()
    tracker = NetworkTracker(model)
    first = tracker.snapshot(0).layer("fc1")
    assert first.delta is None
    with torch.no_grad():
        model.fc1.weight.add_(0.25)
    peek = tracker.snapshot(0, baseline=False).layer("fc1")  # an intermediate look
    np.testing.assert_allclose(peek.delta, peek.matrix - first.matrix, atol=1e-6)
    with torch.no_grad():
        model.fc1.weight.add_(0.25)
    second = tracker.snapshot(1).layer("fc1")
    np.testing.assert_allclose(second.delta, second.matrix - first.matrix, atol=1e-6)
    expected_ratio = np.linalg.norm(second.delta) / np.linalg.norm(first.matrix)
    assert second.update_ratio == pytest.approx(expected_ratio, rel=1e-4)


# --------------------------------------------------------------------- managed runs
def test_every_epoch_carries_the_real_weights_of_that_epoch():
    setup = make_setup(epochs=4)
    copies: dict[int, dict] = {}

    @setup.process.on_epoch_end
    def keep_weights(record):
        copies[record.epoch] = weights_of(setup.model)

    run = HeadlessRun(setup.process)
    run.run()
    events = run.events(EpochCompleted)
    assert [event.record.epoch for event in events] == [1, 2, 3, 4]
    for event in events:
        assert event.weights.epoch == event.record.epoch
        real = copies[event.record.epoch]
        for name in ("fc1", "fc2"):
            layer = event.weights.layer(name)
            np.testing.assert_array_equal(layer.matrix, real[f"{name}.weight"].numpy())
            np.testing.assert_array_equal(layer.bias, real[f"{name}.bias"].numpy())
    first, last = events[0].weights.layer("fc1"), events[-1].weights.layer("fc1")
    assert not np.array_equal(first.matrix, last.matrix)  # the weights really changed


def test_the_result_holds_the_final_weights():
    setup = make_setup(epochs=2)
    result = ht.run(setup.process, gui=False)
    final = result.final_weights.layer("fc2")
    np.testing.assert_array_equal(final.matrix, as_numpy(setup.model.fc2.weight))


# ------------------------------------------------------------- boundary: layer types
class Holder(nn.Module):
    def __init__(self, layer: nn.Module) -> None:
        super().__init__()
        self.layer = layer


LAYER_TYPES = {
    "Linear": lambda: nn.Linear(4, 3),
    "Conv1d": lambda: nn.Conv1d(2, 3, 3),
    "Conv2d": lambda: nn.Conv2d(2, 3, 3),
    "Conv3d": lambda: nn.Conv3d(2, 3, 3),
    "ConvTranspose2d": lambda: nn.ConvTranspose2d(2, 3, 3),
    "Embedding": lambda: nn.Embedding(10, 4),
    "EmbeddingBag": lambda: nn.EmbeddingBag(10, 4),
    "Bilinear": lambda: nn.Bilinear(3, 4, 2),
    "RNN": lambda: nn.RNN(3, 5),
    "LSTM": lambda: nn.LSTM(3, 5),
    "GRU": lambda: nn.GRU(3, 5),
    "MultiheadAttention": lambda: nn.MultiheadAttention(8, 2),
    "TransformerEncoderLayer": lambda: nn.TransformerEncoderLayer(8, 2, dim_feedforward=16),
}


@pytest.mark.parametrize("kind", list(LAYER_TYPES))
def test_every_weight_matrix_of_the_network_is_tracked(kind):
    """Strict reading of "including the weights of each layers": every weight tensor of a
    layer (a parameter with two or more dimensions) must appear in the tracked weights."""
    model = Holder(LAYER_TYPES[kind]())
    tracked = set(NetworkTracker(model).snapshot(0).names())
    untracked = [
        name
        for name, parameter in model.named_parameters()
        if parameter.dim() >= 2 and name.rsplit(".", 1)[0] not in tracked
    ]
    assert untracked == [], f"{kind}: weights not tracked: {untracked} (tracked: {sorted(tracked)})"

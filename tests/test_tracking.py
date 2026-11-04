"""Architecture discovery and weight snapshots."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from torch import nn

from hollywood_training.tracking import (
    LayerRole,
    NetworkTracker,
    build_architecture,
    classify,
    connection_matrix,
    connection_shape,
    display_indices,
    inspect_network,
    is_tracked,
    track,
    track_options,
)


# ------------------------------------------------------------------- decorator
def test_track_bare_decorator_returns_class_unchanged():
    class Net(nn.Linear):
        pass

    assert track(Net) is Net
    assert is_tracked(Net(2, 3))
    assert track_options(Net(2, 3)).name is None


def test_track_with_options_and_inheritance():
    @track(name="demo", input_labels=["a", "b"], output_labels=("y",))
    class Base(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.fc = nn.Linear(2, 1)

    class Child(Base):
        pass

    options = track_options(Child())
    assert options.name == "demo"
    assert options.input_labels == ("a", "b")
    assert options.output_labels == ("y",)


def test_track_instance_form_and_untracked_models():
    plain = nn.Sequential(nn.Linear(2, 2))
    assert not is_tracked(plain)
    assert track(plain) is plain
    assert is_tracked(plain)
    assert not is_tracked(nn.Sequential(nn.Linear(2, 2)))  # other instances unaffected


@pytest.mark.parametrize("target", [int, object(), "net"])
def test_track_rejects_non_modules(target):
    with pytest.raises(TypeError):
        track(target)


def test_track_rejects_single_string_labels():
    with pytest.raises(TypeError):
        track(input_labels="xy")


# ---------------------------------------------------------------- classification
@pytest.mark.parametrize(
    ("module", "role"),
    [
        (nn.Linear(2, 3), LayerRole.CONNECTION),
        (nn.Conv2d(3, 4, 3), LayerRole.CONNECTION),
        (nn.Embedding(5, 2), LayerRole.CONNECTION),
        (nn.ReLU(), LayerRole.ACTIVATION),
        (nn.BatchNorm1d(3), LayerRole.NORMALIZATION),
        (nn.LayerNorm(3), LayerRole.NORMALIZATION),
        (nn.Dropout(0.1), LayerRole.REGULARIZATION),
        (nn.MaxPool2d(2), LayerRole.POOLING),
        (nn.Flatten(), LayerRole.RESHAPE),
        (nn.LSTM(2, 3), LayerRole.OTHER),
    ],
)
def test_classify(module, role):
    assert classify(module) is role


def test_custom_linear_like_module_is_a_connection():
    class Projection(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = nn.Parameter(torch.randn(4, 3))

    module = Projection()
    assert classify(module) is LayerRole.CONNECTION
    assert connection_shape(module) == (3, 4)


# -------------------------------------------------------------------- architecture
def test_build_architecture_groups_operations():
    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(4, 8),
        nn.BatchNorm1d(8),
        nn.ReLU(),
        nn.Identity(),
        nn.Linear(8, 2),
        nn.LSTM(2, 2),
    )
    architecture = build_architecture(model)
    assert architecture.class_name == "Sequential"
    assert [op.summary for op in architecture.input_operations] == ["Flatten"]
    assert [layer.name for layer in architecture.layers] == ["1", "5"]
    assert [op.type_name for op in architecture.layers[0].operations] == ["BatchNorm1d", "ReLU"]
    assert architecture.layers[0].summary == "Linear 4→8"
    assert architecture.unsupported == ["6"]
    info = architecture.describe()
    assert info.normalization_summary() == "BatchNorm1d"
    assert info.layers[0].activation == "ReLU"
    assert info.total_parameters == sum(p.numel() for p in model.parameters())


def test_trace_follows_execution_order_and_reused_activations():
    class Net(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.out = nn.Linear(4, 2)  # registered first, runs last
            self.act = nn.ReLU()  # one module, used twice
            self.fc1 = nn.Linear(3, 4)
            self.fc2 = nn.Linear(4, 4)

        def forward(self, x):
            return self.out(self.act(self.fc2(self.act(self.fc1(x)))))

    model = Net()
    tracker = NetworkTracker(model)
    assert [layer.name for layer in tracker.architecture.layers] == ["out", "fc1", "fc2"]
    tracker.begin_trace()
    model(torch.randn(2, 3))
    model(torch.randn(2, 3))  # only the first pass is recorded
    assert tracker.end_trace() is True
    layers = tracker.architecture.layers
    assert [layer.name for layer in layers] == ["fc1", "fc2", "out"]
    assert [[op.type_name for op in layer.operations] for layer in layers] == [
        ["ReLU"],
        ["ReLU"],
        [],
    ]
    assert tracker.architecture.traced


def test_lazy_modules_resolve_after_trace():
    model = nn.Sequential(nn.LazyLinear(5), nn.ReLU(), nn.Linear(5, 2))
    tracker = NetworkTracker(model)
    assert tracker.architecture.layers[0].in_features is None
    assert tracker.snapshot(0).names() == ("2",)  # uninitialised layer is skipped
    tracker.begin_trace()
    model(torch.randn(3, 7))
    assert tracker.end_trace() is True
    first = tracker.architecture.layers[0]
    assert (first.in_features, first.out_features) == (7, 5)
    assert tracker.snapshot(1).names() == ("0", "2")


def test_data_parallel_is_looked_through():
    inner = nn.Sequential(nn.Linear(2, 3))
    assert inspect_network(nn.DataParallel(inner)).class_name == "Sequential"


# ---------------------------------------------------------------------- snapshots
def test_display_indices():
    assert display_indices(5, 10).tolist() == [0, 1, 2, 3, 4]
    sampled = display_indices(100, 5)
    assert sampled[0] == 0 and sampled[-1] == 99 and len(sampled) == 5
    assert display_indices(0, 5).size == 0


def test_snapshot_matches_weights_and_tracks_changes():
    model = nn.Sequential(nn.Linear(3, 30), nn.ReLU(), nn.Linear(30, 2))
    tracker = NetworkTracker(model, max_neurons=8)
    first = tracker.snapshot(0)
    hidden, output = first.layers
    assert hidden.matrix.shape == (8, 3) and output.matrix.shape == (2, 8)
    assert np.array_equal(hidden.out_index, output.in_index)  # shared column lines up
    assert hidden.truncated and output.truncated  # 30 hidden neurons, 8 drawn
    assert not NetworkTracker(nn.Linear(3, 2)).snapshot(0).layers[0].truncated
    expected = model[0].weight.detach().numpy()[hidden.out_index][:, hidden.in_index]
    np.testing.assert_allclose(hidden.matrix, expected)
    np.testing.assert_allclose(hidden.bias, model[0].bias.detach().numpy()[hidden.out_index])
    assert hidden.delta is None and hidden.update_ratio is None
    assert hidden.stats.count == 90
    assert hidden.stats.mean == pytest.approx(float(model[0].weight.detach().mean()), abs=1e-6)
    with pytest.raises(ValueError):
        hidden.matrix[0, 0] = 1.0  # read-only copy

    with torch.no_grad():
        model[0].weight.add_(0.25)
    peek = tracker.snapshot(1, baseline=False)
    second = tracker.snapshot(1)
    np.testing.assert_allclose(second.layers[0].delta, 0.25, atol=1e-6)
    np.testing.assert_allclose(peek.layers[0].delta, second.layers[0].delta)
    assert second.layers[0].update_ratio > 0
    assert tracker.snapshot(2).layers[0].update_ratio == pytest.approx(0.0)


def test_connection_matrix_linear_conv_transposed_and_embedding():
    linear = nn.Linear(3, 4)
    full = lambda n: display_indices(n, 64)  # noqa: E731
    torch.testing.assert_close(connection_matrix(linear, full(3), full(4)), linear.weight.detach())

    conv = nn.Conv2d(4, 6, 3, groups=2)
    matrix = connection_matrix(conv, full(4), full(6))
    assert matrix.shape == (6, 4)
    assert torch.all(matrix[:3, 2:] == 0) and torch.all(matrix[3:, :2] == 0)
    torch.testing.assert_close(matrix[:3, :2], conv.weight.detach()[:3].mean(dim=(2, 3)))

    transposed = nn.ConvTranspose1d(3, 5, 2)
    matrix = connection_matrix(transposed, full(3), full(5))
    torch.testing.assert_close(matrix, transposed.weight.detach().mean(dim=2).T)

    embedding = nn.Embedding(10, 3)
    matrix = connection_matrix(embedding, full(10), full(3))
    torch.testing.assert_close(matrix, embedding.weight.detach().T)

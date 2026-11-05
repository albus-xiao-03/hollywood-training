"""G1: the GUI shows the architecture, the connections between each layer's neurons and
the REAL weights of each layer, changing with the REAL training at EACH epoch."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from torch import nn

import hollywood_training as ht
from hollywood_training.gui import AnimationClock
from hollywood_training.gui.widgets import NetworkView, build_columns
from hollywood_training.gui.widgets import network_view as network_view_module
from hollywood_training.sync import SessionState
from hollywood_training.tracking import NetworkTracker

from .helpers import TinyNet, epoch_completed, image_rgb, session_started, text_recorder

pytestmark = [pytest.mark.spec("G1"), pytest.mark.gui]


def process_for(model: nn.Module, inputs: int = 1) -> ht.TrainingProcess:
    parameters = [
        p for p in model.parameters() if not isinstance(p, nn.parameter.UninitializedParameter)
    ]
    return ht.TrainingProcess(
        model,
        torch.optim.SGD(parameters or [nn.Parameter(torch.zeros(1))], lr=0.1),
        [(torch.zeros(1, inputs), torch.zeros(1, 1))],
        epochs=3,
        loss_fn=nn.MSELoss(),
    )


def make_view(qtbot, model: nn.Module, *, max_neurons: int = 20, inputs: int = 1):
    """A standalone network view whose animation clock never ticks (frames are stable)."""
    tracker, started = session_started(process_for(model, inputs), max_neurons=max_neurons)
    session = SessionState()
    session.apply(started)
    clock = AnimationClock(60)
    view = NetworkView(session, clock)
    qtbot.addWidget(view)
    view.resize(560, 380)
    return view, session, tracker, clock


def real_weight(model: nn.Module, name: str) -> np.ndarray:
    return dict(model.named_modules())[name].weight.detach().numpy()


# ------------------------------------------------------------- what is drawn
def test_one_column_per_feature_vector_and_one_link_per_layer():
    model = nn.Sequential(nn.Linear(3, 5), nn.ReLU(), nn.Linear(5, 4), nn.Tanh(), nn.Linear(4, 2))
    tracker = NetworkTracker(model)
    columns, links = build_columns(tracker.architecture.describe(), tracker.snapshot(0), 20)
    assert [len(column.indices) for column in columns] == [3, 5, 4, 2]
    assert [column.kind for column in columns] == ["input", "hidden", "hidden", "output"]
    assert [link.name for link in links] == ["0", "2", "4"]
    assert [column.badges for column in columns[1:3]] == [("ReLU",), ("Tanh",)]
    for link in links:
        source, target = columns[link.source], columns[link.target]
        assert link.target == link.source + 1 and not link.bridge
        assert link.weights.matrix.shape == (len(target.indices), len(source.indices))
        np.testing.assert_array_equal(link.weights.matrix, real_weight(model, link.name))


def test_a_reshape_between_layers_is_bridged():
    model = nn.Sequential(nn.Conv2d(1, 2, 3), nn.ReLU(), nn.Flatten(), nn.Linear(8, 3))
    tracker = NetworkTracker(model)
    tracker.begin_trace()
    model(torch.randn(1, 1, 4, 4))
    tracker.end_trace()
    columns, links = build_columns(tracker.architecture.describe(), tracker.snapshot(0), 20)
    assert [len(column.indices) for column in columns] == [1, 2, 8, 3]
    assert [link.bridge for link in links] == [False, True, False]


def test_wide_layers_are_sampled_consistently_and_say_so():
    model = nn.Sequential(nn.Linear(3, 64), nn.ReLU(), nn.Linear(64, 2))
    tracker = NetworkTracker(model, max_neurons=8)
    columns, links = build_columns(tracker.architecture.describe(), tracker.snapshot(0), 8)
    assert [len(column.indices) for column in columns] == [3, 8, 2]
    assert columns[1].caption == "8 of 64 neurons"
    np.testing.assert_array_equal(links[1].weights.in_index, columns[1].indices)
    full = real_weight(model, "0")[columns[1].indices]
    np.testing.assert_array_equal(links[0].weights.matrix, full)


def test_connection_colours_follow_the_sign_of_the_real_weight(qtbot):
    """README: positive weights are drawn cyan, negative ones indigo."""
    counts = {}
    for sign in (1.0, -1.0):
        model = nn.Sequential(nn.Linear(1, 1))
        with torch.no_grad():
            model[0].weight.fill_(sign)
            model[0].bias.zero_()
        view, *_ = make_view(qtbot, model)
        rgb = image_rgb(view.grab()).astype(int)
        r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
        counts[sign] = {
            "cyan": int((((g - r) > 60) & (b > 150)).sum()),
            "indigo": int((((r - g) > 4) & (b > 150) & (r > 60)).sum()),
        }
    assert counts[1.0]["cyan"] > counts[-1.0]["cyan"] + 100, counts
    assert counts[-1.0]["indigo"] > counts[1.0]["indigo"] + 100, counts


def test_new_weights_change_the_drawing(qtbot):
    model = TinyNet()
    view, session, tracker, _clock = make_view(qtbot, model, inputs=2)
    before = image_rgb(view.grab())
    with torch.no_grad():
        model.fc1.weight.mul_(-1.0)
        model.fc2.weight.pow_(3)
    session.apply(epoch_completed(tracker, 1, total=3, train_loss=0.5))
    after = image_rgb(view.grab())
    assert view.last_painted_epoch == 1
    assert np.abs(before.astype(int) - after.astype(int)).sum() > 0


def test_layers_neurons_and_operations_are_labelled(qtbot, monkeypatch):
    texts = text_recorder(monkeypatch, network_view_module)

    @ht.track(input_labels=("x", "y"), output_labels=("no", "yes"))
    class Labelled(TinyNet):
        pass

    view, *_ = make_view(qtbot, Labelled(), inputs=2)
    view.grab()
    for expected in ("INPUT", "FC1", "FC2", "Linear 2→6", "TANH", "6 neurons", "2 features"):
        assert any(expected in text for text in texts), expected
    for label in ("x", "y", "no", "yes"):
        assert label in texts, label


# ---------------------------------------------------------------- edge cases
def test_a_network_without_connection_layers_says_so(qtbot, monkeypatch):
    texts = text_recorder(monkeypatch, network_view_module)
    view, *_ = make_view(qtbot, nn.Sequential(nn.PReLU()))
    view.grab()
    assert any("NO CONNECTION LAYERS" in text for text in texts)


def test_layers_still_waiting_for_their_first_batch_are_drawn(qtbot):
    model = nn.Sequential(nn.LazyLinear(3), nn.ReLU(), nn.Linear(3, 2))
    view, *_ = make_view(qtbot, model, inputs=4)
    image = view.grab()
    assert not image.isNull() and view.last_painted_epoch == 0


def test_layers_without_bias_are_drawn(qtbot):
    view, *_ = make_view(qtbot, nn.Sequential(nn.Linear(2, 3, bias=False)), inputs=2)
    assert not view.grab().isNull() and view.last_painted_epoch == 0


def test_the_neuron_limit_of_run_is_applied(qapp):
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(2, 40), nn.ReLU(), nn.Linear(40, 2))
    process = ht.TrainingProcess(
        model,
        torch.optim.SGD(model.parameters(), lr=0.01),
        [(torch.randn(8, 2), torch.randint(0, 2, (8,)))],
        epochs=2,
        loss_fn=nn.CrossEntropyLoss(),
    )
    result = ht.run(process, isolate_gui=False, close_on_finish=True, max_neurons=5)
    hidden = result.final_weights.layer("0")
    assert hidden.matrix.shape == (5, 2) and hidden.out_features == 40
    assert hidden.stats.count == 80  # statistics still describe the whole layer

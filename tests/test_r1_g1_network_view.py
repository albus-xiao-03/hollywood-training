"""R1 GUI layout 1 - "The GUI **MUST** demonstrate the architecture of the network, the
connection between each layer's neurons, and the **REAL** weights of each layer that should
be changed with **REAL** training update along with **EACH** `epoch`."

* the architecture: an input column plus one column of neurons per connection layer, with
  titles, the layers' summaries, the neuron counts (sampled wide layers say so) and the
  activation / normalisation operations; convolution, reshape and embedding networks;
  neuron labels; the states before a session and without connection layers;
* the connections: between consecutive columns, carrying the layers' **real** weight
  matrices and biases (the per-epoch equality in real runs is verified in
  ``test_r1_05_synchronization.py``); the drawing changes when an epoch arrives and holds
  still otherwise;
* boundaries: weight matrices of recurrent / attention layers must be shown too; very wide
  layers draw quickly; at the window's minimum size every panel stays fully visible.
"""

from __future__ import annotations

import itertools
import time

import numpy as np
import pytest
import torch
from htspec.models import ConvNet, DeepNet, Setup, TinyNet, WideNet, loader, make_setup
from htspec.painting import force_full_repaint
from PySide6.QtWidgets import QApplication
from torch import nn

import hollywood_training as ht
from hollywood_training.gui.widgets import build_columns

pytestmark = pytest.mark.req("R1.G1")


def image_of(widget) -> np.ndarray:
    image = widget.grab().toImage()
    buffer = np.frombuffer(image.constBits(), np.uint8, count=image.sizeInBytes())
    return buffer.reshape(image.height(), image.bytesPerLine())[:, : image.width() * 4].copy()


def drawn_links(rig):
    """The weight links of the view, as the view builds them (``build_columns``)."""
    session = rig.session
    limit = session.info.neuron_limit
    _columns, links = build_columns(session.architecture, session.weights, limit)
    return [link for link in links if not link.bridge]


def rig_for(make_rig, model, **setup_options):
    rig = make_rig(setup=make_setup(model=model, **setup_options))
    rig.feeder.start()
    rig.process_events()
    return rig


# ------------------------------------------------------------------ architecture
def test_an_input_column_and_one_column_per_connection_layer(rig):
    columns = rig.window.network.columns()
    assert [c.title for c in columns] == ["INPUT", "FC1", "FC2"]
    assert [c.kind for c in columns] == ["input", "hidden", "output"]
    assert [len(c.indices) for c in columns] == [2, 8, 3]
    assert [c.subtitle for c in columns[1:]] == ["Linear 2→8", "Linear 8→3"]
    assert columns[1].badges == ("Tanh",)


def test_connections_link_consecutive_columns_with_the_real_weights(rig):
    rig.feeder.epochs(2)
    columns = rig.window.network.columns()
    links = drawn_links(rig)
    assert [(link.source, link.target) for link in links] == [(0, 1), (1, 2)]
    model = rig.setup.model
    for link, module in zip(links, (model.fc1, model.fc2), strict=True):
        matrix = link.weights.matrix
        shape = (len(columns[link.target].indices), len(columns[link.source].indices))
        assert matrix.shape == shape
        np.testing.assert_array_equal(matrix, module.weight.detach().numpy())
        np.testing.assert_array_equal(columns[link.target].bias, module.bias.detach().numpy())


def test_the_drawing_changes_with_an_epoch_and_holds_still_otherwise(rig):
    view = rig.window.network
    rig.window.clock.stop()  # freeze the ambient animation
    QApplication.processEvents()
    before = image_of(view)
    assert np.array_equal(before, image_of(view))  # nothing happened: nothing changed
    rig.feeder.train_epoch()  # real training: the weights change
    after = image_of(view)
    assert not np.array_equal(before, after)
    assert np.array_equal(after, image_of(view))


def test_titles_summaries_captions_and_operations_are_drawn(make_rig, paint_capture):
    rig = rig_for(make_rig, DeepNet(), features=4)
    drawn = paint_capture.render(rig.window.network)
    for text in ("INPUT", "FC1", "FC2", "FC3", "Linear 4→12", "Linear 12→12", "Linear 12→2",
                 "4 features", "12 neurons", "2 neurons"):  # fmt: skip
        assert text in drawn, text
    joined = "\n".join(drawn)
    for operation in ("BATCHNORM1D(12)", "RELU", "TANH", "DROPOUT(P=0.1)"):
        assert operation in joined, operation


def test_wide_layers_are_sampled_and_say_so(make_rig, paint_capture):
    rig = rig_for(make_rig, WideNet(200), features=3)
    columns = rig.window.network.columns()
    assert [len(c.indices) for c in columns] == [3, 20, 2]
    assert "20 of 200 neurons" in paint_capture.render(rig.window.network)


def test_convolution_and_reshape_are_drawn(make_rig, paint_capture):
    rig = rig_for(make_rig, ConvNet())
    columns = rig.window.network.columns()
    assert [c.title for c in columns] == ["INPUT", "CONV", "FC IN", "FC"]
    drawn = paint_capture.render(rig.window.network)
    assert "RESHAPE" in drawn
    assert "1 feature" in drawn and "20 of 64 features" in drawn
    assert any(text.startswith("Conv2d 1→4") for text in drawn)


def test_an_embedding_network_is_drawn(make_rig):
    rig = rig_for(make_rig, nn.Sequential(nn.Embedding(10, 4), nn.Linear(4, 2)))
    columns = rig.window.network.columns()
    assert [len(c.indices) for c in columns] == [10, 4, 2]


def test_neuron_labels_are_drawn(make_rig, paint_capture):
    model = ht.track(TinyNet(), input_labels=("x", "y"), output_labels=("A", "B", "C"))
    rig = rig_for(make_rig, model)
    drawn = paint_capture.render(rig.window.network)
    for label in ("x", "y", "A", "B", "C"):
        assert label in drawn, label


def test_before_a_session_the_view_waits_for_the_network(make_rig, paint_capture):
    rig = make_rig()
    assert "AWAITING NETWORK SIGNAL" in paint_capture.render(rig.window.network)


def test_a_network_without_connection_layers(make_rig, paint_capture):
    model = nn.Sequential(nn.ReLU(), nn.Tanh())
    optimizer = torch.optim.SGD([nn.Parameter(torch.zeros(1))], lr=0.1)
    data = loader()
    process = ht.TrainingProcess(model, optimizer, data, loss_fn=nn.MSELoss(), epochs=2)
    rig = make_rig(setup=Setup(model, optimizer, nn.MSELoss(), data, None, process))
    rig.feeder.start()
    assert rig.window.network.columns() == []
    assert "NO CONNECTION LAYERS TO DRAW" in paint_capture.render(rig.window.network)


# ------------------------------------------------------------------- boundaries
class RecurrentNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.rnn = nn.LSTM(2, 5)
        self.head = nn.Linear(5, 3)


class AttentionNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.inp = nn.Linear(2, 8)
        self.attn = nn.MultiheadAttention(8, 2)
        self.head = nn.Linear(8, 3)


@pytest.mark.parametrize("network", [RecurrentNet, AttentionNet], ids=lambda c: c.__name__)
def test_the_weights_of_every_layer_are_shown(make_rig, network):
    """Strict: every weight matrix of the network is shown as connections."""
    rig = rig_for(make_rig, network())
    shown = {link.name for link in drawn_links(rig)}
    missing = [
        name for name, parameter in rig.setup.model.named_parameters()
        if parameter.dim() >= 2 and name.rsplit(".", 1)[0] not in shown
    ]  # fmt: skip
    assert missing == [], f"weights not shown: {missing}"


def test_very_wide_layers_draw_quickly(make_rig):
    rig = rig_for(make_rig, WideNet(4000), features=3)
    view = rig.window.network
    started = time.perf_counter()
    force_full_repaint(view)
    assert time.perf_counter() - started < 1.0


def _visible_rects(window):
    names = ("header", "network", "terminal", "status", "chart", "usage")
    rects = {}
    for name in names:
        widget = getattr(window, name)
        top_left = widget.mapTo(window, widget.rect().topLeft())
        rects[name] = (top_left.x(), top_left.y(), widget.width(), widget.height())
    return rects


@pytest.mark.req("R1.G2", "R1.G3", "R1.G4", "R1.G5")
@pytest.mark.parametrize(
    "size", [(960, 680), (1202, 680), (1366, 720)], ids=lambda s: f"{s[0]}x{s[1]}"
)
def test_every_panel_stays_fully_visible_at_small_window_sizes(rig, size):
    """The window allows these sizes (its minimum is 960 x 680; a 1366 x 768 laptop screen
    opens it at 680 px high): no panel may overlap another, leave the window or be squeezed
    below its own minimum size - the information of every panel must stay visible."""
    window = rig.window
    window.resize(*size)
    rig.process_events(100)
    assert (window.width(), window.height()) == size
    rects = _visible_rects(window)
    problems = []
    for name, (x, y, w, h) in rects.items():
        widget = getattr(window, name)
        if x < 0 or y < 0 or x + w > window.width() or y + h > window.height():
            problems.append(f"{name} leaves the window: {(x, y, w, h)}")
        if w < widget.minimumWidth() or h < widget.minimumHeight():
            problems.append(f"{name} squeezed to {w}x{h} (minimum {widget.minimumWidth()}x"
                            f"{widget.minimumHeight()})")  # fmt: skip
    for (a, ra), (b, rb) in itertools.combinations(rects.items(), 2):
        overlap_w = min(ra[0] + ra[2], rb[0] + rb[2]) - max(ra[0], rb[0])
        overlap_h = min(ra[1] + ra[3], rb[1] + rb[3]) - max(ra[1], rb[1])
        if overlap_w > 0 and overlap_h > 0:
            problems.append(f"{a} and {b} overlap")
    assert problems == []

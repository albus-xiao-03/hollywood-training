"""G6: the GUI shows the name of the training network - its class name."""

from __future__ import annotations

import io

import pytest
import torch
from torch import nn

import hollywood_training as ht
from hollywood_training.console import ConsoleFrontend
from hollywood_training.gui import MainWindow
from hollywood_training.gui.widgets import header as header_module
from hollywood_training.sync import SessionState, SyncChannel
from hollywood_training.training import TrainingManager

from .helpers import TinyNet, build_process, call_recorder, session_started, text_recorder

pytestmark = pytest.mark.spec("G6")


@ht.track(name="Quadrant classifier")
class QuadrantNet(TinyNet):
    pass


def shown_names(qtbot, monkeypatch, model: nn.Module, *, title: str | None = None):
    """What the window shows for ``model``: (window title, glowing header texts, small texts)."""
    glows = call_recorder(monkeypatch, header_module, "draw_glow_text")
    texts = text_recorder(monkeypatch, header_module)
    bundle = build_process(model=model)
    _tracker, started = session_started(bundle.process)
    session = SessionState()
    window = MainWindow(session, title=title)
    qtbot.addWidget(window)
    with qtbot.waitExposed(window):  # mapped, so the header really paints below
        window.show()
    session.apply(started)
    window.apply(started)
    window.header.repaint()
    return window.windowTitle(), [args[2] for args, _ in glows], texts, window


@pytest.mark.gui
def test_the_header_shows_the_class_name_of_the_network(qtbot, monkeypatch):
    title, glowing, _texts, _window = shown_names(qtbot, monkeypatch, TinyNet())
    assert "TinyNet" in glowing
    assert "TinyNet" in title


@pytest.mark.gui
def test_a_display_name_is_shown_in_addition_to_the_class_name(qtbot, monkeypatch):
    _title, glowing, texts, _window = shown_names(qtbot, monkeypatch, QuadrantNet())
    assert "QuadrantNet" in glowing
    assert any("Quadrant classifier" in text for text in texts)


@pytest.mark.gui
def test_a_custom_window_title_keeps_the_class_name_in_the_header(qtbot, monkeypatch):
    title, glowing, _texts, _window = shown_names(qtbot, monkeypatch, TinyNet(), title="My run")
    assert title == "My run"
    assert "TinyNet" in glowing


@pytest.mark.gui
@pytest.mark.parametrize(
    ("factory", "name"),
    [
        (lambda: nn.Sequential(nn.Linear(2, 3), nn.ReLU(), nn.Linear(3, 2)), "Sequential"),
        (lambda: nn.DataParallel(QuadrantNet()), "QuadrantNet"),
    ],
    ids=["sequential", "data-parallel"],
)
def test_the_class_name_of_other_kinds_of_networks(qtbot, monkeypatch, factory, name):
    title, glowing, _texts, _window = shown_names(qtbot, monkeypatch, factory())
    assert name in glowing and name in title


@pytest.mark.gui
def test_a_compiled_network_is_shown_with_its_own_class_name(qtbot, monkeypatch):
    """``torch.compile`` wraps the network; the window must still name the user's class
    (QA finding M4)."""
    try:
        compiled = torch.compile(QuadrantNet(), backend="eager")
    except Exception as exc:  # pragma: no cover - platform without torch.compile
        pytest.skip(f"torch.compile unavailable: {exc}")
    title, glowing, _texts, _window = shown_names(qtbot, monkeypatch, compiled)
    assert "QuadrantNet" in glowing and "QuadrantNet" in title


@pytest.mark.gui
def test_the_terminal_names_the_network_too(qtbot, monkeypatch):
    _title, _glowing, _texts, window = shown_names(qtbot, monkeypatch, QuadrantNet())
    assert "QuadrantNet" in window.terminal.text()


def test_the_console_frontend_names_the_network():
    stream = io.StringIO()
    channel = SyncChannel()
    ConsoleFrontend(channel, stream=stream)
    TrainingManager(build_process(epochs=1, model=QuadrantNet()).process, channel=channel).run()
    assert "QuadrantNet" in stream.getvalue()

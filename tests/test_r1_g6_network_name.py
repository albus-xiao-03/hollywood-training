"""R1 GUI layout 6 - "The GUI **MUST** show the name of the training network(its class
name)."

* the header draws the network's class name; the window title and the terminal's greeting
  name it too;
* for every way of handing over a network: a class of the user's (tracked or not), an
  ``nn.Sequential``, a ``DataParallel`` wrapper (the network inside), a tracked display
  name (shown *next to* the class name), a custom window title (the header still names the
  class), the default window process (its title);
* boundaries: long and non-ASCII class names; strict - a ``torch.compile``-d network is
  still the user's class (not PyTorch's ``OptimizedModule`` wrapper).
"""

from __future__ import annotations

import pytest
import torch
from htspec.models import TinyNet, make_setup, sequential_net
from htspec.probe_log import LINGER_CLOSE
from torch import nn

import hollywood_training as ht

pytestmark = pytest.mark.req("R1.G6")


def header_texts(make_rig, paint_capture, model, *, title=None):
    """Everything the header draws for ``model`` (its cached name included), and the rig."""
    rig = make_rig(setup=make_setup(model=model), title=title)
    rig.feeder.start()
    return paint_capture.render(rig.window.header), rig


def test_the_header_shows_the_class_name(make_rig, paint_capture):
    texts, _rig = header_texts(make_rig, paint_capture, TinyNet())
    assert "TinyNet" in texts
    assert "TRAINING NETWORK" in texts  # the label above the name


def test_the_window_title_and_the_terminal_name_the_class(rig):
    assert rig.window.windowTitle() == "Hollywood Training — TinyNet"
    assert any(line.startswith("network TinyNet ·") for line in rig.log_lines())


def test_a_sequential_network_is_shown_as_sequential(make_rig, paint_capture):
    texts, rig = header_texts(make_rig, paint_capture, sequential_net())
    assert "Sequential" in texts
    assert rig.window.windowTitle().endswith("— Sequential")


def test_a_wrapped_network_is_shown_by_its_own_class(make_rig, paint_capture):
    texts, _rig = header_texts(make_rig, paint_capture, nn.DataParallel(TinyNet()))
    assert "TinyNet" in texts


def test_a_display_name_is_shown_next_to_the_class_name(make_rig, paint_capture):
    model = ht.track(TinyNet(), name="Spiral Brain")
    texts, _rig = header_texts(make_rig, paint_capture, model)
    assert "TinyNet" in texts
    assert any("Spiral Brain" in text for text in texts)


def test_a_custom_window_title_keeps_the_class_name_in_the_header(make_rig, paint_capture):
    texts, rig = header_texts(make_rig, paint_capture, TinyNet(), title="My experiment")
    assert rig.window.windowTitle() == "My experiment"
    assert "TinyNet" in texts


@pytest.mark.subprocess
def test_the_window_process_names_the_class(probe):
    log = probe(script={"steps": [LINGER_CLOSE]})
    ht.run(make_setup(epochs=2).process)
    titles = {state["title"] for state in log.states() if state["visible"]}
    assert "Hollywood Training — TinyNet" in titles


@pytest.mark.parametrize("name", ["A" * 120, "Réseau_Δ_网络"], ids=["long", "non-ascii"])
def test_unusual_class_names_are_shown(make_rig, paint_capture, name):
    network = type(name, (TinyNet,), {})()
    texts, rig = header_texts(make_rig, paint_capture, network)
    assert name in texts
    assert rig.window.windowTitle().endswith(name)


def test_a_compiled_network_is_still_shown_by_its_own_class(make_rig, paint_capture):
    """Strict: ``torch.compile`` wraps the user's network in ``OptimizedModule``; the GUI must
    still show the training network's class name. (Nothing is compiled here: the name is
    read before any forward pass.)"""
    compiled = torch.compile(TinyNet(), backend="eager")
    texts, rig = header_texts(make_rig, paint_capture, compiled)
    assert "TinyNet" in texts, f"the header shows {[t for t in texts if t.isidentifier()]}"
    assert rig.window.windowTitle() == "Hollywood Training — TinyNet"

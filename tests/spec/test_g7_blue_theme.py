"""G7: the GUI is blue based."""

from __future__ import annotations

import dataclasses
import re

import numpy as np
import pytest
from PySide6.QtGui import QColor

from hollywood_training.gui.theme import PALETTE, stylesheet
from hollywood_training.sync import SessionFinished, StateChanged, TrainingState

from .helpers import blue_share, hue_chroma, image_rgb

pytestmark = [pytest.mark.spec("G7"), pytest.mark.gui]

BASE_COLOURS = (
    "void", "background_top", "background_bottom", "panel", "panel_edge", "grid", "accent",
    "accent_soft", "electric", "indigo", "text", "text_dim",
)  # fmt: skip


def hue_of(name: str) -> float:
    colour = QColor(name)
    return colour.hsvHueF() * 360.0


def test_every_base_colour_of_the_theme_is_blue():
    for field in BASE_COLOURS:
        value = getattr(PALETTE, field)
        colour = QColor(value)
        assert colour.blue() >= colour.red(), f"{field} {value} is not blue based"
        assert 180.0 <= hue_of(value) <= 260.0, f"{field} {value} hue {hue_of(value):.0f}"


def test_only_status_accents_leave_the_blue_range():
    others = {f.name for f in dataclasses.fields(PALETTE)} - set(BASE_COLOURS)
    assert others <= {"ice", "success", "warning", "error"}


def test_the_widget_stylesheet_uses_blue_colours():
    colours = re.findall(r"#[0-9a-fA-F]{6}", stylesheet())
    rgba = re.findall(r"rgba\((\d+),\s*(\d+),\s*(\d+)", stylesheet())
    colours += ["#{:02x}{:02x}{:02x}".format(*map(int, triple)) for triple in rgba]
    assert colours
    for value in colours:
        colour = QColor(value)
        chroma = max(colour.red(), colour.green(), colour.blue()) - min(
            colour.red(), colour.green(), colour.blue()
        )
        if chroma >= 30:  # greys carry no hue
            assert 170.0 <= hue_of(value) <= 270.0, value


def _assert_blue(rgb: np.ndarray) -> None:
    share = blue_share(rgb)
    assert share >= 0.8, f"only {share:.0%} of the coloured pixels are blue"
    mean = rgb.reshape(-1, 3).mean(axis=0)
    assert mean[2] > mean[0] and mean[2] > mean[1], f"mean colour {mean} is not blue"


def test_the_rendered_window_is_blue(gui):
    gui.complete_epoch(1)
    gui.complete_epoch(2, train_loss=0.6, val_loss=0.7)
    _assert_blue(image_rgb(gui.window.grab()))


@pytest.mark.parametrize(
    "state", [TrainingState.PAUSED, TrainingState.FINISHED, TrainingState.FAILED]
)
def test_the_window_stays_blue_in_every_state(gui, state):
    gui.complete_epoch(1)
    gui.push(StateChanged(state, 1, 5, 0.1, 0.0, 0.1, f"now {state.value}"))
    if state is not TrainingState.PAUSED:
        weights = gui.tracker.snapshot(1, baseline=False)
        gui.push(
            SessionFinished(
                state, 1, 0.1, weights, "done", "Boom" if state.value == "failed" else None
            )
        )
    _assert_blue(image_rgb(gui.window.grab()))


def test_the_hue_helper_itself():
    rgb = np.array([[[0, 0, 255], [0, 229, 255], [255, 0, 0], [128, 128, 128]]], dtype=np.uint8)
    hue, chroma, _value = hue_chroma(rgb)
    assert hue[0] == pytest.approx(240.0) and 180 <= hue[1] <= 190
    assert hue[2] == pytest.approx(0.0) and np.isnan(hue[3]) and chroma[3] == 0

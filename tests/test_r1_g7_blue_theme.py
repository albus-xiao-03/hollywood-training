"""R1 GUI layout 7 - "The GUI **MUST** be blue based."

Measured on what is actually on screen: the colourful pixels (saturation >= 0.25, value >=
0.15 - so near-black, grey and near-white pixels do not decide) of a rendered window must be
blue: hue in the blue range 170°-265°, which spans the theme's own base colours (cyan
#00e5ff at 186° to indigo #7a6cff at 246°) with a margin. The status accents (green success,
yellow warning, pink error) are expected to stay small, so at least 80 % of the colourful
pixels must be blue - for the whole window, for every panel, for the window icon and for a
screenshot taken by the real window process; the average colour is blue-dominant; and the
theme's base colours themselves are blue.
"""

from __future__ import annotations

import time

import numpy as np
import pytest
from htspec.models import make_setup
from htspec.probe_log import LINGER_CLOSE
from PySide6.QtGui import QColor, QImage

import hollywood_training as ht
from hollywood_training.gui.theme import PALETTE, app_icon

pytestmark = pytest.mark.req("R1.G7")

BLUE_HUES = (170.0, 265.0)
BASE_COLOURS = (
    "void", "background_top", "background_bottom", "panel", "panel_edge", "grid", "accent",
    "accent_soft", "electric", "indigo", "ice", "text", "text_dim",
)  # fmt: skip
PANELS = ("header", "network", "terminal", "status", "chart", "usage")


def rgb_of(image: QImage) -> np.ndarray:
    image = image.convertToFormat(QImage.Format.Format_RGB32)
    raw = np.frombuffer(image.constBits(), np.uint8, count=image.sizeInBytes())
    pixels = raw.reshape(image.height(), image.bytesPerLine())[:, : image.width() * 4]
    bgra = pixels.reshape(image.height(), image.width(), 4).astype(np.float64) / 255.0
    return bgra[..., [2, 1, 0]]  # Format_RGB32 is stored B, G, R, A


def hue_saturation_value(rgb: np.ndarray):
    high, low = rgb.max(axis=-1), rgb.min(axis=-1)
    delta = high - low
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    hue = np.zeros_like(high)
    safe = np.where(delta > 0, delta, 1.0)
    red = (delta > 0) & (high == r)
    green = (delta > 0) & (high == g) & ~red
    blue = (delta > 0) & ~red & ~green
    hue[red] = (60.0 * (g - b)[red] / safe[red]) % 360.0
    hue[green] = 60.0 * (b - r)[green] / safe[green] + 120.0
    hue[blue] = 60.0 * (r - g)[blue] / safe[blue] + 240.0
    saturation = np.where(high > 0, delta / np.where(high > 0, high, 1.0), 0.0)
    return hue, saturation, high


def blue_share(rgb: np.ndarray) -> tuple[float, float]:
    """``(share of the colourful pixels that are blue, share of colourful pixels)``."""
    hue, saturation, value = hue_saturation_value(rgb)
    colourful = (saturation >= 0.25) & (value >= 0.15)
    if not colourful.any():
        return 0.0, 0.0
    blue = colourful & (hue >= BLUE_HUES[0]) & (hue <= BLUE_HUES[1])
    return float(blue.sum() / colourful.sum()), float(colourful.mean())


@pytest.fixture
def busy(rig):
    rig.feeder.epochs(4)
    rig.feeder.log("a log line")
    rig.type_line("status")
    rig.process_events(100)
    return rig


def test_the_window_is_blue_based(busy):
    rgb = rgb_of(busy.window.grab().toImage())
    share, colourful = blue_share(rgb)
    assert colourful > 0.02, "almost no colour in the window"
    assert share >= 0.8, f"only {share:.0%} of the colourful pixels are blue"
    mean = rgb.reshape(-1, 3).mean(axis=0)
    assert mean[2] > mean[0] and mean[2] > mean[1], f"average colour {mean}"


@pytest.mark.parametrize("panel", PANELS)
def test_every_panel_is_blue_based(busy, panel):
    rgb = rgb_of(getattr(busy.window, panel).grab().toImage())
    share, colourful = blue_share(rgb)
    assert colourful > 0, f"{panel} has no colour at all"
    assert share >= 0.8, f"{panel}: only {share:.0%} of the colourful pixels are blue"


@pytest.mark.parametrize("name", BASE_COLOURS)
def test_the_themes_base_colours_are_blue(name):
    colour = QColor(getattr(PALETTE, name))
    hue = colour.hsvHueF() * 360.0
    assert BLUE_HUES[0] <= hue <= BLUE_HUES[1], f"{name} {colour.name()} has hue {hue:.0f}°"


def test_the_window_icon_is_blue(qapp):
    share, colourful = blue_share(rgb_of(app_icon().pixmap(64, 64).toImage()))
    assert colourful > 0 and share >= 0.8


@pytest.mark.subprocess
def test_a_screenshot_of_the_window_process_is_blue_based(probe, tmp_path):
    target = tmp_path / "window.png"
    steps = [
        {"on": "epoch", "value": 2, "do": "keys", "keys": f"screenshot {target}<Return>"},
        {**LINGER_CLOSE, "delay_ms": 800},
    ]
    probe(script={"steps": steps})
    ht.run(make_setup(epochs=4, train_hook=lambda _b: time.sleep(0.005)).process)
    image = QImage(str(target))
    assert not image.isNull(), "no screenshot was written"
    share, colourful = blue_share(rgb_of(image))
    assert colourful > 0.02 and share >= 0.8, f"{share:.0%} blue"

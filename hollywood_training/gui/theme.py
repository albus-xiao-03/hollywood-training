"""Visual identity of the GUI: a blue HUD ("Hollywood") palette, fonts and painting helpers.

Fonts fall back through common monospace families of Windows, macOS and Linux to the
system's fixed-pitch font, so the look is consistent without bundling font files.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import (
    QColor,
    QFont,
    QFontDatabase,
    QFontMetricsF,
    QIcon,
    QPainter,
    QPainterPath,
    QPen,
    QPixmap,
    QRadialGradient,
)


@dataclass(frozen=True, slots=True)
class Palette:
    """Colour names used throughout the GUI (all blue-based, plus status accents)."""

    void: str = "#01040c"
    background_top: str = "#03102a"
    background_bottom: str = "#062656"
    panel: str = "#06193a"
    panel_edge: str = "#1f6fd1"
    grid: str = "#0f3d7d"
    accent: str = "#00e5ff"  # cyan: primary highlight, positive weights, training loss
    accent_soft: str = "#5cc8ff"
    electric: str = "#2f7bff"
    indigo: str = "#7a6cff"  # negative weights, validation loss
    ice: str = "#e2f8ff"  # brightest text
    text: str = "#a8d8ff"
    text_dim: str = "#5d8cc4"
    success: str = "#3dffcf"
    warning: str = "#ffd166"
    error: str = "#ff5c8a"


PALETTE = Palette()

MONO_FAMILIES = (
    "JetBrains Mono",
    "Cascadia Mono",
    "Cascadia Code",
    "SF Mono",
    "Menlo",
    "Consolas",
    "Fira Code",
    "Source Code Pro",
    "Hack",
    "DejaVu Sans Mono",
    "Ubuntu Mono",
    "Liberation Mono",
    "Noto Sans Mono",
    "Courier New",
)


def color(name: str, alpha: int = 255) -> QColor:
    """``QColor`` from a hex name with an optional alpha (0-255)."""
    value = QColor(name)
    value.setAlpha(max(0, min(255, int(alpha))))
    return value


def mix(a: QColor, b: QColor, t: float) -> QColor:
    """Linear blend between two colours (``t`` = 0 gives ``a``)."""
    t = max(0.0, min(1.0, t))
    return QColor.fromRgbF(
        a.redF() + (b.redF() - a.redF()) * t,
        a.greenF() + (b.greenF() - a.greenF()) * t,
        a.blueF() + (b.blueF() - a.blueF()) * t,
        a.alphaF() + (b.alphaF() - a.alphaF()) * t,
    )


@lru_cache(maxsize=1)
def mono_family() -> str:
    """The first available monospace family (needs a running QGuiApplication)."""
    available = set(QFontDatabase.families())
    for family in MONO_FAMILIES:
        if family in available:
            return family
    return QFontDatabase.systemFont(QFontDatabase.SystemFont.FixedFont).family()


def mono_font(point_size: float = 10.0, *, bold: bool = False, spacing: float = 0.0) -> QFont:
    font = QFont(mono_family())
    font.setPointSizeF(point_size)
    font.setStyleHint(QFont.StyleHint.TypeWriter)
    font.setFixedPitch(True)
    if bold:
        font.setWeight(QFont.Weight.Bold)
    if spacing:
        font.setLetterSpacing(QFont.SpacingType.AbsoluteSpacing, spacing)
    return font


def stylesheet() -> str:
    """Style for the standard widgets inside the main window (not applied app-wide)."""
    p = PALETTE
    return f"""
    QMainWindow {{ background: {p.void}; }}
    QToolTip {{
        color: {p.ice}; background: {p.panel}; border: 1px solid {p.panel_edge}; padding: 4px;
    }}
    QPlainTextEdit#terminalOutput {{
        background: transparent; color: {p.text}; border: none;
        selection-background-color: {p.electric};
    }}
    QLineEdit#terminalInput {{
        background: rgba(3, 18, 44, 190); color: {p.ice}; border: 1px solid {p.grid};
        border-radius: 2px; padding: 5px 8px; selection-background-color: {p.electric};
    }}
    QLineEdit#terminalInput:focus {{ border: 1px solid {p.accent}; }}
    QLabel#terminalPrompt {{ color: {p.accent}; }}
    QScrollBar:vertical {{ background: transparent; width: 8px; margin: 2px; }}
    QScrollBar::handle:vertical {{ background: {p.grid}; min-height: 24px; border-radius: 3px; }}
    QScrollBar::handle:vertical:hover {{ background: {p.electric}; }}
    QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0; }}
    QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {{ background: none; }}
    QSplitter::handle {{ background: transparent; }}
    QSplitter::handle:hover {{ background: rgba(0, 229, 255, 50); }}
    """


def chamfer_path(rect: QRectF, cut: float) -> QPainterPath:
    """Rectangle with the top-left and bottom-right corners cut off (HUD frame shape)."""
    path = QPainterPath()
    path.moveTo(rect.left() + cut, rect.top())
    path.lineTo(rect.right(), rect.top())
    path.lineTo(rect.right(), rect.bottom() - cut)
    path.lineTo(rect.right() - cut, rect.bottom())
    path.lineTo(rect.left(), rect.bottom())
    path.lineTo(rect.left(), rect.top() + cut)
    path.closeSubpath()
    return path


def draw_glow_text(
    painter: QPainter,
    position: QPointF,
    text: str,
    font: QFont,
    fill: QColor,
    glow: QColor,
    strength: float = 1.0,
) -> None:
    """Text with a soft neon halo (for headings; drawn as a path)."""
    path = QPainterPath()
    path.addText(position, font, text)
    painter.save()
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    for width, alpha in ((8.0, 18), (4.5, 38), (2.0, 70)):
        halo = QColor(glow)
        halo.setAlpha(int(alpha * strength))
        pen = QPen(halo, width)
        pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
        painter.strokePath(path, pen)
    painter.fillPath(path, fill)
    painter.restore()


_GLOW_PAD = 6


@lru_cache(maxsize=1024)
def _glyph(
    char: str, family: str, size: float, weight: int, spacing: float, fill: int, glow: int,
    strength: float, dpr: float,
) -> tuple[QPixmap, float]:  # fmt: skip
    font = QFont(family)
    font.setPointSizeF(size)
    font.setWeight(QFont.Weight(weight))
    font.setLetterSpacing(QFont.SpacingType.AbsoluteSpacing, spacing)
    metrics = QFontMetricsF(font)
    width = metrics.horizontalAdvance(char) + 2 * _GLOW_PAD
    height = metrics.height() + 2 * _GLOW_PAD
    pixmap = QPixmap(math.ceil(width * dpr), math.ceil(height * dpr))
    pixmap.setDevicePixelRatio(dpr)
    pixmap.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pixmap)
    baseline = _GLOW_PAD + metrics.ascent()
    draw_glow_text(
        painter, QPointF(_GLOW_PAD, baseline), char, font,
        QColor.fromRgba(fill), QColor.fromRgba(glow), strength,
    )  # fmt: skip
    painter.end()
    return pixmap, baseline


def glow_text_width(text: str, font: QFont) -> float:
    """Width of ``text`` as drawn by :func:`draw_glow_mono` (fixed advance per character)."""
    metrics = QFontMetricsF(font)
    return (metrics.horizontalAdvance("00") - metrics.horizontalAdvance("0")) * len(text)


def draw_glow_mono(
    painter: QPainter,
    position: QPointF,
    text: str,
    font: QFont,
    fill: QColor,
    glow: QColor,
    strength: float = 1.0,
) -> None:
    """Like :func:`draw_glow_text`, but from cached per-character sprites.

    Cheap enough for text that changes every frame (timers, counters); assumes a
    monospace ``font``. ``position`` is the left end of the baseline.
    """
    metrics = QFontMetricsF(font)
    advance = metrics.horizontalAdvance("00") - metrics.horizontalAdvance("0")
    dpr = round(painter.device().devicePixelRatioF(), 2)
    key = (
        font.family(), font.pointSizeF(), font.weight().value, font.letterSpacing(),
        fill.rgba(), glow.rgba(), strength, dpr,
    )  # fmt: skip
    x = position.x()
    for char in text:
        if not char.isspace():
            sprite, baseline = _glyph(char, *key)
            painter.drawPixmap(QPointF(x - _GLOW_PAD, position.y() - baseline), sprite)
        x += advance


@lru_cache(maxsize=64)
def _sprite(radius: int, rgba: int, core: float, dpr: float) -> QPixmap:
    size = math.ceil(radius * 2 * dpr)
    pixmap = QPixmap(size, size)
    pixmap.setDevicePixelRatio(dpr)
    pixmap.fill(Qt.GlobalColor.transparent)
    tint = QColor.fromRgba(rgba)
    gradient = QRadialGradient(QPointF(radius, radius), radius)
    gradient.setColorAt(0.0, QColor(255, 255, 255, tint.alpha()))
    gradient.setColorAt(core, tint)
    faded = QColor(tint)
    faded.setAlpha(0)
    gradient.setColorAt(1.0, faded)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    painter.setPen(Qt.PenStyle.NoPen)
    painter.setBrush(gradient)
    painter.drawEllipse(QRectF(0, 0, radius * 2, radius * 2))
    painter.end()
    return pixmap


def glow_sprite(radius: int, tint: QColor, *, core: float = 0.25, dpr: float = 1.0) -> QPixmap:
    """Cached radial glow (white-hot centre fading to transparent ``tint``)."""
    return _sprite(max(1, int(radius)), tint.rgba(), core, round(dpr, 2))


def app_icon() -> QIcon:
    """Window icon drawn at runtime (no asset files)."""
    size = 64
    pixmap = QPixmap(size, size)
    pixmap.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    painter.setBrush(color(PALETTE.void))
    painter.setPen(QPen(color(PALETTE.accent), 3))
    painter.drawEllipse(QRectF(4, 4, size - 8, size - 8))
    nodes = [QPointF(20, 22), QPointF(20, 42), QPointF(44, 32)]
    painter.setPen(QPen(color(PALETTE.accent_soft, 200), 2))
    painter.drawLine(nodes[0], nodes[2])
    painter.drawLine(nodes[1], nodes[2])
    painter.setPen(Qt.PenStyle.NoPen)
    painter.setBrush(color(PALETTE.ice))
    for node in nodes:
        painter.drawEllipse(node, 5, 5)
    painter.end()
    return QIcon(pixmap)

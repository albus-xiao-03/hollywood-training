"""``PaintCapture``: what the panels actually draw - text, curves and bars.

The panels paint everything with ``QPainter`` (there are no label widgets), so the harness
records the painting itself. While a capture is active, the panel modules use a
``QPainter`` subclass that records every ``drawText`` string and every ``drawPath`` (its
number of path elements and pen), and the module-level helpers that draw glowing text
(``draw_glow_text`` in the header, ``draw_glow_mono`` in the status panel) and level bars
(``paint_bar``) are wrapped. Each record names the panel class that painted it.

Panels cache their static parts in pixmaps that are only redrawn when their content
changes. :func:`force_full_repaint` drops those caches so that a repaint draws everything
(it resets the cache keys the panels compare; nothing else is touched). Install the capture
*before* feeding events to observe the natural paints instead.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from importlib import import_module
from typing import Any

from PySide6.QtGui import QPainter
from PySide6.QtWidgets import QWidget

PANEL_MODULES = (
    "hollywood_training.gui.widgets.hud",
    "hollywood_training.gui.widgets.header",
    "hollywood_training.gui.widgets.loss_chart",
    "hollywood_training.gui.widgets.network_view",
    "hollywood_training.gui.widgets.status_panel",
    "hollywood_training.gui.widgets.usage_panel",
    "hollywood_training.gui.window",
)
CACHE_KEYS = ("_static_key", "_cache_key", "_key")


@dataclass(frozen=True)
class Drawn:
    panel: str  # class name of the painting widget ("" when unknown)
    text: str
    kind: str  # "text" (drawText) or "glow" (glowing heading / counter text)


@dataclass(frozen=True)
class DrawnPath:
    panel: str
    elements: int  # QPainterPath.elementCount()
    moves: int  # MoveTo elements: separate stretches of the path (a gap starts a new one)
    color: str  # pen colour, "#rrggbb"
    alpha: int
    width: float
    dashed: bool


@dataclass(frozen=True)
class DrawnBar:
    panel: str
    fraction: float | None


def _painting_panel() -> str:
    """Class name of the gui widget whose paint code is running (from the call stack)."""
    frame = sys._getframe(2)
    for _ in range(12):
        if frame is None:
            break
        owner = frame.f_locals.get("self")
        if isinstance(owner, QWidget) and type(owner).__module__.startswith("hollywood_training"):
            return type(owner).__name__
        frame = frame.f_back
    return ""


class RecordingPainter(QPainter):
    """A QPainter that reports to the active :class:`PaintCapture`."""

    capture: PaintCapture | None = None

    def drawText(self, *args: Any) -> Any:
        capture = RecordingPainter.capture
        if capture is not None:
            for arg in args:
                if isinstance(arg, str):
                    capture.texts.append(Drawn(_painting_panel(), arg, "text"))
        return super().drawText(*args)

    def drawPath(self, path: Any) -> Any:
        capture = RecordingPainter.capture
        if capture is not None:
            pen = self.pen()
            ink = pen.color()
            count = path.elementCount()
            capture.paths.append(
                DrawnPath(
                    _painting_panel(),
                    count,
                    sum(1 for k in range(count) if path.elementAt(k).isMoveTo()),
                    ink.name(),
                    ink.alpha(),
                    pen.widthF(),
                    pen.style().name != "SolidLine",
                )
            )
        return super().drawPath(path)


class PaintCapture:
    """Records painting of the Hollywood Training panels while active.

    Use it through the ``paint_capture`` fixture (built on ``monkeypatch``, so everything is
    restored after the test).
    """

    def __init__(self, monkeypatch) -> None:
        self.texts: list[Drawn] = []
        self.paths: list[DrawnPath] = []
        self.bars: list[DrawnBar] = []
        for name in PANEL_MODULES:
            module = import_module(name)
            if hasattr(module, "QPainter"):
                monkeypatch.setattr(module, "QPainter", RecordingPainter)
        monkeypatch.setattr(RecordingPainter, "capture", self)
        header = import_module("hollywood_training.gui.widgets.header")
        status = import_module("hollywood_training.gui.widgets.status_panel")
        usage = import_module("hollywood_training.gui.widgets.usage_panel")
        glow_text = self._glow(header.draw_glow_text, "HeaderBar")
        glow_mono = self._glow(status.draw_glow_mono, "StatusPanel")
        monkeypatch.setattr(header, "draw_glow_text", glow_text)
        monkeypatch.setattr(status, "draw_glow_mono", glow_mono)
        monkeypatch.setattr(status, "paint_bar", self._bar(status.paint_bar, "StatusPanel"))
        monkeypatch.setattr(usage, "paint_bar", self._bar(usage.paint_bar, "UsagePanel"))

    def _glow(self, function, panel: str):
        def wrapper(painter, position, text, *args, **kwargs):
            self.texts.append(Drawn(panel, text, "glow"))
            return function(painter, position, text, *args, **kwargs)

        return wrapper

    def _bar(self, function, panel: str):
        def wrapper(painter, rect, fraction, *args, **kwargs):
            self.bars.append(DrawnBar(panel, None if fraction is None else float(fraction)))
            return function(painter, rect, fraction, *args, **kwargs)

        return wrapper

    # ------------------------------------------------------------------ queries
    def clear(self) -> None:
        self.texts.clear()
        self.paths.clear()
        self.bars.clear()

    def strings(self, panel: str | None = None) -> list[str]:
        return [d.text for d in self.texts if panel is None or d.panel == panel]

    def joined(self, panel: str | None = None) -> str:
        return "\n".join(self.strings(panel))

    def paths_of(self, panel: str) -> list[DrawnPath]:
        return [p for p in self.paths if p.panel == panel]

    def bars_of(self, panel: str) -> list[DrawnBar]:
        return [b for b in self.bars if b.panel == panel]

    def render(self, widget: QWidget, *, full: bool = True) -> list[str]:
        """Clear, paint ``widget`` once (everything when ``full``) and return the texts that
        ``widget`` (or a panel inside it) drew - a repaint may flush other dirty panels of
        the window as well; their texts are left out."""
        self.clear()
        if full:
            force_full_repaint(widget, paint=False)
        widget.repaint()
        own = {type(w).__name__ for w in (widget, *widget.findChildren(QWidget))}
        return [d.text for d in self.texts if d.panel in own]


def force_full_repaint(widget: QWidget, *, paint: bool = True) -> None:
    """Drop the cached static layers of ``widget`` (and its children); repaint it unless
    ``paint`` is False."""
    for each in (widget, *widget.findChildren(QWidget)):
        for key in CACHE_KEYS:
            if hasattr(each, key):
                setattr(each, key, None)
    if paint:
        widget.repaint()

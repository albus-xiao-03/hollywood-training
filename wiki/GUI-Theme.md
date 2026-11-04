# Theme

`hollywood_training/gui/theme.py` · part of [GUI](GUI)

This module defines the visual identity of the window: a blue-based palette, monospace
fonts with a cross-platform fallback list, the stylesheet for the standard Qt widgets,
and painting helpers for the HUD look (chamfered frames, neon glow text, glow sprites and
the window icon). Effects that are drawn every frame come from cached pixmaps, which
keeps them cheap. No font or image files are bundled.

## Responsibilities

* Name every colour of the GUI in one frozen `Palette`, which also meets the SPEC's "blue
  based" requirement.
* Pick a monospace font family that exists on the current system.
* Style the standard widgets inside the main window (terminal, tooltips, scroll bars,
  splitter handles).
* Provide the drawing primitives the panels share, and cache the expensive ones.

## API

```python
@dataclass(frozen=True, slots=True)
class Palette:
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
```

| Field | Used for |
|---|---|
| `void` | darkest background: window, bar troughs, hover box |
| `background_top`, `background_bottom` | the backdrop gradient |
| `panel` | panel fill, tooltip background |
| `panel_edge` | panel outlines, bar frames, chart axes |
| `grid` | backdrop grid, chart grid, separators, input border |
| `accent` | cyan: primary highlight, positive weights, training loss |
| `accent_soft` | secondary highlights: header details, layer badges |
| `electric` | bar gradient start, `PAUSED` badge, text selection |
| `indigo` | negative weights, validation loss |
| `ice` | brightest text, neuron cores |
| `text` | normal text (terminal `info`) |
| `text_dim` | labels, captions, dim terminal lines |
| `success`, `warning`, `error` | status accents: terminal levels, state badge, best-epoch marker |

```python
MONO_FAMILIES = (
    "JetBrains Mono", "Cascadia Mono", "Cascadia Code", "SF Mono", "Menlo", "Consolas",
    "Fira Code", "Source Code Pro", "Hack", "DejaVu Sans Mono", "Ubuntu Mono",
    "Liberation Mono", "Noto Sans Mono", "Courier New",
)

@lru_cache(maxsize=1)
def mono_family() -> str
def mono_font(point_size: float = 10.0, *, bold: bool = False, spacing: float = 0.0) -> QFont
```

* `mono_family()` returns the first family in `MONO_FAMILIES` that is installed
  (`QFontDatabase.families()`), or else the family of the system's fixed-pitch font. It
  needs a running `QGuiApplication`, and it is computed once per process.
* `mono_font()` returns a `QFont` of that family at `point_size`, with the `TypeWriter`
  style hint and fixed pitch, bold weight when `bold`, and an absolute letter spacing of
  `spacing` when it is not 0.

```python
def color(name: str, alpha: int = 255) -> QColor
def mix(a: QColor, b: QColor, t: float) -> QColor
```

* `color()` returns `QColor(name)` with `alpha` clamped to 0-255.
* `mix()` blends the RGBA components linearly. `t` is clamped to 0-1, and `0` gives `a`.

```python
def stylesheet() -> str
def chamfer_path(rect: QRectF, cut: float) -> QPainterPath
```

* `stylesheet()` returns the Qt style sheet for `QMainWindow`, `QToolTip`,
  `QPlainTextEdit#terminalOutput`, `QLineEdit#terminalInput` (plus `:focus`),
  `QLabel#terminalPrompt`, the vertical `QScrollBar` and `QSplitter::handle`.
  `MainWindow` sets it on itself, not on the application.
* `chamfer_path()` returns a rectangle whose top-left and bottom-right corners are cut
  off by `cut` px (the HUD frame shape).

```python
def draw_glow_text(
    painter: QPainter,
    position: QPointF,
    text: str,
    font: QFont,
    fill: QColor,
    glow: QColor,
    strength: float = 1.0,
) -> None
def glow_text_width(text: str, font: QFont) -> float
def draw_glow_mono(
    painter: QPainter,
    position: QPointF,
    text: str,
    font: QFont,
    fill: QColor,
    glow: QColor,
    strength: float = 1.0,
) -> None
def glow_sprite(radius: int, tint: QColor, *, core: float = 0.25, dpr: float = 1.0) -> QPixmap
def app_icon() -> QIcon
```

* `draw_glow_text()` draws `text` as a path. It strokes three halos in `glow` (8, 4.5
  and 2 px wide, with alpha 18, 38 and 70, each multiplied by `strength`) and then fills
  the path with `fill`. `position` is the start of the baseline. It is meant for
  headings.
* `draw_glow_mono()` gives the same look, built from cached per-character sprites. It is
  meant for text that changes every frame (timers, counters) and assumes a monospace
  `font`. Every character advances by the width of one `0`, and whitespace is skipped
  but still advances. `position` is the left end of the baseline.
* `glow_text_width()` returns the width of `text` as `draw_glow_mono()` draws it.
* `glow_sprite()` returns a cached radial glow `2 × radius` px wide: white at the centre
  (with the tint's alpha), `tint` at `core`, and transparent at the edge. `radius` is
  truncated to an int of at least 1, and `dpr` is rounded to 2 decimals.
* `app_icon()` returns a 64 × 64 window icon drawn at runtime: a dark disc with a cyan
  ring and a three-node network.

## How it works

| Cache | Size | Key |
|---|---|---|
| `mono_family()` | 1 | (none) |
| `_glyph(...)`, used by `draw_glow_mono` | 1024 sprites | character, family, size, weight, spacing, fill, glow, strength, DPR |
| `_sprite(...)`, used by `glow_sprite` | 64 sprites | radius, tint, core, DPR |

Glyph sprites are rendered with `draw_glow_text` at the device pixel ratio of the painter's
device, with 6 px of padding for the halo (`_GLOW_PAD`). After the first frame, a
ticking timer only blits pixmaps. The [Performance](Performance) page has the
measurements.

## Used by / uses

* Used by every panel ([HUD Panel](GUI-HUD-Panel), [Header](GUI-Header),
  [Network View](GUI-Network-View), [Loss Chart](GUI-Loss-Chart),
  [Status Panel](GUI-Status-Panel), [Usage Panel](GUI-Usage-Panel),
  [Terminal](GUI-Terminal)), and by the [Main Window](GUI-Main-Window) for the
  stylesheet, the icon and the backdrop colours.
* Uses only PySide6 (`QtCore`, `QtGui`).

## Example

```python
from PySide6.QtCore import QPointF
from PySide6.QtGui import QPainter
from PySide6.QtWidgets import QWidget

from hollywood_training.gui.theme import PALETTE, color, draw_glow_mono, draw_glow_text, mono_font


class Readout(QWidget):
    def paintEvent(self, _event) -> None:
        painter = QPainter(self)
        painter.fillRect(self.rect(), color(PALETTE.void))
        draw_glow_text(painter, QPointF(12, 30), "SIGNAL", mono_font(14, bold=True),
                       color(PALETTE.ice), color(PALETTE.accent))
        draw_glow_mono(painter, QPointF(12, 62), "00:01:23.456", mono_font(16, bold=True),
                       color(PALETTE.accent), color(PALETTE.accent), 0.7)
        painter.end()
```

## Notes

* `mono_family()` caches its first result, so it must first be called after the
  `QApplication` exists. The panels call it (through `mono_font`) while they paint, and
  the terminal also calls it in its constructor.
* Palette values are hex strings. `color()` converts them where a `QColor` is needed, and
  `paint_bar` takes them as names ([HUD Panel](GUI-HUD-Panel)).

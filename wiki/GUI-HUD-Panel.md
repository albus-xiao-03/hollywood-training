# HUD Panel

`hollywood_training/gui/widgets/hud.py` · part of [GUI](GUI)

`HudPanel` is the framed, titled base widget for every section of the window except the
header. It draws a translucent blue body with chamfered corners, bright corner brackets,
and an uppercase title with a small panel code. The module also provides `paint_bar`,
the HUD progress and level bar used by the status and usage panels.

## Responsibilities

* Draw the same frame and title strip for every panel.
* Reserve the title area through the widget's contents margins, so painted content and
  child widgets start below it.
* Draw determinate and indeterminate bars with a moving highlight.

## API

```python
TITLE_HEIGHT = 26
CUT = 10.0
```

`TITLE_HEIGHT` is the height of the title strip in px (the top contents margin is
`TITLE_HEIGHT + 6`). `CUT` is the size of the chamfered corners.

```python
def paint_bar(
    painter: QPainter,
    rect: QRectF,
    fraction: float | None,
    time: float,
    *,
    start: str = PALETTE.electric,
    end: str = PALETTE.accent,
    ticks: int = 10,
) -> None
```

* `fraction`: the filled part, clamped to 0-1. `None` draws stripes that move at
  40 px/s, for unknown progress.
* `time`: the animation time in seconds. It moves the stripes and the highlight.
* `start`, `end`: palette colour names (hex strings). The fill gradient between them
  spans the full bar width.
* `ticks`: the number of segments, separated by `ticks - 1` dark lines.

The painter state is saved and restored.

```python
class HudPanel(QWidget):
    def __init__(self, title: str, code: str = "", parent: QWidget | None = None) -> None

    @property
    def title(self) -> str
    @property
    def code(self) -> str
    def set_code(self, code: str) -> None
    def content_rect(self) -> QRectF
    def paintEvent(self, _event) -> None
    def paint_content(self, painter: QPainter, rect: QRectF) -> None
    def paint_frame(self, painter: QPainter) -> None
```

* `title` and `code` are stored upper-cased. The read-only properties `title` and
  `code` return them.
* `set_code(code)` replaces the code, the small text at the right of the title strip. It
  stores it upper-cased and calls `update()`.
* Contents margins are 12 px left and right, `TITLE_HEIGHT + 6` at the top and 10 px at
  the bottom.
* `content_rect()` returns `contentsRect()` as a `QRectF`.
* `paintEvent()` creates an antialiased painter, calls `paint_frame()`, then
  `paint_content(painter, content_rect())`.
* `paint_content()` is the method to override for drawing content. The base does
  nothing.
* `paint_frame()` draws the frame and the title strip (below).

## How it works

**Frame** (`paint_frame`), on the widget rect inset by 1.5 px:

1. `chamfer_path(rect, CUT)` cuts the top-left and bottom-right corners.
2. A vertical gradient of `panel` (alpha 200 to 135) fills it, and `panel_edge`
   (alpha 150) outlines it.
3. Corner brackets (2 px wide, 14 px arms, `accent`) mark the two square corners
   (top-right and bottom-left). 1.5 px highlight lines run along the two cut corners.
4. The title strip shows `▸ TITLE` in an 8.5 pt bold monospace font (letter spacing 2),
   18 px below the top edge. The code is right-aligned in 7.5 pt `text_dim`. Between
   them runs a cyan underline that fades to transparent. It ends 12 px before the code,
   using the code's measured width, or 18 px from the right edge when there is no code.

**Bar** (`paint_bar`): an outlined trough (a `panel_edge` frame with a translucent `void`
fill), with the content clipped 2 px inside. It then draws either the moving stripes in
the `start` colour, or the gradient fill plus a 36 px `ice` highlight that sweeps along
the filled part about every 1.8 s (when the fill is wider than 6 px). The separators come
last.

**Ways to build on it:**

| Approach | Panels |
|---|---|
| override `paint_content(painter, rect)` | [Status Panel](GUI-Status-Panel), [Usage Panel](GUI-Usage-Panel) |
| put child widgets in a layout, which respects the contents margins | [Terminal](GUI-Terminal) |
| override `paintEvent` and call `paint_frame` while rendering a cached pixmap | [Network View](GUI-Network-View), [Loss Chart](GUI-Loss-Chart) |

## Used by / uses

* Used by every panel listed above, and `paint_bar` by the status and usage panels.
* Uses [Theme](GUI-Theme) (`PALETTE`, `chamfer_path`, `color`, `mono_font`).

## Example

```python
from PySide6.QtCore import QPointF, QRectF
from PySide6.QtGui import QPainter

from hollywood_training.gui.theme import PALETTE, color, mono_font
from hollywood_training.gui.widgets import HudPanel, paint_bar


class AccuracyPanel(HudPanel):
    def __init__(self) -> None:
        super().__init__("Accuracy", "ACC-06")      # shown as "▸ ACCURACY" and "ACC-06"
        self.value: float | None = None             # None: striped "unknown" bar
        self.time = 0.0                             # advance it from an AnimationClock

    def paint_content(self, painter: QPainter, rect: QRectF) -> None:
        painter.setFont(mono_font(8.0, bold=True))
        painter.setPen(color(PALETTE.text_dim))
        painter.drawText(QPointF(rect.left(), rect.top() + 10), "VALIDATION")
        bar = QRectF(rect.left(), rect.top() + 16, rect.width(), 10)
        paint_bar(painter, bar, self.value, self.time, end=PALETTE.success)
```

## Notes

* The code is read while painting, so `set_code()` shows up on the next repaint.
  `NetworkView` sets its code directly while it renders its cached frame: `NET-01`
  without an architecture, otherwise its layer and parameter count, such as
  `3 LAYERS · 403 PARAMS`.
* The panel caches nothing itself. Panels that redraw often over a static frame cache it
  themselves (see the last row of the table above).

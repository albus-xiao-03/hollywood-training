# Loss Chart

`hollywood_training/gui/widgets/loss_chart.py` · part of [GUI](GUI)

`LossChart` plots the training and validation loss of every completed epoch (SPEC GUI 3).
Its x axis spans all requested epochs, so the curves grow toward the right edge as the
run progresses. Axes, grid and curves are rendered into a cached pixmap when a new epoch
arrives, which happens during the synchronous hand-off. Pulsing markers at the newest
points and the hover read-out are drawn on top every frame. The `scale` command (or a
click) switches the loss axis between linear and logarithmic.

## Responsibilities

* Draw `session.history` as a training curve and a validation curve, and mark the best
  epoch.
* Choose readable axis ranges and ticks for a linear or a logarithmic loss axis.
* Rebuild the plot only when the history, the epoch total, the scale, the size or the
  device pixel ratio change.
* Show the values of the hovered epoch.

## API

```python
def nice_ticks(low: float, high: float, count: int = 5) -> list[float]
```

Returns round tick values covering `[low, high]` with about `count` intervals. The step
is 1, 2, 2.5, 5 or 10 times a power of ten. When the span is not finite or not positive,
it returns `[low]`. For example, `nice_ticks(0, 1, 5)` is `[0.0, 0.2, 0.4, 0.6, 0.8, 1.0]`
and `nice_ticks(0, 2.6, 5)` is `[0.0, 0.5, 1.0, 1.5, 2.0, 2.5]`.

```python
def decade_ticks(low: float, high: float) -> list[float]
```

Returns the powers of ten from `low` to `high`, inclusive (the exponents are rounded).
For example, `decade_ticks(0.01, 10)` is `[0.01, 0.1, 1.0, 10.0]`.

```python
@dataclass(slots=True)
class Axes:
    plot: QRectF
    x0: float
    x1: float
    y0: float
    y1: float
    log: bool

    def x(self, epoch: float) -> float
    def y(self, value: float) -> float
    def visible(self, value: float) -> bool
```

`Axes` maps epochs (`x0` to `x1`) and loss values (`y0` to `y1`) into the `plot`
rectangle. With `log`, `y()` maps `log10(value)` and clamps values below `y0` to the
bottom edge. `visible()` is `True` for finite values, and in log mode only for values
above 0.

```python
class LossChart(HudPanel):
    def __init__(
        self, session: SessionState, clock: AnimationClock, parent: QWidget | None = None
    ) -> None

    @property
    def log_scale(self) -> bool
    def set_log_scale(self, enabled: bool) -> None
    def has_positive_loss(self) -> bool
```

The chart draws `session.history`. Its title is "Loss", its minimum size 320 × 200. It
has a cross cursor and the tooltip `Click: switch between linear and logarithmic scale`.
`set_log_scale()` invalidates the cache and schedules a repaint; the `scale` command
calls it ([Main Window](GUI-Main-Window)). `has_positive_loss()` tells whether a
logarithmic axis can be drawn: some training or validation loss is finite and above 0.
The attribute
`last_painted_epoch: int | None` holds the epoch of the last history record at the
latest paint.

## How it works

**Data.** The epochs and training losses come from `session.history`. The validation
losses come from the same records, with `NaN` for epochs without one.

**Axes** (`_make_axes`), from all finite losses:

| | Linear | Logarithmic (needs at least one positive loss) |
|---|---|---|
| x range | 1 to `max(total_epochs, last epoch, 2)` | the same |
| y range | `min(0, smallest loss)` to the largest loss + 8 % | the decade below the smallest positive loss to the decade above the largest |
| y ticks | `nice_ticks(y0, y1, 5)`, labelled with `{:.4g}` | `decade_ticks(y0, y1)`, labelled with `{:.0e}` |
| x ticks | the whole numbers ≥ 1 among `nice_ticks(x0, x1, 6)` | the same |

When log scale is on but no loss is positive, the axes fall back to linear, and the legend
says `LIN`. When there is no finite loss at all, the plot shows `AWAITING FIRST EPOCH`.

**Static layer.** The cache key holds `history_version`, `total_epochs`, the log flag,
the width, the height and the device pixel ratio. The pixmap holds:

* the frame, the grid, tick labels, the axes and the `EPOCH` label;
* the training curve, a 2 px cyan line with a 6 px translucent glow, over an area filled
  with cyan that fades downward;
* the validation curve, dashed indigo with the same glow;
* a green diamond at the best epoch, from `SessionState.best_record()`. That is the
  record with the lowest finite `EpochRecord.monitored_loss` (its validation loss, or its
  training loss when it has none), and there is no marker while no such loss exists;
* the legend `TRAIN <last>`, `VAL <last>`, `BEST <loss> @ <epoch>`, and the scale that
  is actually drawn: `LOG` only when the axes are logarithmic, `LIN` otherwise.

A value that is not visible (not finite, or 0 or less in log mode) breaks the curve.

**Overlay**, every frame once there is history: a pulsing ring and a dot at the newest
training and validation points. While the mouse is inside the plot, a dashed vertical
line marks the nearest recorded epoch, with a box showing `EPOCH n`, `TRAIN ...` and
`VAL ...`.

**Interaction.** The `scale [lin|log]` command sets the log scale, or toggles it without
an argument. A left click toggles it too. Moving the mouse sets the hover
point, and leaving the widget clears it. A resize clears the cache key.

## Used by / uses

* Used by the [Main Window](GUI-Main-Window) as `window.chart`. It is repainted by
  `render_now()` during every synchronous hand-off.
* Uses [Session State](Sync-Session-State) (`history`, `history_version`,
  `total_epochs`, `best_record()`), [Sync Events](Sync-Events)
  (`EpochRecord.monitored_loss`), [HUD Panel](GUI-HUD-Panel), [Theme](GUI-Theme),
  [Animation](GUI-Animation) (`pulse`) and [Formatting](Formatting) (`loss`).

## Example

```python
from hollywood_training.gui.widgets import nice_ticks

chart = window.chart          # the LossChart of a MainWindow
chart.set_log_scale(True)     # what `scale log` does
assert chart.log_scale

nice_ticks(1, 400, 6)         # [50, 100, 150, 200, 250, 300, 350, 400]: the x ticks of a 400-epoch run
```

## Notes

* The x axis always starts at epoch 1 and covers at least 2 epochs, so a single point is
  drawn on a valid scale.
* Raising the epoch total with the `epochs` command changes `total_epochs`, which
  rebuilds the chart with a longer x axis.

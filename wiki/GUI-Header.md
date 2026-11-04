# Header

`hollywood_training/gui/widgets/header.py` · part of [GUI](GUI)

`HeaderBar` is the 80 px strip across the top of the window. It shows the product title
and version, the training network's class name in large glowing letters (SPEC GUI 6),
the network's display name and parameter count, a state badge, and the training device.
Text that rarely changes is rendered once into a cached pixmap. The rotating emblem and
the state badge are drawn every frame.

## Responsibilities

* Show `SessionInfo.network_class` prominently, centred in the window.
* Show the session state as a coloured badge that blinks while the state is active.
* Keep the expensive glow text out of the per-frame work.

## API

```python
STATE_BADGES: dict[TrainingState, tuple[str, str, bool]] = {
    TrainingState.PREPARING: ("BOOTING", PALETTE.accent_soft, True),
    TrainingState.RUNNING: ("TRAINING", PALETTE.accent, True),
    TrainingState.PAUSED: ("PAUSED", PALETTE.electric, True),
    TrainingState.FINISHED: ("COMPLETE", PALETTE.success, False),
    TrainingState.STOPPED: ("STOPPED", PALETTE.text, False),
    TrainingState.FAILED: ("FAILURE", PALETTE.error, True),
}
```

The badge for each state, as `(text, colour, blinks)`. A state missing from the table
shows `—` without blinking.

```python
class HeaderBar(QWidget):
    HEIGHT = 80

    def __init__(
        self, session: SessionState, clock: AnimationClock, parent: QWidget | None = None
    ) -> None
```

The bar reads `session` on every paint and connects to `clock.tick`. Its height is fixed
to `HEIGHT`. It is a plain `QWidget`, not a [HUD Panel](GUI-HUD-Panel).

## How it works

| Element | Where | Source | Drawn |
|---|---|---|---|
| background strip and a bottom cyan line that fades at both ends | full width | (none) | cached |
| `HOLLYWOOD TRAINING` (15 pt bold, glow) | left | (none) | cached |
| `NEURAL TRAINING VISUALIZER · v<version>` | under the title | `hollywood_training._version.__version__` | cached |
| `TRAINING NETWORK` label | centre, top | (none) | cached |
| network class name (19 pt bold, glow strength 1.2) | centre | `session.network_name`, or `—` before a session | cached |
| display name and `N parameters`, separated by `·` | centre, below the name | `info.display_name`, `info.architecture.total_parameters` | cached |
| emblem: two outer arcs turning at 140°/s, an inner arc turning the other way at 220°/s, and a pulsing core | left edge | animation time | every frame |
| state badge, whose outline, fill and dot pulse when it blinks (period 1.2 s) | top right | `session.state` and `STATE_BADGES` | every frame |
| `DEVICE <device>`, upper-cased | right, under the badge | `info.device` | every frame, once a session exists |

* The cache key holds the network name, display name, parameter count, width, height
  and device pixel ratio. Any change rebuilds the pixmap on the next paint.
* The details line leaves out parts that are missing (no display name, or a parameter
  count of 0).
* For every synchronous event, `MainWindow.render_now()` repaints the header too. The
  badge therefore shows `PAUSED`, or the final state, before the frontend acknowledges
  the pause or the end of the session.

## Used by / uses

* Used by the [Main Window](GUI-Main-Window) as `window.header`.
* Uses [Session State](Sync-Session-State), [Theme](GUI-Theme) (`draw_glow_text`,
  `mono_font`, `PALETTE`), [Animation](GUI-Animation) (`pulse`) and
  [Formatting](Formatting) (`count`).

## Example

```python
from PySide6.QtWidgets import QApplication

from hollywood_training.gui import AnimationClock
from hollywood_training.gui.widgets import HeaderBar
from hollywood_training.sync import SessionState

app = QApplication([])
session = SessionState()
clock = AnimationClock()
header = HeaderBar(session, clock)   # "—" and the BOOTING badge until a session starts
session.apply(started)               # a SessionStarted event
header.repaint()                     # cache rebuilt with the class name and parameters
clock.start()                        # emblem and badge animate
```

## Notes

* Positions are fixed pixel offsets: the title starts at x = 86 and the emblem is
  centred at x = 42. The class name is centred and not elided.
* Device text such as `cpu (5 threads)` comes from
  [Training Descriptions](Training-Descriptions) and is shown upper-cased.

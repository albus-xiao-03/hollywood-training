# Animation

`hollywood_training/gui/animation.py` · part of [GUI](GUI)

Each window has one `AnimationClock`, a precise QTimer at about 60 fps, and it drives
every animated panel. All panels therefore move in step, and only one timer is needed.
The module also has small helpers for smoothing and pulsing that do not depend on the
frame rate. Animations are purely visual. The training thread waits only for the
synchronous epoch hand-off, never for an animation frame.

## Responsibilities

* Emit `tick(dt)` about 60 times a second, with the real time since the previous tick.
* Smooth values toward a target independently of the frame rate (`approach`).
* Provide a smooth periodic curve for pulsing effects (`pulse`).

## API

```python
class AnimationClock(QObject):
    tick = Signal(float)

    def __init__(self, fps: int = 60, parent: QObject | None = None) -> None

    @property
    def running(self) -> bool
    def start(self) -> None
    def stop(self) -> None
```

* `tick(dt)`: the seconds since the previous tick, clamped to 0-0.1 s.
* `fps`: sets the timer interval to `max(1, round(1000 / fps))` ms, which is 17 ms for
  60, using `Qt.TimerType.PreciseTimer`.
* `time`: an attribute holding the animation seconds accumulated over all ticks. It
  starts at `0.0`.
* `running`: whether the timer is active.
* `start()` restarts the elapsed-time measurement and starts the timer. `stop()` stops
  it.

```python
def approach(current: float, target: float, dt: float, half_life: float) -> float
```

Exponential smoothing. It returns `target + (current - target) * 0.5 ** (dt / half_life)`,
so the remaining distance halves every `half_life` seconds at any frame rate. A
`half_life` of 0 or less returns `target`.

```python
def pulse(time: float, period: float, phase: float = 0.0) -> float
```

Returns `0.5 + 0.5 * sin(2π · (time / period + phase))`, a smooth oscillation between 0
and 1 with a period of `period` seconds.

## How it works

`MainWindow` creates `AnimationClock(60, self)` and starts it in its constructor.
`MainWindow.closeEvent` stops it. Each animated panel connects `tick` to its own
`_advance(dt)`. That method adds `dt` to the panel's own time, updates the panel's
animation state and calls `update()`. Qt coalesces those calls into one paint per panel
per frame.

| Subscriber | What the tick animates |
|---|---|
| [Header](GUI-Header) | rotating emblem arcs, pulsing core (`pulse`, 1.1 s), blinking state badge (1.2 s) |
| [Network View](GUI-Network-View) | neuron halos, signal particles (speed scaled by the training state), energy decay, scanning band, the pulse of the waiting text (1.6 s) |
| [Loss Chart](GUI-Loss-Chart) | pulsing rings at the newest points (1.3 s). It repaints only once there is history |
| [Status Panel](GUI-Status-Panel) | process timer (read from `SessionState.elapsed_now()`), moving bar highlights and stripes |
| [Usage Panel](GUI-Usage-Panel) | bar levels eased toward each reading (`approach`, half-life 0.12 s), bar highlights |

The epoch hand-off does not go through the clock. `MainWindow.render_now()` calls
`repaint()` directly, so a synchronous event is drawn at once and not at the next tick.

## Used by / uses

* Used by the [Main Window](GUI-Main-Window), which owns the clock, and by the panels
  listed above.
* Uses PySide6 `QTimer`, `QElapsedTimer` and `Signal`.

## Example

```python
from PySide6.QtWidgets import QApplication, QWidget

from hollywood_training.gui.animation import AnimationClock, approach, pulse


class Gauge(QWidget):
    def __init__(self, clock: AnimationClock) -> None:
        super().__init__()
        self.level, self.target, self.t = 0.0, 0.8, 0.0
        clock.tick.connect(self._advance)

    def _advance(self, dt: float) -> None:
        self.t += dt
        self.level = approach(self.level, self.target, dt, 0.12)
        self.update()   # paintEvent can use pulse(self.t, 1.2) for a breathing glow


app = QApplication([])
clock = AnimationClock(60)
gauge = Gauge(clock)
gauge.show()
clock.start()
app.exec()
```

## Notes

* **Animations never delay training.** In the default setup the clock runs in the window
  process. In the [In-Process Frontend](GUI-In-Process-Frontend), its ticks are delivered
  only while the training loop pumps Qt, between batches or while waiting.
* The 0.1 s clamp on `dt` keeps animations from jumping after a stall, such as a long
  batch in the in-process frontend.
* The current panels do not read `AnimationClock.time`. Each panel keeps its own time.

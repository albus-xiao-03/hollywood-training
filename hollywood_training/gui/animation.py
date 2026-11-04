"""Animation timing: one shared clock drives every animated widget, plus easing helpers.

A single ``AnimationClock`` (a precise ~60 fps ``QTimer``) keeps all panels in step and
costs one timer instead of one per widget. Animations are purely visual: they never
delay training (the training thread only waits for the synchronous epoch hand-off).
"""

from __future__ import annotations

import math

from PySide6.QtCore import QElapsedTimer, QObject, Qt, QTimer, Signal


class AnimationClock(QObject):
    """Emits :attr:`tick` with the seconds elapsed since the previous tick."""

    tick = Signal(float)

    def __init__(self, fps: int = 60, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._timer = QTimer(self)
        self._timer.setTimerType(Qt.TimerType.PreciseTimer)
        self._timer.setInterval(max(1, round(1000 / fps)))
        self._timer.timeout.connect(self._advance)
        self._elapsed = QElapsedTimer()
        self._last_ms = 0
        self.time = 0.0  # seconds of animation time since start

    @property
    def running(self) -> bool:
        return self._timer.isActive()

    def start(self) -> None:
        self._elapsed.start()
        self._last_ms = 0
        self._timer.start()

    def stop(self) -> None:
        self._timer.stop()

    def _advance(self) -> None:
        now = self._elapsed.elapsed()
        dt = min(0.1, max(0.0, (now - self._last_ms) / 1000.0))  # clamp after stalls
        self._last_ms = now
        self.time += dt
        self.tick.emit(dt)


def approach(current: float, target: float, dt: float, half_life: float) -> float:
    """Frame-rate independent exponential smoothing toward ``target``."""
    if half_life <= 0:
        return target
    return target + (current - target) * 0.5 ** (dt / half_life)


def pulse(time: float, period: float, phase: float = 0.0) -> float:
    """Smooth 0→1→0 oscillation with the given period (seconds)."""
    return 0.5 + 0.5 * math.sin(2.0 * math.pi * (time / period + phase))


def ease_out_cubic(t: float) -> float:
    t = max(0.0, min(1.0, t))
    return 1.0 - (1.0 - t) ** 3

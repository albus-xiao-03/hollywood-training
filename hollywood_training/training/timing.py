"""``ActiveClock``: wall time spent actively training, excluding pauses."""

from __future__ import annotations

import time


class ActiveClock:
    """A pausable stopwatch based on ``time.perf_counter``."""

    def __init__(self) -> None:
        self._accumulated = 0.0
        self._started_at: float | None = None

    @property
    def running(self) -> bool:
        return self._started_at is not None

    def start(self) -> None:
        if self._started_at is None:
            self._started_at = time.perf_counter()

    def pause(self) -> None:
        if self._started_at is not None:
            self._accumulated += time.perf_counter() - self._started_at
            self._started_at = None

    def elapsed(self) -> float:
        if self._started_at is None:
            return self._accumulated
        return self._accumulated + (time.perf_counter() - self._started_at)

"""``HeadlessRun``: a real ``TrainingManager`` with a recording frontend, no GUI.

The manager runs on a background thread, exactly as it would under a frontend. The
recording sink folds every event into a ``SessionState`` and acknowledges synchronous
events - immediately, or after ``ack_delay`` seconds to stand in for a slow drawing. Command
lines run through the real built-in registry with a real ``CommandContext`` (from the test
thread, like the window process's command thread), so pause / step / lr / ... are tested
against the real control flow.
"""

from __future__ import annotations

import threading
import time
import weakref
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from hollywood_training.commands.builtin import create_registry
from hollywood_training.commands.context import CommandContext
from hollywood_training.commands.registry import CommandResult
from hollywood_training.sync.channel import Envelope, SyncChannel
from hollywood_training.sync.control import TrainingControl, TrainingTarget
from hollywood_training.sync.session import SessionState
from hollywood_training.tracking.tracker import NetworkTracker
from hollywood_training.training.manager import TrainingManager

_STARTED: weakref.WeakSet = weakref.WeakSet()  # runs started (see stop_leftover_runs)


def stop_leftover_runs(timeout: float = 10.0) -> int:
    """Stop every run still training (a test that failed half-way leaves one behind,
    which would otherwise keep a core busy for the following tests); how many were."""
    leftover = [run for run in list(_STARTED) if run.alive]
    for run in leftover:
        run.control.stop()
        run.channel.close()  # release a hand-off nobody will acknowledge any more
    for run in leftover:
        if run._thread is not None:
            run._thread.join(timeout)
    return len(leftover)


class FakeFrontend:
    """``FrontendActions`` that records what commands ask of the frontend."""

    def __init__(self, view_answer: Callable[[Any], str | None] | None = None) -> None:
        self.calls: list[tuple] = []
        self._view_answer = view_answer

    def clear_output(self) -> None:
        self.calls.append(("clear",))

    def close(self) -> None:
        self.calls.append(("close",))

    def screenshot(self, path) -> None:
        self.calls.append(("screenshot", path))
        return None

    def view(self, request) -> str | None:
        self.calls.append(("view", request))
        return self._view_answer(request) if self._view_answer else None


@dataclass(frozen=True)
class Received:
    time: float  # time.perf_counter() when the sink received it
    envelope: Envelope

    @property
    def event(self) -> Any:
        return self.envelope.event


class HeadlessRun:
    """Run ``process`` under a real manager on a background thread."""

    def __init__(
        self,
        process,
        *,
        linger: bool = False,
        start_paused: bool = False,
        ack_delay: float | Callable[[Any], float] = 0.0,
        max_neurons: int = 20,
        frontend: Any = None,
        progress_interval: float = 1 / 30,
    ) -> None:
        self.process = process
        self.channel = SyncChannel()
        self.control = TrainingControl(process.epochs, start_paused=start_paused)
        self.tracker = NetworkTracker(process.model, max_neurons=max_neurons)
        self.session = SessionState()
        self.received: list[Received] = []
        self.acks: list[tuple[float, int]] = []
        self._ack_delay = ack_delay
        self._condition = threading.Condition()
        self.channel.connect(self._sink)
        self.manager = TrainingManager(
            process,
            channel=self.channel,
            control=self.control,
            tracker=self.tracker,
            linger=linger,
            progress_interval=progress_interval,
        )
        self.registry = create_registry(process.commands)
        self.frontend = frontend if frontend is not None else FakeFrontend()
        self.context = CommandContext(
            control=self.control,
            session=self.session,
            registry=self.registry,
            target=TrainingTarget(process.model, process.optimizer),
            frontend=self.frontend,
        )
        self.result = None
        self.exception: BaseException | None = None
        self.thread_id: int | None = None
        self._thread: threading.Thread | None = None

    # ------------------------------------------------------------------ sink
    def _sink(self, envelope: Envelope) -> None:
        now = time.perf_counter()
        with self._condition:
            self.session.apply(envelope.event)
            self.received.append(Received(now, envelope))
            self._condition.notify_all()
        if envelope.synchronous:
            delay = self._ack_delay
            if callable(delay):
                delay = delay(envelope.event)
            if delay:
                time.sleep(delay)
            with self._condition:
                self.acks.append((time.perf_counter(), envelope.sequence))
            self.channel.acknowledge(envelope.sequence)

    # ---------------------------------------------------------------- running
    def start(self) -> HeadlessRun:
        def target() -> None:
            self.thread_id = threading.get_ident()
            try:
                self.result = self.manager.run()
            except BaseException as exc:  # recorded for the test
                self.exception = exc
            finally:
                with self._condition:
                    self._condition.notify_all()

        self._thread = threading.Thread(target=target, name="htspec-training", daemon=True)
        _STARTED.add(self)
        self._thread.start()
        return self

    def run(self, timeout: float = 60.0):
        """Start, wait for the end and return the ``TrainingResult``."""
        return self.start().join(timeout)

    def join(self, timeout: float = 60.0):
        assert self._thread is not None, "start() first"
        self._thread.join(timeout)
        if self._thread.is_alive():
            self.control.stop()
            self._thread.join(10)
            raise AssertionError(f"the training session did not end within {timeout} s")
        if self.exception is not None:
            raise self.exception
        return self.result

    @property
    def alive(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def stop(self, timeout: float = 30.0):
        self.control.stop()
        return self.join(timeout)

    # ----------------------------------------------------------------- waiting
    def wait_for(self, predicate: Callable[[HeadlessRun], bool], timeout: float = 20.0) -> None:
        deadline = time.monotonic() + timeout
        with self._condition:
            while not predicate(self):
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise AssertionError(f"condition not reached within {timeout} s")
                if self._thread is not None and not self._thread.is_alive():
                    if predicate(self):
                        return
                    raise AssertionError("the training session ended before the condition")
                self._condition.wait(min(remaining, 0.05))

    def wait_events(self, kind: type, count: int = 1, timeout: float = 20.0) -> list[Any]:
        self.wait_for(lambda run: len(run.events(kind)) >= count, timeout)
        return self.events(kind)

    def events(self, kind: type | None = None) -> list[Any]:
        with self._condition:
            return [r.event for r in self.received if kind is None or isinstance(r.event, kind)]

    def times(self, kind: type) -> list[float]:
        with self._condition:
            return [r.time for r in self.received if isinstance(r.event, kind)]

    # ---------------------------------------------------------------- commands
    def command(self, line: str) -> CommandResult:
        return self.registry.execute(line, self.context)

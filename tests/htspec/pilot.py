"""``Pilot``: drives a real ``ht.run(process, isolate_gui=False)`` from the keyboard.

``ht.run`` blocks until its window closes, and in the in-process mode the window lives on
the training thread (the test's main thread). The pilot therefore works from inside: it
wraps ``MainWindow.apply`` / ``MainWindow.render_now`` (class level, via ``monkeypatch``) to
see every event the window shows and every synchronous repaint, and runs scheduled actions
on the GUI thread right after the triggering event has been handled. Actions usually type
on the keyboard (``pilot.keys("pause<Return>")``), so commands take the real input path:
native window -> focus widget -> command line -> frontend -> registry.

A lingering run (the default ``close_on_finish=False``) ends with ``StateChanged(FINISHED)``
and waits for more epochs; ``SessionFinished`` only follows once the window is closed.
:meth:`Pilot.on_finish` fires on whichever comes first.

A watchdog closes the window after ``watchdog`` seconds; :meth:`Pilot.run` then fails the
test, so a broken run cannot hang the session. The window is deleted when ``ht.run``
returns: read its state inside actions, or use :attr:`Pilot.final_log` afterwards.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication

from hollywood_training.gui.window import MainWindow
from hollywood_training.sync.events import (
    EpochCompleted,
    SessionFinished,
    SessionStarted,
    StateChanged,
    TrainingPaused,
    TrainingState,
)

from . import keys as keymod

Action = Callable[[MainWindow], Any]


@dataclass
class Render:
    """One synchronous ``render_now`` of the window."""

    started: float  # time.perf_counter()
    ended: float
    epoch: int  # session.epoch when rendered
    network_epoch: int | None  # NetworkView.last_painted_epoch after the repaint
    chart_epoch: int | None  # LossChart.last_painted_epoch after the repaint
    weights_epoch: int | None
    weights: Any  # the WeightSnapshot the window had
    state: TrainingState


@dataclass
class _Trigger:
    test: Callable[[Any, MainWindow], bool]
    action: Action
    once: bool = True
    sync: bool = False
    fired: bool = False


@dataclass
class Pilot:
    monkeypatch: Any
    watchdog: float = 90.0
    window: MainWindow | None = None
    events: list[tuple[float, Any]] = field(default_factory=list)
    renders: list[Render] = field(default_factory=list)
    errors: list[BaseException] = field(default_factory=list)
    watchdog_fired: bool = False
    final_log: str = ""  # the terminal text when the window closed
    _triggers: list[_Trigger] = field(default_factory=list)

    def __post_init__(self) -> None:
        pilot = self
        original_apply = MainWindow.apply
        original_render = MainWindow.render_now

        def apply(window: MainWindow, event: Any) -> None:
            if pilot.window is None:
                pilot._adopt(window)
            original_apply(window, event)
            pilot.events.append((time.perf_counter(), event))
            pilot._check(event, window)

        def render_now(window: MainWindow) -> None:
            started = time.perf_counter()
            original_render(window)
            weights = window.session.weights
            pilot.renders.append(
                Render(
                    started,
                    time.perf_counter(),
                    window.session.epoch,
                    window.network.last_painted_epoch,
                    window.chart.last_painted_epoch,
                    weights.epoch if weights is not None else None,
                    weights,
                    window.session.state,
                )
            )

        self.monkeypatch.setattr(MainWindow, "apply", apply)
        self.monkeypatch.setattr(MainWindow, "render_now", render_now)

    # ------------------------------------------------------------ scheduling
    def on(
        self,
        test: Callable[[Any, MainWindow], bool],
        action: Action,
        *,
        once: bool = True,
        sync: bool = False,
    ) -> Pilot:
        """Run ``action(window)`` after the first event for which ``test(event, window)``
        (after every such event when ``once`` is False).

        By default the action runs on the next turn of the event loop, i.e. at the training
        loop's next safe point. With ``sync`` it runs right away, while the window is still
        handling the event - before training can go on (for reading the window's state at
        exactly that event, or for typing a command that must act before the next epoch).
        """
        self._triggers.append(_Trigger(test, action, once, sync))
        return self

    def on_start(self, action: Action) -> Pilot:
        return self.on(lambda e, w: isinstance(e, SessionStarted), action)

    def on_epoch(self, epoch: int, action: Action, *, sync: bool = False) -> Pilot:
        def done(event: Any, _window: MainWindow) -> bool:
            return isinstance(event, EpochCompleted) and event.record.epoch == epoch

        return self.on(done, action, sync=sync)

    def on_paused(self, action: Action, *, once: bool = True) -> Pilot:
        """When a pause takes effect (``TrainingPaused`` - once per pause; a ``StateChanged``
        with state PAUSED is also re-sent whenever a setting changes during a pause)."""
        return self.on(lambda e, w: isinstance(e, TrainingPaused), action, once=once)

    def on_state(self, state: TrainingState, action: Action, *, once: bool = True) -> Pilot:
        """After a ``StateChanged`` to ``state``. Note that the manager re-sends its current
        state whenever a setting changes (``lr``, ``epochs``, ...), so with ``once=False``
        this also fires for those; use :meth:`on_paused` to count pauses."""

        def changed(event: Any, _window: MainWindow) -> bool:
            return isinstance(event, StateChanged) and event.state is state

        return self.on(changed, action, once=once)

    def on_finish(self, action: Action) -> Pilot:
        """When training has ended: ``StateChanged(FINISHED)`` (a lingering run waits for
        more epochs then) or ``SessionFinished`` (stopped / failed / closed), first wins."""

        def ended(event: Any, _window: MainWindow) -> bool:
            if isinstance(event, SessionFinished):
                return True
            return isinstance(event, StateChanged) and event.state is TrainingState.FINISHED

        return self.on(ended, action)

    def after(self, seconds: float, action: Action) -> Action:
        """An action that runs ``action`` ``seconds`` later (on the GUI thread)."""

        def delayed(window: MainWindow) -> None:
            QTimer.singleShot(int(seconds * 1000), window, lambda: self._safely(action, window))

        return delayed

    # --------------------------------------------------------------- actions
    @staticmethod
    def keys(sequence: str, target: str | Callable[[MainWindow], Any] = "native") -> Action:
        """Type ``sequence`` (``htspec.keys`` notation). ``target``: ``"native"`` (the
        window's QWindow - the real input path), ``"window"``, ``"focus"``, or a function
        returning a widget of the window."""

        def action(window: MainWindow) -> None:
            if target == "native":
                where = keymod.native(window)
            elif target == "window":
                where = window
            elif target == "focus":
                where = QApplication.focusWidget() or window
            else:
                where = target(window)
            keymod.send(where, sequence)

        return action

    @staticmethod
    def line(text: str, target: str | Callable[[MainWindow], Any] = "native") -> Action:
        """Type ``text`` literally and press Return."""
        return Pilot.keys(keymod.escape(text) + "<Return>", target)

    @staticmethod
    def close(window: MainWindow) -> None:
        window.close()

    @staticmethod
    def chain(*actions: Action) -> Action:
        def run_all(window: MainWindow) -> None:
            for action in actions:
                action(window)

        return run_all

    # ----------------------------------------------------------------- running
    def run(self, process, **options: Any):
        """``ht.run(process, isolate_gui=False, **options)`` under the pilot."""
        import hollywood_training as ht

        options.setdefault("isolate_gui", False)
        options.setdefault("gui", True)
        try:
            result = ht.run(process, **options)
        finally:
            if self.errors:
                raise AssertionError(f"pilot action failed: {self.errors[0]!r}") from self.errors[0]
        if self.watchdog_fired:
            raise AssertionError(
                f"the pilot's watchdog closed the window after {self.watchdog} s: the scenario "
                "did not end by itself"
            )
        return result

    # ----------------------------------------------------------------- private
    def _adopt(self, window: MainWindow) -> None:
        self.window = window
        window.closing.connect(lambda: setattr(self, "final_log", window.terminal.text()))

        def fire() -> None:
            try:
                visible = self.window is not None and self.window.isVisible()
            except RuntimeError:  # already deleted
                return
            if visible:
                self.watchdog_fired = True
                self.window.close()

        QTimer.singleShot(int(self.watchdog * 1000), window, fire)  # dies with the window

    def _check(self, event: Any, window: MainWindow) -> None:
        for trigger in self._triggers:
            if trigger.fired and trigger.once:
                continue
            try:
                matched = trigger.test(event, window)
            except Exception as exc:  # a broken trigger must fail the test, not the run
                self.errors.append(exc)
                continue
            if matched:
                trigger.fired = True
                if trigger.sync:
                    self._safely(trigger.action, window)
                else:
                    QTimer.singleShot(0, window, lambda a=trigger.action: self._safely(a, window))

    def _safely(self, action: Action, window: MainWindow) -> None:
        try:
            action(window)
        except BaseException as exc:
            self.errors.append(exc)
            window.close()

    # ----------------------------------------------------------------- queries
    def of_type(self, kind: type) -> list[Any]:
        return [event for _t, event in self.events if isinstance(event, kind)]

    def log(self) -> str:
        """The terminal text: live while the window is open, else as it was when it closed."""
        try:
            if self.window is not None and self.window.isVisible():
                return self.window.terminal.text()
        except RuntimeError:  # the window has been deleted after ht.run returned
            pass
        return self.final_log

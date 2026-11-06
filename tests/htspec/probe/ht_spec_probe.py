"""Read-only probe of the Hollywood Training window process (SPEC harness).

Installed by ``sitecustomize.py`` in this folder, only in the window process and only when
``HT_SPEC_PROBE_LOG`` names a log file. It wraps a few methods of ``MainWindow``, ``Link``
and ``UsagePanel`` at class level - every wrapper calls the original unchanged - and
appends JSON lines to the log:

=============  ==============================================================================
``hello``      process start (pid)
``event``      a session event the window applied (type, epoch, state)
``render``     one synchronous ``render_now``: start/end time, the epoch the network view and
               the loss chart painted, and (``HT_SPEC_PROBE_WEIGHTS=1``) the weights drawn
``send``       a message to the training process (``ready``, ``ack``, ``command``, ``closed``)
``view``       a window-only command request and its outcome
``result``     a command result shown in the terminal
``state``      a snapshot of the window every ``HT_SPEC_PROBE_INTERVAL`` ms (default 100)
``step``       an autopilot step that ran
=============  ==============================================================================

Times are ``time.monotonic()`` (system-wide on Linux, so comparable with the training
process). The optional autopilot (``HT_SPEC_PROBE_SCRIPT``, JSON) types keys or sends mouse
input *inside* the window process at given moments - for offscreen tests of the isolated
window; see ``_Autopilot``.
"""

from __future__ import annotations

import json
import os
import sys
import threading
import time
import traceback

_lock = threading.Lock()
_log_path = os.environ.get("HT_SPEC_PROBE_LOG", "")
_record_weights = os.environ.get("HT_SPEC_PROBE_WEIGHTS") == "1"
_interval = int(os.environ.get("HT_SPEC_PROBE_INTERVAL", "100"))


def log(kind: str, **fields) -> None:
    fields["kind"] = kind
    fields.setdefault("t", time.monotonic())
    line = json.dumps(fields, default=str)
    with _lock, open(_log_path, "a", encoding="utf-8") as stream:
        stream.write(line + "\n")


def _guard(function):
    def wrapper(*args, **kwargs):
        try:
            return function(*args, **kwargs)
        except Exception:
            log("probe-error", error=traceback.format_exc())
            return None

    return wrapper


def _flag_name(value) -> str:
    name = getattr(value, "name", None)
    return name if name else str(value)


def install() -> None:
    from PySide6.QtCore import QEvent, QObject, QTimer
    from PySide6.QtGui import QGuiApplication
    from PySide6.QtWidgets import QApplication, QWidget

    from hollywood_training.gui import link as link_module
    from hollywood_training.gui import window as window_module
    from hollywood_training.gui.widgets import usage_panel as usage_module

    main_window = window_module.MainWindow
    log("hello", pid=os.getpid())

    original_init = main_window.__init__
    original_apply = main_window.apply
    original_render = main_window.render_now
    original_view = main_window.handle_view
    original_result = main_window.show_command_result
    original_send = link_module.Link.send
    original_sample = usage_module.UsagePanel.set_sample

    class PaintCounter(QObject):
        def __init__(self, parent) -> None:
            super().__init__(parent)
            self.counts: dict[str, int] = {}

        def eventFilter(self, watched, event) -> bool:  # observes only, never consumes
            if event.type() == QEvent.Type.Paint:
                name = type(watched).__name__
                self.counts[name] = self.counts.get(name, 0) + 1
            return False

    state: dict = {"samples": 0, "last_sample": None, "painter": None}

    def init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        counter = PaintCounter(self)
        for widget in (self.header, self.network, self.status, self.chart, self.usage):
            widget.installEventFilter(counter)
        state["painter"] = counter
        timer = QTimer(self)
        timer.setInterval(_interval)
        timer.timeout.connect(_guard(lambda: log("state", **_snapshot(self))))
        timer.start()
        script = os.environ.get("HT_SPEC_PROBE_SCRIPT")
        if script:
            state["autopilot"] = _Autopilot(self, json.loads(script))

    def apply(self, event):
        result = original_apply(self, event)
        _guard(_log_event)(self, event)
        pilot = state.get("autopilot")
        if pilot is not None:
            _guard(pilot.after_event)(event)
        return result

    def render_now(self):
        started = time.monotonic()
        result = original_render(self)
        _guard(_log_render)(self, started, time.monotonic())
        return result

    def handle_view(self, request):
        ok, text = original_view(self, request)
        log("view", action=request.action, args=list(request.arguments), ok=ok, text=text)
        return ok, text

    def show_command_result(self, ok, output):
        log("result", ok=ok, output=output)
        return original_result(self, ok, output)

    def send(self, *message):
        if message and message[0] in ("ready", "ack", "command", "closed"):
            log("send", msg=message[0], value=message[1] if len(message) > 1 else None)
        return original_send(self, *message)

    def set_sample(self, sample):
        state["samples"] += 1
        state["last_sample"] = sample
        return original_sample(self, sample)

    main_window.__init__ = init
    main_window.apply = apply
    main_window.render_now = render_now
    main_window.handle_view = handle_view
    main_window.show_command_result = show_command_result
    link_module.Link.send = send
    usage_module.UsagePanel.set_sample = set_sample

    # ------------------------------------------------------------------ records
    def _log_event(window, event):
        fields = {"type": type(event).__name__}
        record = getattr(event, "record", None)
        if record is not None:
            fields["epoch"] = record.epoch
        event_state = getattr(event, "state", None)
        if event_state is not None:
            fields["state"] = event_state.value
        fields["session_epoch"] = window.session.epoch
        fields["session_state"] = window.session.state.value
        log("event", **fields)

    def _log_render(window, started, ended):
        weights = window.session.weights
        fields = {
            "t0": started,
            "t1": ended,
            "epoch": window.session.epoch,
            "state": window.session.state.value,
            "network_epoch": window.network.last_painted_epoch,
            "chart_epoch": window.chart.last_painted_epoch,
            "weights_epoch": None if weights is None else weights.epoch,
            "history": len(window.session.history),
        }
        if _record_weights and weights is not None:
            fields["weights"] = {
                layer.name: {
                    "matrix": layer.matrix.tolist(),
                    "bias": None if layer.bias is None else layer.bias.tolist(),
                    "in_index": layer.in_index.tolist(),
                    "out_index": layer.out_index.tolist(),
                }
                for layer in weights.layers
            }
        log("render", **fields)

    def _rect(widget):
        top_left = widget.mapToGlobal(widget.rect().topLeft())
        return [top_left.x(), top_left.y(), widget.width(), widget.height()]

    def _snapshot(window):
        app = QApplication.instance()
        focus = app.focusWidget()
        terminal = window.terminal
        output = terminal.output
        bar = output.verticalScrollBar()
        line = terminal.input
        panels = {
            "header": window.header,
            "network": window.network,
            "terminal": terminal,
            "output": output,
            "input": line,
            "status": window.status,
            "chart": window.chart,
            "usage": window.usage,
        }
        handles = {}
        for name, splitter in window.splitters.items():
            for index in range(1, splitter.count()):
                handles[f"{name}:{index}"] = _rect(splitter.handle(index))
        native = window.windowHandle()
        widgets = [window, *window.findChildren(QWidget)]
        popups = [
            type(w).__name__ for w in app.topLevelWidgets() if w is not window and w.isVisible()
        ]
        sample = state["last_sample"]
        history = window.session.history
        counter = state["painter"]
        return {
            "win_id": int(window.winId()) if native is not None else None,
            "visible": window.isVisible(),
            "active": window.isActiveWindow(),
            "window_state": _flag_name(window.windowState()),
            "geometry": [window.x(), window.y(), window.width(), window.height()],
            "focus": None
            if focus is None
            else {
                "class": type(focus).__name__,
                "name": focus.objectName(),
                "in_window": focus.window() is window,
            },
            "input": {
                "text": line.text(),
                "cursor": line.cursorPosition(),
                "selected": line.selectedText(),
            },
            "output": {
                "lines": output.blockCount(),
                "tail": terminal.text().splitlines()[-40:],
                "scroll": bar.value(),
                "max": bar.maximum(),
                "selection": output.textCursor().hasSelection(),
            },
            "chart": {
                "log": window.chart.log_scale,
                "probe": window.chart.probe_epoch,
                "painted": window.chart.last_painted_epoch,
                "readout": window.chart.last_probe_readout,
            },
            "network": {
                "selected": window.network.selected_neuron,
                "painted": window.network.last_painted_epoch,
                "readout": window.network.last_selection_readout,
            },
            "splitters": {name: s.sizes() for name, s in window.splitters.items()},
            "panels": {name: _rect(w) for name, w in panels.items()},
            "handles": handles,
            "session": {
                "state": window.session.state.value,
                "epoch": window.session.epoch,
                "total": window.session.total_epochs,
                "history": len(history),
                "last_train": history[-1].train_loss if history else None,
            },
            "cursors": sorted({w.cursor().shape().name for w in widgets}),
            "native_cursor": None if native is None else native.cursor().shape().name,
            "popups": popups,
            "paints": dict(counter.counts) if counter is not None else {},
            "usage": {
                "samples": state["samples"],
                "cpu": None if sample is None else sample.cpu_percent,
                "process_cpu": None if sample is None else sample.process_cpu_percent,
                "memory_used": None if sample is None else sample.memory_used,
                "memory_total": None if sample is None else sample.memory_total,
                "process_memory": None if sample is None else sample.process_memory,
                "gpu": None if sample is None or sample.gpu is None else sample.gpu.name,
            },
            "clipboard": QGuiApplication.clipboard().text(),
            "title": window.windowTitle(),
            "platform": QGuiApplication.platformName(),
            "modules": sorted(m for m in sys.modules if "web" in m.lower() or m == "torch"),
        }

    class _Autopilot:
        """Scripted input inside the window process (offscreen tests of the isolated window).

        Script: ``{"activate": bool, "steps": [step, ...]}``. A step fires once, when its
        condition first holds: ``{"on": "shown"}``, ``{"on": "epoch", "value": 3}``,
        ``{"on": "state", "value": "paused"}``, ``{"on": "finished"}`` (``SessionFinished``),
        ``{"on": "time", "value": seconds after the window was shown}`` or ``{"on": "step",
        "value": index of an earlier step}`` (once that step has run); optional
        ``"delay_ms"``. Actions:
        ``{"do": "keys", "keys": "...", "target": "native"|"window"|"focus"|<panel>}``,
        ``{"do": "close"}``, ``{"do": "activate"}``, ``{"do": "mouse", "action": "click"|
        "dclick"|"press"|"release"|"move", "at": <panel>, "button": "left"|"right"|"middle",
        "offset": [dx, dy]}``.
        """

        def __init__(self, window, script) -> None:
            self.window = window
            self.steps = list(script.get("steps", []))
            self.done: set[int] = set()  # fired (their delay may still be running)
            self.ran: set[int] = set()  # carried out
            self.started = None
            self.activate_on_show = bool(script.get("activate", True))
            self.clock = QTimer(window)
            self.clock.setInterval(50)
            self.clock.timeout.connect(_guard(self.tick))
            self.clock.start()

        def tick(self) -> None:
            window = self.window
            if self.started is None and window.isVisible():
                self.started = time.monotonic()
                if self.activate_on_show:  # what a window manager does for a new window
                    window.activateWindow()
                    window.raise_()
            if self.started is None:
                return
            elapsed = time.monotonic() - self.started
            for index, step in enumerate(self.steps):
                if index in self.done:
                    continue
                on = step.get("on")
                if (
                    on == "shown"
                    or (on == "time" and elapsed >= float(step["value"]))
                    or (on == "step" and int(step["value"]) in self.ran)
                ):
                    self.fire(index, step)

        def after_event(self, event) -> None:
            from hollywood_training.sync.events import EpochCompleted, SessionFinished

            session = self.window.session
            for index, step in enumerate(self.steps):
                if index in self.done:
                    continue
                on = step.get("on")
                if (
                    (
                        on == "epoch"
                        and isinstance(event, EpochCompleted)
                        and event.record.epoch == int(step["value"])
                    )
                    or (on == "state" and session.state.value == step["value"])
                    or (on == "finished" and isinstance(event, SessionFinished))
                ):
                    self.fire(index, step)

        def fire(self, index, step) -> None:
            self.done.add(index)
            delay = int(step.get("delay_ms", 0))
            QTimer.singleShot(delay, self.window, _guard(lambda: self.run(index, step)))

        def run(self, index, step) -> None:
            from PySide6.QtCore import QPoint, Qt
            from PySide6.QtTest import QTest

            window = self.window
            action = step.get("do")
            self.ran.add(index)
            log("step", index=index, action=action)
            if action == "close":
                window.close()
            elif action == "activate":
                window.activateWindow()
            elif action == "keys":
                from htspec import keys as harness_keys  # the harness's key notation

                target = step.get("target", "native")
                if target == "native":
                    where = window.windowHandle()
                elif target == "window":
                    where = window
                elif target == "focus":
                    where = QApplication.focusWidget() or window
                else:
                    where = _panel(window, target)
                harness_keys.send(where, step["keys"])
            elif action == "mouse":
                buttons = {
                    "left": Qt.MouseButton.LeftButton,
                    "right": Qt.MouseButton.RightButton,
                    "middle": Qt.MouseButton.MiddleButton,
                }
                widget = _panel(window, step.get("at", "network"))
                center = widget.mapTo(window, widget.rect().center())
                dx, dy = step.get("offset", (0, 0))
                point = QPoint(center.x() + int(dx), center.y() + int(dy))
                native = window.windowHandle()
                button = buttons[step.get("button", "left")]
                kind = step.get("action", "click")
                none = Qt.KeyboardModifier.NoModifier
                if kind == "click":
                    QTest.mouseClick(native, button, none, point)
                elif kind == "dclick":
                    QTest.mouseDClick(native, button, none, point)
                elif kind == "press":
                    QTest.mousePress(native, button, none, point)
                elif kind == "release":
                    QTest.mouseRelease(native, button, none, point)
                elif kind == "move":
                    QTest.mouseMove(native, point)

    def _panel(window, name):
        return {
            "header": window.header,
            "network": window.network,
            "terminal": window.terminal,
            "output": window.terminal.output,
            "input": window.terminal.input,
            "status": window.status,
            "chart": window.chart,
            "usage": window.usage,
            "handle": window.splitters["columns"].handle(1),
        }[name]

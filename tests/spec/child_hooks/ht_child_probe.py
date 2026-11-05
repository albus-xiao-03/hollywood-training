"""Probe installed inside the GUI window process by ``sitecustomize`` (tests only).

It observes the real window process from the inside and writes JSON lines to the file
named by ``HT_TEST_CHILD_LOG``:

=================  ==========================================================================
kind               fields
=================  ==========================================================================
``installed``      ``pid``
``received``       link message from the training process (``message``, ``sequence``,
                   ``synchronous``, ``event``, ``epoch``)
``sent``           link message to the training process (``message``, ``sequence``, ``line``)
``applied``        an event was applied by ``MainWindow.apply`` (and so drawn): ``event``,
                   ``epoch``, ``network_painted``, ``chart_painted``, ``session_epoch``,
                   ``state``, ``title``, ``weights`` (small layers only)
``terminal``       a line written to the terminal (``level``, ``text``)
``usage``          a CPU / memory / GPU reading reached the usage panel (first few only)
``action``         a scripted action ran; ``error`` when something failed
``closing``        the window closes: ``terminal`` text, ``torch_loaded``, ``web_modules``
=================  ==========================================================================

``HT_TEST_CHILD_SCRIPT`` holds rules ``{"on": trigger, "do": [action, ...]}``; each fires
once. Triggers: ``{"event": name}`` with optional ``"epoch"``, ``"state"`` and ``"nth"``
(the n-th such event), or ``{"text": substring}`` (terminal output). Actions:
``{"type": line}`` (typed into the terminal, then Enter), ``{"close": true}``,
``{"screenshot": path}``; each may carry ``"after_ms"`` (default 30) after the previous.
``HT_TEST_CHILD_DEADLINE`` (seconds) closes the window if a test never finishes.
"""

from __future__ import annotations

import json
import os
import sys
import threading
import time
from typing import Any

_LOCK = threading.Lock()
_PATH = os.environ.get("HT_TEST_CHILD_LOG", "")
_MAX_WEIGHTS = 256  # log weight matrices up to this many elements
_MAX_USAGE = 5


def log(kind: str, **fields: Any) -> None:
    fields["kind"] = kind
    fields["t"] = time.monotonic()
    line = json.dumps(fields, default=str)
    with _LOCK, open(_PATH, "a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def _epoch_of(event: Any) -> int | None:
    record = getattr(event, "record", None)
    if record is not None:
        return record.epoch
    epoch = getattr(event, "epoch", None)
    if epoch is not None:
        return epoch
    return getattr(event, "epochs_completed", None)


def install() -> None:
    from PySide6.QtCore import Qt, QTimer
    from PySide6.QtTest import QTest

    from hollywood_training.gui import link as link_module
    from hollywood_training.gui import window as window_module
    from hollywood_training.gui.widgets import terminal as terminal_module
    from hollywood_training.gui.widgets import usage_panel as usage_module

    rules: list[dict[str, Any]] = json.loads(os.environ.get("HT_TEST_CHILD_SCRIPT", "[]"))
    deadline = float(os.environ.get("HT_TEST_CHILD_DEADLINE", "90"))
    fired: set[int] = set()
    counts: dict[str, int] = {}
    windows: list[Any] = []
    usage_seen = [0]

    # ------------------------------------------------------------- actions
    def perform(window: Any, action: dict[str, Any]) -> None:
        try:
            if "type" in action:
                field_ = window.terminal.input
                field_.clear()
                QTest.keyClicks(field_, action["type"])
                QTest.keyClick(field_, Qt.Key.Key_Return)
            elif action.get("close"):
                window.close()
            elif "screenshot" in action:
                window.grab().save(action["screenshot"], "PNG")
            log("action", action=action)
        except Exception as exc:
            log("error", text=f"action {action}: {type(exc).__name__}: {exc}")

    def schedule(actions: list[dict[str, Any]]) -> None:
        if not windows:
            return
        window = windows[-1]
        delay = 0
        for action in actions:
            delay += int(action.get("after_ms", 30))
            QTimer.singleShot(delay, lambda a=action: perform(window, a))

    def fire_event_rules(event: Any) -> None:
        name = type(event).__name__
        counts[name] = counts.get(name, 0) + 1
        state = getattr(getattr(event, "state", None), "value", None)
        if state is not None:
            counts[f"{name}:{state}"] = counts.get(f"{name}:{state}", 0) + 1
        for index, rule in enumerate(rules):
            trigger = rule.get("on", {})
            if index in fired or trigger.get("event") != name:
                continue
            if "epoch" in trigger and _epoch_of(event) != trigger["epoch"]:
                continue
            if "state" in trigger and state != trigger["state"]:
                continue
            if "nth" in trigger:
                key = f"{name}:{trigger['state']}" if "state" in trigger else name
                if counts.get(key, 0) != trigger["nth"]:
                    continue
            fired.add(index)
            schedule(rule.get("do", []))

    def fire_text_rules(text: str) -> None:
        for index, rule in enumerate(rules):
            trigger = rule.get("on", {})
            if index in fired or "text" not in trigger or trigger["text"] not in text:
                continue
            fired.add(index)
            schedule(rule.get("do", []))

    # ---------------------------------------------------------- link traffic
    original_receive = link_module.Link.receive
    original_send = link_module.Link.send

    def receive(self: Any) -> tuple:
        message = original_receive(self)
        entry: dict[str, Any] = {"message": message[0]}
        if message[0] == "event":
            envelope = message[1]
            entry.update(
                sequence=envelope.sequence,
                synchronous=envelope.synchronous,
                event=type(envelope.event).__name__,
                epoch=_epoch_of(envelope.event),
            )
        log("received", **entry)
        return message

    def send(self: Any, *message: Any) -> bool:
        entry: dict[str, Any] = {"message": message[0] if message else None}
        if message and message[0] == "ack":
            entry["sequence"] = message[1]
        elif message and message[0] == "command":
            entry["line"] = message[1]
        log("sent", **entry)
        return original_send(self, *message)

    link_module.Link.receive = receive
    link_module.Link.send = send

    # ------------------------------------------------------- window drawing
    original_init = window_module.MainWindow.__init__
    original_apply = window_module.MainWindow.apply

    def init(self: Any, *args: Any, **kwargs: Any) -> None:
        original_init(self, *args, **kwargs)
        windows.append(self)

        def closing() -> None:
            log(
                "closing",
                terminal=self.terminal.text(),
                title=self.windowTitle(),
                torch_loaded="torch" in sys.modules,
                web_modules=sorted(
                    name
                    for name in sys.modules
                    if "WebEngine" in name or "WebView" in name or name == "webbrowser"
                ),
            )

        self.closing.connect(closing)
        QTimer.singleShot(int(deadline * 1000), self.close)

    def apply(self: Any, event: Any) -> None:
        original_apply(self, event)
        entry: dict[str, Any] = {
            "event": type(event).__name__,
            "epoch": _epoch_of(event),
            "network_painted": self.network.last_painted_epoch,
            "chart_painted": self.chart.last_painted_epoch,
            "session_epoch": self.session.epoch,
            "state": self.session.state.value,
            "title": self.windowTitle(),
        }
        weights = self.session.weights
        if weights is not None:
            entry["weights_epoch"] = weights.epoch
            entry["weights"] = {
                layer.name: layer.matrix.tolist()
                for layer in weights.layers
                if layer.matrix.size <= _MAX_WEIGHTS
            }
        log("applied", **entry)
        try:
            fire_event_rules(event)
        except Exception as exc:
            log("error", text=f"rules: {type(exc).__name__}: {exc}")

    window_module.MainWindow.__init__ = init
    window_module.MainWindow.apply = apply

    # ------------------------------------------------------ terminal, usage
    original_write = terminal_module.TerminalPanel.write

    def write(self: Any, text: str, level: str = "info") -> None:
        original_write(self, text, level)
        if text:
            log("terminal", level=level, text=text)
            try:
                fire_text_rules(text)
            except Exception as exc:
                log("error", text=f"text rules: {type(exc).__name__}: {exc}")

    terminal_module.TerminalPanel.write = write

    original_set_sample = usage_module.UsagePanel.set_sample

    def set_sample(self: Any, sample: Any) -> None:
        original_set_sample(self, sample)
        if usage_seen[0] < _MAX_USAGE:
            usage_seen[0] += 1
            log(
                "usage",
                cpu_percent=sample.cpu_percent,
                cpu_count=sample.cpu_count,
                memory_used=sample.memory_used,
                memory_total=sample.memory_total,
                process_memory=sample.process_memory,
                gpu=sample.gpu is not None,
            )

    usage_module.UsagePanel.set_sample = set_sample
    log("installed", pid=os.getpid())

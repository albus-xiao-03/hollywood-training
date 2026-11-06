"""Scenario: another application's window, next to the Hollywood Training window.

``python other_app.py X Y WIDTH HEIGHT`` opens a plain Qt window (a line edit inside) at
that position and reports, one JSON object per line on stdout, the input it receives:

* ``{"event": "ready", "pid": ..., "win_id": ..., "geometry": [x, y, w, h]}``
* ``{"event": "mouse", "type": "press"|"release"|"dblclick"|"move", "button": n, "x", "y"}``
* ``{"event": "wheel", "dy": ...}`` · ``{"event": "enter"}`` · ``{"event": "leave"}``
* ``{"event": "key", "text": "...", "key": n}`` · ``{"event": "focus_in"}`` ·
  ``{"event": "focus_out"}`` · ``{"event": "context"}`` (a context-menu request; no menu
  is opened, so it cannot take the input of later steps)

Its cursor is ``Qt.CrossCursor``: the harness checks that the pointer looks and works as
usual over other applications (SPEC R2 requirement 6). It quits after 300 s at the latest.
"""

from __future__ import annotations

import json
import os
import sys

from PySide6.QtCore import QEvent, QObject, Qt, QTimer
from PySide6.QtWidgets import QApplication, QLineEdit, QVBoxLayout, QWidget


def say(**fields) -> None:
    print(json.dumps(fields), flush=True)


class Reporter(QObject):
    def eventFilter(self, watched, event) -> bool:
        kind = event.type()
        types = QEvent.Type
        name = type(watched).__name__
        mouse = {
            types.MouseButtonPress: "press",
            types.MouseButtonRelease: "release",
            types.MouseButtonDblClick: "dblclick",
            types.MouseMove: "move",
        }
        if kind in mouse:
            position = event.position()
            say(
                event="mouse", type=mouse[kind], button=int(event.button().value),
                x=round(position.x()), y=round(position.y()), widget=name,
            )  # fmt: skip
        elif kind == types.Wheel:
            say(event="wheel", dy=event.angleDelta().y(), widget=name)
        elif kind == types.Enter:
            say(event="enter", widget=name)
        elif kind == types.Leave:
            say(event="leave", widget=name)
        elif kind == types.KeyPress:
            say(event="key", text=event.text(), key=int(event.key()), widget=name)
        elif kind == types.FocusIn:
            say(event="focus_in", widget=name)
        elif kind == types.FocusOut:
            say(event="focus_out", widget=name)
        elif kind == types.ContextMenu:
            # reported, but no menu: an open popup would take the wheel, the keys and a
            # pointer grab away from the rest of the test
            say(event="context", widget=name)
            return True
        return False


def main() -> int:
    x, y, width, height = (int(v) for v in sys.argv[1:5])
    app = QApplication(["htspec-other-app"])
    window = QWidget()
    window.setWindowTitle("Spec Harness Other App")
    window.setCursor(Qt.CursorShape.CrossCursor)
    edit = QLineEdit(window)
    edit.setCursor(Qt.CursorShape.CrossCursor)
    layout = QVBoxLayout(window)
    layout.addWidget(edit)
    reporter = Reporter()
    for widget in (window, edit):
        widget.installEventFilter(reporter)
        widget.setMouseTracking(True)
    window.setGeometry(x, y, width, height)
    window.show()
    edit.setFocus()

    def ready() -> None:
        geometry = [window.x(), window.y(), window.width(), window.height()]
        say(event="ready", pid=os.getpid(), win_id=int(window.winId()), geometry=geometry)

    QTimer.singleShot(200, ready)
    QTimer.singleShot(300_000, app.quit)
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())

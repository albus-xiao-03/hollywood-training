"""Observable window state and pointer gestures for the R2 tests (offscreen, in-process).

:func:`snapshot` collects everything a mouse could change in the window - panel sizes,
the loss axis, the probe line, the highlighted neuron, the log's selection and scroll
position, the command line's text, cursor and selection, the focus, the window geometry
and state, open popups / tooltips, the clipboard - so "nothing changed" is one comparison.

The gestures send pointer input to the window's native ``QWindow`` with ``QTest`` - the
path real mouse input takes (window system -> QGuiApplication -> QWindow -> widgets).
:func:`popup_watch` guards against a context menu that would block the test if a right
click ever got through: it closes any popup after a moment and records that it saw one.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any

from PySide6.QtCore import QPoint, QPointF, Qt, QTimer
from PySide6.QtGui import QGuiApplication
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QToolTip, QWidget

LEFT = Qt.MouseButton.LeftButton
RIGHT = Qt.MouseButton.RightButton
MIDDLE = Qt.MouseButton.MiddleButton
NO_MOD = Qt.KeyboardModifier.NoModifier


def all_widgets(window: QWidget) -> list[QWidget]:
    return [window, *window.findChildren(QWidget)]


def panels(window) -> dict[str, QWidget]:
    terminal = window.terminal
    return {
        "header": window.header,
        "network": window.network,
        "terminal": terminal,
        "output": terminal.output,
        "scrollbar": terminal.output.verticalScrollBar(),
        "input": terminal.input,
        "status": window.status,
        "chart": window.chart,
        "usage": window.usage,
    }


def handles(window) -> dict[str, QWidget]:
    found = {}
    for name, splitter in window.splitters.items():
        for index in range(1, splitter.count()):
            found[f"{name}:{index}"] = splitter.handle(index)
    return found


def snapshot(window) -> dict[str, Any]:
    """Everything the mouse could have changed (and nothing that animates by itself)."""
    terminal = window.terminal
    line = terminal.input
    output = terminal.output
    focus = QApplication.focusWidget()
    popup = QApplication.activePopupWidget()
    return {
        "splitters": {name: s.sizes() for name, s in window.splitters.items()},
        "log_scale": window.chart.log_scale,
        "probe": window.chart.probe_epoch,
        "neuron": window.network.selected_neuron,
        "output_selection": output.textCursor().selectedText(),
        "output_scroll": output.verticalScrollBar().value(),
        "output_text": terminal.text(),
        "input_text": line.text(),
        "input_cursor": line.cursorPosition(),
        "input_selection": line.selectedText(),
        "focus": None if focus is None else (type(focus).__name__, focus.objectName()),
        "geometry": window.geometry().getRect(),
        "window_state": window.windowState(),
        "visible": window.isVisible(),
        "popup": None if popup is None else type(popup).__name__,
        "tooltip": QToolTip.isVisible(),
        "top_levels": sorted(
            type(w).__name__ for w in QApplication.topLevelWidgets() if w.isVisible()
        ),
        "clipboard": QGuiApplication.clipboard().text(),
    }


def changes(before: dict[str, Any], after: dict[str, Any]) -> dict[str, tuple[Any, Any]]:
    return {key: (before[key], after[key]) for key in before if before[key] != after[key]}


@contextmanager
def popup_watch(seen: list[str], delay_ms: int = 600):
    """Close (and record) any popup that opens while the block runs - a context menu that
    got through would otherwise block the test in its own event loop."""

    def sweep() -> None:
        popup = QApplication.activePopupWidget()
        while popup is not None:
            seen.append(type(popup).__name__)
            popup.close()
            popup = QApplication.activePopupWidget()

    timer = QTimer()
    timer.setInterval(delay_ms)
    timer.timeout.connect(sweep)
    timer.start()
    try:
        yield
    finally:
        sweep()
        timer.stop()


# ------------------------------------------------------------------ geometry
def window_point(window, widget: QWidget, local: QPoint | None = None) -> QPoint:
    """A point of ``widget`` (its centre by default) in the window's coordinates."""
    local = widget.rect().center() if local is None else local
    return widget.mapTo(window, local)


def native(window):
    return window.windowHandle()


# ------------------------------------------------------------------ gestures
def settle(ms: int = 30) -> None:
    QApplication.processEvents()
    QTest.qWait(ms)


def click(window, point: QPoint, button=LEFT) -> None:
    QTest.mouseClick(native(window), button, NO_MOD, point)
    settle()


def double_click(window, point: QPoint, button=LEFT) -> None:
    QTest.mouseDClick(native(window), button, NO_MOD, point)
    settle()


def press(window, point: QPoint, button=LEFT) -> None:
    QTest.mousePress(native(window), button, NO_MOD, point)


def release(window, point: QPoint, button=LEFT) -> None:
    QTest.mouseRelease(native(window), button, NO_MOD, point)


def move(window, point: QPoint) -> None:
    QTest.mouseMove(native(window), point)


def drag(window, start: QPoint, end: QPoint, button=LEFT, steps: int = 10) -> None:
    QTest.mouseMove(native(window), start)
    press(window, start, button)
    for k in range(1, steps + 1):
        QTest.mouseMove(
            native(window),
            QPoint(
                round(start.x() + (end.x() - start.x()) * k / steps),
                round(start.y() + (end.y() - start.y()) * k / steps),
            ),
        )
    release(window, end, button)
    settle()


def hover(window, points: list[QPoint], dwell_ms: int = 0) -> None:
    for point in points:
        QTest.mouseMove(native(window), point)
        if dwell_ms:
            QTest.qWait(dwell_ms)
    settle()


def wheel(window, point: QPoint, notches: int) -> None:
    for _ in range(abs(notches)):
        QTest.wheelEvent(native(window), QPointF(point), QPoint(0, 120 if notches > 0 else -120))
    settle()


GESTURES = {
    "left-click": lambda w, p: click(w, p, LEFT),
    "right-click": lambda w, p: click(w, p, RIGHT),
    "middle-click": lambda w, p: click(w, p, MIDDLE),
    "double-click": lambda w, p: double_click(w, p, LEFT),
    "drag": lambda w, p: drag(w, p, QPoint(p.x() + 120, p.y() + 60)),
    "drag-back": lambda w, p: drag(w, p, QPoint(max(1, p.x() - 140), max(1, p.y() - 70))),
    "wheel-up": lambda w, p: wheel(w, p, 3),
    "wheel-down": lambda w, p: wheel(w, p, -3),
    "hover": lambda w, p: hover(w, [p, QPoint(p.x() + 5, p.y() + 3), p], dwell_ms=150),
}


# ------------------------------------------------------------------ fixtures of state
def fill_log(window, count: int = 300) -> None:
    """Write ``count`` lines, so the log can scroll and has text everywhere to select."""
    for index in range(count):
        window.terminal.write(f"filler line {index:03d} for scrolling and selection", "dim")
    QApplication.processEvents()


def scroll_to_middle(window) -> int:
    """Scroll the log half-way (neither end), so any scrolling would show; returns the value."""
    bar = window.terminal.output.verticalScrollBar()
    bar.setValue(bar.maximum() // 2)
    QApplication.processEvents()
    return bar.value()

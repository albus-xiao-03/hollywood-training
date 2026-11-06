"""R2 requirement 6 - "The mouse pointer and mouse interactions **OUTSIDE** the `Hollywood
Training` window **MUST NOT** be affected."

* Offscreen, in-process: no application-wide override cursor, no mouse or keyboard grab,
  and another top-level window of the *same* ``QApplication`` (the host application of
  ``ht.run(process, isolate_gui=False)``) keeps its cursors and receives clicks, moves,
  wheel turns, hover / tooltips, drops and context menus - while the Hollywood window is
  open, during a real in-process run and after the Hollywood window closed.
* X11 (``x11``): another *application* next to the real Hollywood window (separate
  process): the pointer moves freely across the window's boundary, looks normal outside it
  (on the bare screen and over the other application, whose own cursor is shown), the other
  application receives clicks, moves, wheel and - when it has the focus - the keyboard, and
  nobody holds a pointer or keyboard grab; all of that stays true after the session ends.
"""

from __future__ import annotations

import time

import pytest
from htspec import paths
from htspec.gui_state import GESTURES, LEFT, NO_MOD, panels, popup_watch, window_point
from htspec.live import LiveApp
from htspec.models import make_setup
from htspec.subproc import spawn_python
from PySide6.QtCore import QEvent, QMimeData, QPoint, QPointF, Qt
from PySide6.QtGui import (
    QContextMenuEvent,
    QDragEnterEvent,
    QDragMoveEvent,
    QDropEvent,
    QHelpEvent,
)
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QLineEdit, QPushButton, QToolTip, QVBoxLayout, QWidget

pytestmark = pytest.mark.req("R2.6")


class OtherWindow(QWidget):
    """Another top-level window of the same application, recording its pointer input."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("another window of the host application")
        self.events: list[str] = []
        self.clicks = 0
        self.button = QPushButton("press me", self)
        self.button.clicked.connect(self._clicked)
        self.button.setToolTip("a tooltip of the host application")
        self.button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.edit = QLineEdit(self)
        layout = QVBoxLayout(self)
        layout.addWidget(self.button)
        layout.addWidget(self.edit)
        self.setCursor(Qt.CursorShape.CrossCursor)
        self.setMouseTracking(True)
        self.setGeometry(40, 40, 360, 220)

    def _clicked(self) -> None:
        self.clicks += 1

    def mousePressEvent(self, event) -> None:
        self.events.append(f"press:{event.button().name}")
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:
        self.events.append("move")
        super().mouseMoveEvent(event)

    def wheelEvent(self, event) -> None:
        self.events.append("wheel")
        super().wheelEvent(event)

    def enterEvent(self, event) -> None:
        self.events.append("enter")
        super().enterEvent(event)

    def contextMenuEvent(self, event) -> None:
        self.events.append("context")


@pytest.fixture
def other(qtbot):
    window = OtherWindow()
    qtbot.addWidget(window)
    window.show()
    QTest.qWaitForWindowExposed(window, 5000)
    return window


def exercise(other: OtherWindow) -> dict:
    """Mouse input to the other window through its native window; what it received."""
    handle = other.windowHandle()
    blank = QPoint(other.width() - 4, other.height() - 4)  # the window's own margin
    button_center = other.button.mapTo(other, other.button.rect().center())
    before_clicks = other.clicks
    other.events.clear()
    QTest.mouseMove(handle, blank)
    QTest.mouseMove(handle, QPoint(blank.x() - 5, blank.y() - 5))
    QTest.mouseClick(handle, LEFT, NO_MOD, button_center)
    QTest.mouseClick(handle, Qt.MouseButton.RightButton, NO_MOD, blank)
    QTest.wheelEvent(handle, QPointF(blank), QPoint(0, 120))
    QApplication.processEvents()
    return {
        "clicks": other.clicks - before_clicks,
        "moves": other.events.count("move") > 0,
        "right press": "press:RightButton" in other.events,
        "wheel": "wheel" in other.events,
    }


EXPECTED = {"clicks": 1, "moves": True, "right press": True, "wheel": True}
CURSORS = {
    "window": Qt.CursorShape.CrossCursor,
    "button": Qt.CursorShape.PointingHandCursor,
    "edit": Qt.CursorShape.IBeamCursor,
}


def cursors_of(other: OtherWindow) -> dict:
    return {
        "window": other.cursor().shape(),
        "button": other.button.cursor().shape(),
        "edit": other.edit.cursor().shape(),
    }


# ------------------------------------------------------------------ offscreen
def test_no_application_wide_override_cursor(rig):
    assert QApplication.overrideCursor() is None
    rig.feeder.epochs(2)
    for line in ("scale log", "probe 1", "neuron fc1 0", "layout chart 40", "window max"):
        rig.type_line(line)
    window = rig.window
    for name in ("chart", "network", "input"):
        GESTURES["left-click"](window, window_point(window, panels(window)[name]))
    assert QApplication.overrideCursor() is None


def test_no_mouse_or_keyboard_grab(rig):
    assert QWidget.mouseGrabber() is None and QWidget.keyboardGrabber() is None
    window = rig.window
    handle = window.windowHandle()
    for name in ("chart", "output", "input"):
        point = window_point(window, panels(window)[name])
        QTest.mousePress(handle, LEFT, NO_MOD, point)
        assert QWidget.mouseGrabber() is None, name
        QTest.mouseRelease(handle, LEFT, NO_MOD, point)
    rig.type("abc")
    assert QWidget.keyboardGrabber() is None


def test_another_window_of_the_application_keeps_its_cursors(other, make_rig):
    before = cursors_of(other)
    assert before == CURSORS
    rig = make_rig()
    rig.feeder.start()
    rig.feeder.epochs(1)
    QApplication.processEvents()
    assert cursors_of(other) == CURSORS
    later = OtherWindow()  # created after the Hollywood window
    later.show()
    QApplication.processEvents()
    try:
        assert cursors_of(later) == CURSORS
    finally:
        later.close()


def test_another_window_of_the_application_receives_mouse_input(other, rig):
    rig.feeder.epochs(1)
    assert exercise(other) == EXPECTED


def test_another_window_of_the_application_gets_hover_and_tooltips(other, rig):
    rig.feeder.epochs(1)
    handle = other.windowHandle()
    other.events.clear()
    QTest.mouseMove(handle, QPoint(5, 5))
    QTest.mouseMove(handle, other.button.mapTo(other, other.button.rect().center()))
    QApplication.processEvents()
    assert "move" in other.events
    position = other.button.rect().center()
    help_event = QHelpEvent(QEvent.Type.ToolTip, position, other.button.mapToGlobal(position))
    QApplication.sendEvent(other.button, help_event)
    QApplication.processEvents()
    try:
        assert QToolTip.isVisible() and QToolTip.text() == "a tooltip of the host application"
    finally:
        QToolTip.hideText()


def test_another_window_of_the_application_accepts_drops_and_context_menus(other, rig):
    rig.feeder.epochs(1)
    edit = other.edit
    mime = QMimeData()
    mime.setText("dropped text")
    copy = Qt.DropAction.CopyAction
    position = edit.rect().center()
    QApplication.sendEvent(edit, QDragEnterEvent(position, copy, mime, LEFT, NO_MOD))
    QApplication.sendEvent(edit, QDragMoveEvent(position, copy, mime, LEFT, NO_MOD))
    QApplication.sendEvent(edit, QDropEvent(QPointF(position), copy, mime, LEFT, NO_MOD))
    assert "dropped text" in edit.text()
    seen: list[str] = []
    menu_event = QContextMenuEvent(
        QContextMenuEvent.Reason.Mouse, position, edit.mapToGlobal(position)
    )
    with popup_watch(seen, delay_ms=300):
        QApplication.sendEvent(edit, menu_event)
    assert seen, "the host application's text field opened no context menu"


def test_another_window_works_during_a_real_in_process_run(pilot, other):
    """The host-application scenario: ``ht.run(process, isolate_gui=False)`` next to
    another window of the same application."""
    observed: dict = {}

    def check(_window) -> None:
        observed["input"] = exercise(other)
        observed["cursors"] = cursors_of(other)
        observed["override"] = QApplication.overrideCursor()

    pilot.on_epoch(2, check)
    pilot.on_finish(pilot.close)
    result = pilot.run(make_setup(epochs=4).process)
    assert result.epochs_completed == 4
    assert observed == {"input": EXPECTED, "cursors": CURSORS, "override": None}


def test_another_window_works_after_the_hollywood_window_closed(other, rig):
    rig.window.close()
    QApplication.processEvents()
    assert cursors_of(other) == CURSORS
    assert exercise(other) == EXPECTED


# ---------------------------------------------------------------------- X11 level
OTHER_GEOMETRY = (20, 20, 380, 220)  # left of the Hollywood window on the 2560 x 1440 screen


@pytest.fixture(scope="module")
def outside(xserver, xclient, tmp_path_factory):
    """The real Hollywood window plus another application next to it."""
    app = LiveApp(xserver, xclient, tmp_path_factory.mktemp("r2_6_live"))
    neighbour = None
    try:  # also when the setup fails half-way: a failed setup gets no teardown
        neighbour = spawn_python(
            [str(paths.SCENARIO_DIR / "other_app.py"), *map(str, OTHER_GEOMETRY)],
            display=xserver.display,
        )
        ready = neighbour.wait_message(lambda m: m.get("event") == "ready", timeout=60)
        yield app, neighbour, ready
    finally:
        if neighbour is not None:
            neighbour.terminate()
        app.close()


def _other_point(ready, dx=0, dy=0) -> tuple[int, int]:
    x, y, w, h = ready["geometry"]
    return x + w // 2 + dx, y + h // 2 + dy


def _new_events(neighbour, start: int) -> list[dict]:
    time.sleep(0.3)
    return neighbour.messages[start:]


@pytest.mark.x11
@pytest.mark.subprocess
def test_the_pointer_moves_freely_across_the_window_boundary(outside, xclient):
    app, _neighbour, ready = outside
    x, y, w, h = app.window_rect()
    points = [
        (x - 30, y + 40), (x + 30, y + 40), (x + w // 2, y + h // 2), (x + w + 30, y + h // 2),
        (x + w - 30, y + h - 30), _other_point(ready), (5, 5),
    ]  # fmt: skip
    for point in points:
        xclient.move(*point)
        assert xclient.pointer() == point


@pytest.mark.x11
@pytest.mark.subprocess
def test_the_pointer_looks_normal_outside_the_window(outside, xclient):
    app, _neighbour, ready = outside
    x, y, w, h = app.window_rect()
    for _ in range(2):  # also right after having been over the Hollywood window
        assert xclient.cursor_at(x + w // 2, y + h // 2).blank
        bare = xclient.cursor_at(x + w + 60, y + h // 2)
        assert not bare.blank
        over_other = xclient.cursor_at(*_other_point(ready))
        assert not over_other.blank
        assert over_other.serial != bare.serial  # the other application's own cursor


@pytest.mark.x11
@pytest.mark.subprocess
def test_another_application_receives_clicks_moves_and_wheel(outside, xclient):
    app, neighbour, ready = outside
    app.focus()  # the Hollywood window is the active one meanwhile
    start = len(neighbour.messages)
    point = _other_point(ready)
    xclient.move(point[0] - 20, point[1] - 10)
    xclient.glide(point[0] - 20, point[1] - 10, *point, steps=5)
    xclient.click(*point, button=1)
    xclient.click(*point, button=3)
    xclient.wheel(*point, up=True, notches=2)
    events = _new_events(neighbour, start)
    kinds = {(e["event"], e.get("type"), e.get("button")) for e in events}
    assert ("mouse", "move", 0) in kinds
    assert ("mouse", "press", 1) in kinds and ("mouse", "release", 1) in kinds
    assert ("mouse", "press", 2) in kinds  # Qt.RightButton == 2
    assert any(e["event"] == "context" for e in events)  # the right click asked for a menu
    assert any(e["event"] == "wheel" for e in events)


@pytest.mark.x11
@pytest.mark.subprocess
def test_another_application_gets_the_keyboard_when_it_has_the_focus(outside, xclient):
    app, neighbour, ready = outside
    hollywood_text = app.state()["input"]["text"]
    xclient.focus(ready["win_id"])
    time.sleep(0.3)
    start = len(neighbour.messages)
    xclient.type_text("outside")
    events = _new_events(neighbour, start)
    typed = "".join(e["text"] for e in events if e["event"] == "key")
    assert typed == "outside"
    assert app.state()["input"]["text"] == hollywood_text  # nothing reached the terminal
    app.focus()  # give the keyboard back to the Hollywood window


@pytest.mark.x11
@pytest.mark.subprocess
def test_nobody_holds_a_pointer_or_keyboard_grab(outside, xclient):
    app, _neighbour, _ready = outside
    app.focus()
    assert xclient.try_grab_pointer() == 0
    assert xclient.try_grab_keyboard() == 0
    app.enter("pause", expect="paus")
    assert xclient.try_grab_pointer() == 0
    assert xclient.try_grab_keyboard() == 0
    app.enter("resume", expect="resum")
    x, y = app.center("network")
    xclient.click(x, y)  # a click inside the window leaves no grab behind
    assert xclient.try_grab_pointer() == 0


@pytest.mark.x11
@pytest.mark.subprocess
def test_outside_stays_normal_after_the_session_ended(outside, xclient):
    """Last test of the module: it ends the module's session (``quit``)."""
    app, neighbour, ready = outside
    x, y, w, h = app.window_rect()
    result = app.quit()
    assert result["state"] in ("stopped", "finished")
    time.sleep(0.3)
    assert not xclient.cursor_at(x + w // 2, y + h // 2).blank  # the window is gone
    assert not xclient.cursor_at(*_other_point(ready)).blank
    start = len(neighbour.messages)
    xclient.click(*_other_point(ready))
    events = _new_events(neighbour, start)
    assert any(e["event"] == "mouse" and e["type"] == "press" for e in events)
    assert xclient.try_grab_pointer() == 0

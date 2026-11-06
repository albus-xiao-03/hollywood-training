"""R2 requirement 1 - "The mouse pointer **MUST NOT** be displayed **ANYWHERE** inside the
`Hollywood Training` window."

Two levels:

* **Qt** (offscreen; in-process and in the real isolated window process): the cursor of the
  window, of its native ``QWindow`` and of every widget inside it is blank - also for widgets
  that normally show a pointer shape (text fields: I-beam, splitter handles: resize arrows),
  for cursors set later, for widgets added later, before a session starts and through a
  whole session.
* **X11** (``x11``: a real ``ht.run`` on the harness's private Xvfb): the cursor image the X
  server actually displays (XFixes) while the pointer is over any point of the window is
  fully transparent - on every panel and splitter handle, at the very edges, while a button
  is held, after pausing and after window-mode changes, and when the window appears under a
  resting pointer. Just outside the window the normal cursor is shown (the boundary; R2.6
  covers the outside in depth).
"""

from __future__ import annotations

import time

import pytest
from htspec.gui_state import all_widgets, handles, panels
from htspec.live import PANELS, LiveApp
from htspec.models import make_setup
from htspec.probe_log import LINGER_CLOSE
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication, QLabel, QLineEdit, QPlainTextEdit, QPushButton, QWidget

from hollywood_training.sync.events import TrainingState

pytestmark = pytest.mark.req("R2.1")

BLANK = Qt.CursorShape.BlankCursor
SHAPES = [
    Qt.CursorShape.ArrowCursor,
    Qt.CursorShape.IBeamCursor,
    Qt.CursorShape.PointingHandCursor,
    Qt.CursorShape.SplitHCursor,
    Qt.CursorShape.SplitVCursor,
    Qt.CursorShape.CrossCursor,
    Qt.CursorShape.WaitCursor,
    Qt.CursorShape.OpenHandCursor,
    Qt.CursorShape.SizeAllCursor,
]


def visible_cursors(window) -> list[str]:
    """Widgets of the window whose cursor is not blank (with their shape)."""
    found = []
    for widget in all_widgets(window):
        shape = widget.cursor().shape()
        if shape != BLANK:
            found.append(f"{type(widget).__name__}({widget.objectName()!r}): {shape.name}")
    handle = window.windowHandle()
    if handle is not None and handle.cursor().shape() != BLANK:
        found.append(f"native QWindow: {handle.cursor().shape().name}")
    return found


# ---------------------------------------------------------------------- Qt level
def test_window_native_window_and_every_widget_have_a_blank_cursor(rig):
    assert len(all_widgets(rig.window)) > 20  # the whole widget tree is checked
    assert visible_cursors(rig.window) == []


def test_no_pointer_before_a_session_starts(make_rig):
    """The window is up before the first event arrives ("awaiting network signal")."""
    fresh = make_rig()
    assert fresh.session.info is None
    assert visible_cursors(fresh.window) == []


def test_widgets_that_normally_show_a_pointer_shape_are_blank(rig):
    """Text fields (I-beam), the log's viewport and scroll bar, the splitter handles (resize
    arrows) and every panel."""
    window = rig.window
    candidates = {
        **panels(window),
        "viewport": window.terminal.output.viewport(),
        **{f"handle {name}": handle for name, handle in handles(window).items()},
        **{f"splitter {name}": splitter for name, splitter in window.splitters.items()},
    }
    shapes = {name: widget.cursor().shape() for name, widget in candidates.items()}
    assert {name: shape.name for name, shape in shapes.items() if shape != BLANK} == {}


@pytest.mark.parametrize("shape", SHAPES, ids=lambda s: s.name)
def test_a_cursor_set_later_on_any_widget_is_blanked_again(rig, shape):
    """Qt widgets change their own cursor at runtime (e.g. a text field that becomes
    editable); whatever sets a cursor inside the window, it must not become visible."""
    for widget in all_widgets(rig.window):
        widget.setCursor(shape)
    QApplication.processEvents()
    assert visible_cursors(rig.window) == []


def test_unsetting_cursors_keeps_them_blank(rig):
    for widget in all_widgets(rig.window):
        widget.unsetCursor()
    QApplication.processEvents()
    assert visible_cursors(rig.window) == []


def test_widgets_added_later_inside_the_window_get_a_blank_cursor(rig):
    window = rig.window
    edit = QLineEdit(window.status)
    container = QWidget(window.chart)
    button = QPushButton("later", container)
    text = QPlainTextEdit(container)
    label = QLabel("later", window.network)
    for widget in (edit, container, label):
        widget.show()
    QApplication.processEvents()
    added = [edit, container, button, text, text.viewport(), label]
    assert [type(w).__name__ for w in added if w.cursor().shape() != BLANK] == []
    edit.setCursor(Qt.CursorShape.IBeamCursor)
    QApplication.processEvents()
    assert edit.cursor().shape() == BLANK


def test_cursor_stays_blank_through_a_whole_session(rig):
    feeder = rig.feeder
    feeder.epochs(2)
    feeder.progress(1)
    feeder.paused()
    feeder.state(TrainingState.PAUSED, "paused")
    feeder.state(TrainingState.RUNNING, "resumed")
    feeder.epochs(2)
    feeder.log("a log line")
    feeder.state(TrainingState.FINISHED, "done")
    feeder.finish()
    QApplication.processEvents()
    assert visible_cursors(rig.window) == []


@pytest.mark.parametrize(
    "line",
    ["help", "scale log", "probe 1", "neuron fc1 0", "layout chart 60", "layout reset", "copy",
     "window max", "window full", "window normal", "clear", "status"],
)  # fmt: skip
def test_cursor_stays_blank_after_commands_that_change_the_window(rig, line):
    rig.feeder.epochs(2)
    result = rig.type_line(line)
    assert result is not None and result.ok, result
    QApplication.processEvents()
    assert visible_cursors(rig.window) == []


def test_cursor_stays_blank_in_every_window_mode(rig):
    window = rig.window
    for show in (window.showMaximized, window.showFullScreen, window.showNormal):
        show()
        QApplication.processEvents()
        assert visible_cursors(window) == [], show.__name__


def test_no_pointer_in_a_real_in_process_run(pilot):
    """``ht.run(process, isolate_gui=False)``: checked while training runs and at the end."""
    seen: list[list[str]] = []
    pilot.on_epoch(2, lambda window: seen.append(visible_cursors(window)))
    pilot.on_finish(pilot.chain(lambda window: seen.append(visible_cursors(window)), pilot.close))
    result = pilot.run(make_setup(epochs=3).process)
    assert result.epochs_completed == 3
    assert seen == [[], []]


@pytest.mark.subprocess
def test_no_pointer_in_the_real_isolated_window_process(probe):
    """The default ``ht.run(process)``: every snapshot of the window process shows only
    blank cursors, on its widgets and on its native window."""
    import hollywood_training as ht

    log = probe(script={"steps": [LINGER_CLOSE]})
    result = ht.run(make_setup(epochs=4).process)
    assert result.epochs_completed == 4
    states = [state for state in log.states() if state["visible"]]
    assert states, "the probe recorded no state of the shown window"
    assert {tuple(s["cursors"]) for s in states} == {("BlankCursor",)}
    assert {s["native_cursor"] for s in states} == {"BlankCursor"}
    assert log.errors() == []


# --------------------------------------------------------------------- X11 level
@pytest.fixture
def live(xserver, xclient, tmp_path):
    app = LiveApp(xserver, xclient, tmp_path)
    yield app
    app.close()


@pytest.mark.x11
@pytest.mark.subprocess
def test_window_appearing_under_a_resting_pointer_hides_it(xserver, xclient, tmp_path):
    """The pointer rests where the window will appear; it must vanish without moving."""
    center = (xserver.size[0] // 2, xserver.size[1] // 2)
    xclient.move(*center)
    time.sleep(0.1)
    assert not xclient.cursor().blank  # the bare screen shows the normal cursor
    app = LiveApp(xserver, xclient, tmp_path, focus=False)
    try:
        time.sleep(0.5)
        assert xclient.pointer() == center
        assert xclient.cursor().blank
    finally:
        app.close()


@pytest.mark.x11
@pytest.mark.subprocess
def test_real_cursor_is_invisible_at_every_point_of_the_window(live, xclient):
    """A 16 x 10 grid over the whole window, 2 px inside its edges."""
    visible = []
    for x, y in live.grid(16, 10, inset=2):
        cursor = xclient.cursor_at(x, y, settle=0.02)
        if not cursor.blank:
            visible.append((x, y, cursor.opaque))
    assert visible == []


@pytest.mark.x11
@pytest.mark.subprocess
def test_real_cursor_is_invisible_over_every_panel_and_splitter_handle(live, xclient):
    state = live.state()
    visible: dict[str, list] = {}
    for panel in PANELS:
        x, y, w, h = state["panels"][panel]
        for px, py in ((x + w // 2, y + h // 2), (x + 3, y + 3), (x + w - 4, y + h - 4)):
            if not xclient.cursor_at(px, py, settle=0.03).blank:
                visible.setdefault(panel, []).append((px, py))
    for name in state["handles"]:
        px, py = live.handle_center(name, state)
        if not xclient.cursor_at(px, py, settle=0.03).blank:
            visible.setdefault(f"handle {name}", []).append((px, py))
    assert visible == {}


@pytest.mark.x11
@pytest.mark.subprocess
def test_real_cursor_appears_just_outside_and_vanishes_just_inside_each_edge(live, xclient):
    x, y, w, h = live.window_rect()
    pairs = [  # (just inside, just outside)
        ((x + 1, y + h // 2), (x - 2, y + h // 2)),
        ((x + w - 2, y + h // 2), (x + w + 1, y + h // 2)),
        ((x + w // 2, y + 1), (x + w // 2, y - 2)),
        ((x + w // 2, y + h - 2), (x + w // 2, y + h + 1)),
    ]
    for inside, outside in pairs * 2:
        assert xclient.cursor_at(*inside, settle=0.05).blank, inside
        assert not xclient.cursor_at(*outside, settle=0.05).blank, outside


@pytest.mark.x11
@pytest.mark.subprocess
def test_real_cursor_is_invisible_while_a_button_is_held_and_dragged(live, xclient):
    state = live.state()
    xclient.move(*live.center("network", state))
    xclient.button(1, True)
    try:
        hidden = []
        for panel in ("status", "chart", "usage", "terminal", "network"):
            target = live.center(panel, state)
            xclient.glide(*xclient.pointer(), *target, steps=6, pause=0.005)
            hidden.append((panel, xclient.cursor().blank))
    finally:
        xclient.button(1, False)
    assert all(blank for _panel, blank in hidden), hidden


@pytest.mark.x11
@pytest.mark.subprocess
def test_real_cursor_stays_invisible_when_paused_and_after_window_mode_changes(live, xclient):
    live.enter("pause", expect="paus")
    points = [live.center(panel) for panel in ("network", "chart", "input")]
    assert all(xclient.cursor_at(*point).blank for point in points)
    for mode in ("max", "full", "normal"):
        live.enter(f"window {mode}", expect="window:")
        time.sleep(0.3)
        x, y, w, h = live.window_rect()
        assert xclient.cursor_at(x + w // 2, y + h // 2).blank, mode
        assert xclient.cursor_at(x + 5, y + 5).blank, mode
    live.enter("resume", expect="resum")
    assert all(xclient.cursor_at(*point).blank for point in points)


@pytest.mark.x11
@pytest.mark.subprocess
def test_real_cursor_is_invisible_after_typing_and_terminal_output(live, xclient):
    """Terminal output and typed text (widgets repaint, the log grows) do not bring the
    pointer back."""
    live.enter("help", expect="type 'help <command>' for details")
    xclient.move(*live.center("output"))
    time.sleep(0.1)
    assert xclient.cursor().blank
    live.type("abc")
    time.sleep(0.2)
    assert xclient.cursor().blank
    for _ in "abc":
        xclient.tap("BackSpace")
    assert xclient.cursor().blank

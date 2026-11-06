"""R2 requirement 2 - "**ALL** mouse-based interactions **MUST** be disabled within the
application."

What the mouse could do in a Qt window like this one - and did in the first version: click
the loss chart (axis scale), hover it (read-out), hover a neuron (tooltip), drag the
splitters (panel sizes), select and copy log text, scroll the log (wheel, scroll bar), click
into / select text in the command line, middle-click paste, drop text, open context menus,
take the keyboard focus by clicking - plus touch input. Every test does such interactions
and asserts that **nothing changed** (``htspec.gui_state.snapshot``) and no popup or
tooltip opened.

* Offscreen, in-process: gestures through the platform input path (the native ``QWindow``)
  on every panel and every splitter handle; pointer events sent straight to every widget;
  drag and drop; touch; the keyboard focus; a real in-process run.
* The real isolated window process: mouse input inside it (probe autopilot).
* X11 (``x11``): real button presses, drags, wheel turns and middle clicks from the X server.
"""

from __future__ import annotations

import time

import pytest
from htspec.gui_state import (
    GESTURES,
    LEFT,
    NO_MOD,
    all_widgets,
    changes,
    click,
    fill_log,
    handles,
    native,
    panels,
    popup_watch,
    scroll_to_middle,
    snapshot,
    window_point,
)
from htspec.live import PANELS, LiveApp
from htspec.models import make_setup
from htspec.probe_log import LINGER_CLOSE
from PySide6.QtCore import QEvent, QMimeData, QPoint, QPointF, Qt
from PySide6.QtGui import (
    QContextMenuEvent,
    QDragEnterEvent,
    QDragMoveEvent,
    QDropEvent,
    QMouseEvent,
    QWheelEvent,
)
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

pytestmark = pytest.mark.req("R2.2")

PANEL_TARGETS = (
    "header", "network", "terminal", "output", "scrollbar", "input", "status", "chart", "usage",
)  # fmt: skip
HANDLE_NAMES = ("columns:1", "left:1", "right:1", "right:2")
HANDLE_GESTURES = ("drag", "drag-back", "left-click", "double-click")
# what a mouse could change; nothing in this list changes by itself while the window idles
STABLE = (
    "splitters", "log_scale", "probe", "neuron", "output_selection", "input_text",
    "input_cursor", "input_selection", "focus", "geometry", "window_state", "popup", "tooltip",
)  # fmt: skip


def target_point(window, name: str) -> QPoint:
    """Where to point at a panel: over text for the text widgets, else the centre."""
    widget = panels(window)[name]
    if name == "input":
        local = QPoint(12, widget.height() // 2)
    elif name == "output":
        local = QPoint(60, 30)
    else:
        local = widget.rect().center()
    return window_point(window, widget, local)


@pytest.fixture
def busy(rig):
    """A window with history, a long log scrolled half-way and text in the command line -
    so a click, a drag, a double click or a wheel turn would visibly change something."""
    rig.feeder.epochs(3)
    fill_log(rig.window)
    scroll_to_middle(rig.window)
    rig.window.terminal.input.setText("status of the run")  # the cursor goes to the end
    QApplication.processEvents()
    return rig


# ------------------------------------------------------- offscreen, platform path
@pytest.mark.parametrize("gesture", sorted(GESTURES))
@pytest.mark.parametrize("panel", PANEL_TARGETS)
def test_no_pointer_gesture_on_any_panel_changes_anything(busy, panel, gesture):
    window = busy.window
    point = target_point(window, panel)
    before = snapshot(window)
    seen: list[str] = []
    with popup_watch(seen):
        GESTURES[gesture](window, point)
    assert seen == [], f"a popup opened: {seen}"
    assert changes(before, snapshot(window)) == {}


@pytest.mark.parametrize("gesture", HANDLE_GESTURES)
@pytest.mark.parametrize("handle", HANDLE_NAMES)
def test_no_pointer_gesture_on_a_splitter_handle_resizes_a_panel(busy, handle, gesture):
    """Dragging the splitters resized the panels in the first version."""
    window = busy.window
    point = window_point(window, handles(window)[handle])
    before = snapshot(window)
    GESTURES[gesture](window, point)
    assert changes(before, snapshot(window)) == {}


def test_clicking_the_loss_chart_does_not_switch_its_axis(busy):
    """The first version switched linear/log on a click of the chart."""
    window = busy.window
    for _ in range(3):
        click(window, target_point(window, "chart"))
    assert window.chart.log_scale is False


def test_hovering_neurons_and_the_chart_shows_nothing(busy):
    """The first version showed a neuron's details and the chart's values on hover."""
    window = busy.window
    view = window.network
    points = []
    for column in view.columns():
        for y in column.ys:
            points.append(window_point(window, view, QPoint(round(column.x), round(float(y)))))
    assert points, "no neurons are drawn"
    before = snapshot(window)
    for point in points:
        QTest.mouseMove(native(window), point)
        QTest.qWait(15)
    chart = window.chart
    for fraction in (0.2, 0.4, 0.6, 0.8):
        local = QPoint(round(chart.width() * fraction), chart.height() // 2)
        QTest.mouseMove(native(window), window_point(window, chart, local))
        QTest.qWait(80)
    QTest.qWait(700)  # longer than Qt's tooltip delay
    view.repaint()
    chart.repaint()
    assert changes(before, snapshot(window)) == {}
    assert view.last_selection_readout is None
    assert chart.last_probe_readout is None


# ------------------------------------------------------ offscreen, straight to widgets
def _pointer_events(widget):
    center = QPointF(widget.rect().center())
    glob = QPointF(widget.mapToGlobal(widget.rect().center()))
    moved = QPointF(center.x() + 80, center.y() + 40)
    moved_glob = QPointF(glob.x() + 80, glob.y() + 40)
    none = Qt.MouseButton.NoButton
    yield QMouseEvent(QEvent.Type.MouseButtonPress, center, glob, LEFT, LEFT, NO_MOD)
    yield QMouseEvent(QEvent.Type.MouseMove, moved, moved_glob, none, LEFT, NO_MOD)
    yield QMouseEvent(QEvent.Type.MouseButtonRelease, moved, moved_glob, LEFT, none, NO_MOD)
    yield QMouseEvent(QEvent.Type.MouseButtonDblClick, center, glob, LEFT, LEFT, NO_MOD)
    yield QMouseEvent(QEvent.Type.MouseButtonRelease, center, glob, LEFT, none, NO_MOD)
    yield QWheelEvent(
        center, glob, QPoint(0, 0), QPoint(0, 360), none, NO_MOD, Qt.ScrollPhase.NoScrollPhase,
        False,
    )  # fmt: skip
    yield QContextMenuEvent(QContextMenuEvent.Reason.Mouse, center.toPoint(), glob.toPoint())


def test_pointer_events_sent_straight_to_any_widget_have_no_effect(busy):
    """Pointer events that bypass the native window (sent to a widget directly) are
    disabled as well - for every widget of the window."""
    window = busy.window
    before = snapshot(window)
    seen: list[str] = []
    with popup_watch(seen):
        for widget in all_widgets(window):
            for event in _pointer_events(widget):
                QApplication.sendEvent(widget, event)
        QApplication.processEvents()
    assert seen == []
    assert changes(before, snapshot(window)) == {}


def test_dropping_text_onto_the_window_inserts_nothing(busy):
    window = busy.window
    mime = QMimeData()
    mime.setText(" DROPPED")
    copy = Qt.DropAction.CopyAction
    before = snapshot(window)
    targets = (window.terminal.input, window.terminal.output.viewport(), window, window.chart)
    for widget in targets:
        pos = widget.rect().center()
        QApplication.sendEvent(widget, QDragEnterEvent(pos, copy, mime, LEFT, NO_MOD))
        QApplication.sendEvent(widget, QDragMoveEvent(pos, copy, mime, LEFT, NO_MOD))
        QApplication.sendEvent(widget, QDropEvent(QPointF(pos), copy, mime, LEFT, NO_MOD))
    QApplication.processEvents()
    assert changes(before, snapshot(window)) == {}


def test_touch_input_has_no_effect(busy):
    window = busy.window
    device = QTest.createTouchDevice()
    before = snapshot(window)
    for name in ("input", "output", "chart", "network"):
        point = target_point(window, name)
        moved = QPoint(point.x() + 60, point.y())
        sequence = QTest.touchEvent(native(window), device)
        sequence.press(0, point)
        sequence.commit()
        sequence.move(0, moved)
        sequence.commit()
        sequence.release(0, moved)
        sequence.commit()
        QTest.qWait(20)
    assert changes(before, snapshot(window)) == {}


def test_clicks_never_take_the_keyboard_focus_from_the_terminal(busy):
    window = busy.window
    line = window.terminal.input
    for name in PANEL_TARGETS:
        click(window, target_point(window, name))
        assert QApplication.focusWidget() is line, name
    busy.type("xyz")
    assert line.text().endswith("xyz")


# --------------------------------------------------------------- real sessions
def test_mouse_input_during_a_real_in_process_run_changes_nothing(pilot):
    observed: dict = {}

    def stable(window) -> dict:
        state = snapshot(window)
        return {key: state[key] for key in STABLE}

    def use_the_mouse(window) -> None:
        before = stable(window)
        seen: list[str] = []
        with popup_watch(seen):
            for name in ("network", "chart", "output", "input", "status"):
                point = target_point(window, name)
                for gesture in ("left-click", "right-click", "double-click", "drag", "wheel-up"):
                    GESTURES[gesture](window, point)
        observed["changes"] = changes(before, stable(window))
        observed["popups"] = seen

    pilot.on_epoch(2, use_the_mouse)
    pilot.on_finish(pilot.close)
    result = pilot.run(make_setup(epochs=5).process)
    assert result.epochs_completed == 5  # training went on unaffected
    assert observed == {"changes": {}, "popups": []}


def _stable_probe(state: dict) -> dict:
    return {
        "log": state["chart"]["log"],
        "probe": state["chart"]["probe"],
        "neuron": state["network"]["selected"],
        "splitters": state["splitters"],
        "input": state["input"],
        "output_selection": state["output"]["selection"],
        "focus": state["focus"],
        "popups": state["popups"],
        "window_state": state["window_state"],
    }


@pytest.mark.subprocess
def test_mouse_input_in_the_isolated_window_process_changes_nothing(probe):
    import hollywood_training as ht

    mouse = [
        {"action": "click", "at": "chart"},
        {"action": "click", "at": "chart", "button": "right"},
        {"action": "dclick", "at": "input"},
        {"action": "dclick", "at": "output"},
        {"action": "click", "at": "output", "button": "middle"},
        {"action": "press", "at": "handle"},
        {"action": "move", "at": "handle", "offset": [150, 0]},
        {"action": "release", "at": "handle", "offset": [150, 0]},
        {"action": "move", "at": "network"},
        {"action": "move", "at": "chart"},
    ]
    steps: list[dict] = [
        {"on": "epoch", "value": 2, "do": "keys", "keys": "pause<Return>"},
        {"on": "state", "value": "paused", "do": "activate", "delay_ms": 300},
    ]
    for number, gesture in enumerate(mouse):
        delay = 600 if number == 0 else 150  # room for "before" snapshots after the marker
        steps.append({"on": "step", "value": len(steps) - 1, "do": "mouse", "delay_ms": delay,
                      **gesture})  # fmt: skip
    last_mouse = len(steps) - 1
    steps.append({"on": "step", "value": last_mouse, "do": "keys", "keys": "resume<Return>",
                  "delay_ms": 600})  # fmt: skip
    steps.append(LINGER_CLOSE)
    log = probe(script={"steps": steps})
    # ~30 ms per epoch: the run is still going when the typed "pause" arrives
    result = ht.run(make_setup(epochs=30, train_hook=lambda _b: time.sleep(0.005)).process)
    assert result.epochs_completed == 30
    ran = {record["index"]: record["t"] for record in log.records("step")}
    assert {1, last_mouse, last_mouse + 1} <= set(ran), ran
    states = log.states()
    before = [s for s in states if ran[1] + 0.2 < s["t"] < ran[2]]
    after = [s for s in states if ran[last_mouse] + 0.2 < s["t"] < ran[last_mouse + 1]]
    assert before and after, "the probe took no snapshot around the mouse input"
    assert _stable_probe(after[-1]) == _stable_probe(before[-1])
    assert log.errors() == []


# ---------------------------------------------------------------------- X11 level
@pytest.fixture(scope="module")
def live(xserver, xclient, tmp_path_factory):
    app = LiveApp(xserver, xclient, tmp_path_factory.mktemp("r2_2_live"))
    try:  # also when the setup fails half-way: a failed setup gets no teardown
        for _ in range(3):  # a log long enough to scroll and to select from
            app.enter("help", expect="type 'help <command>' for details")
        yield app
    finally:
        app.close()


def _stable_live(state: dict) -> dict:
    return {**_stable_probe(state), "scroll": state["output"]["scroll"],
            "geometry": state["geometry"], "readout": state["network"]["readout"]}  # fmt: skip


@pytest.mark.x11
@pytest.mark.subprocess
def test_real_clicks_on_every_panel_change_nothing(live, xclient):
    state = live.state()
    baseline = _stable_live(state)
    windows = len(xclient.viewable_windows(live.window_pid))
    for panel in PANELS:
        x, y = live.center(panel, state)
        for button, count in ((1, 1), (1, 2), (1, 3), (3, 1), (2, 1)):
            xclient.click(x, y, button, count=count)
            time.sleep(0.05)
    time.sleep(0.3)
    assert _stable_live(live.state()) == baseline
    assert len(xclient.viewable_windows(live.window_pid)) == windows  # no popup / tooltip


@pytest.mark.x11
@pytest.mark.subprocess
def test_real_drags_on_every_splitter_handle_change_nothing(live, xclient):
    state = live.state()
    before = state["splitters"]
    for name in state["handles"]:
        x, y = live.handle_center(name, state)
        xclient.drag(x, y, x + 150, y + 80)
        xclient.drag(x, y, x - 150, y - 80)
    time.sleep(0.3)
    assert live.state()["splitters"] == before


@pytest.mark.x11
@pytest.mark.subprocess
def test_real_text_selection_with_the_mouse_selects_nothing(live, xclient):
    state = live.state()
    x, y, w, h = state["panels"]["output"]
    xclient.drag(x + 20, y + 15, x + w - 30, y + h // 2)
    xclient.click(x + 40, y + 30, 1, count=3)
    live.type("select me")
    ix, iy, _width, ih = state["panels"]["input"]
    xclient.drag(ix + 5, iy + ih // 2, ix + 220, iy + ih // 2)
    xclient.click(ix + 20, iy + ih // 2, 1, count=2)
    time.sleep(0.3)
    after = live.state()
    try:
        assert after["output"]["selection"] is False
        assert after["input"]["selected"] == ""
        assert after["input"]["text"] == "select me"
        assert after["input"]["cursor"] == len("select me")
    finally:
        xclient.tap("a", ctrl=True)
        xclient.tap("BackSpace")


@pytest.mark.x11
@pytest.mark.subprocess
def test_real_wheel_turns_scroll_nothing(live, xclient):
    xclient.tap("Prior")  # PgUp: away from the bottom, so any scrolling would show
    xclient.tap("Prior")
    time.sleep(0.2)
    state = live.state()
    before = _stable_live(state)
    for panel in ("output", "chart", "network", "input"):
        x, y = live.center(panel, state)
        xclient.wheel(x, y, up=True, notches=5)
        xclient.wheel(x, y, up=False, notches=8)
    time.sleep(0.3)
    try:
        assert _stable_live(live.state()) == before
    finally:
        xclient.tap("End", ctrl=True)


@pytest.mark.x11
@pytest.mark.subprocess
def test_real_middle_click_does_not_paste(live, xclient):
    """On X11 a middle click pastes the selection into a text field."""
    live.type("paste-me")
    xclient.tap("Home", shift=True)  # select the text: it becomes the X11 primary selection
    xclient.tap("End")
    time.sleep(0.2)
    state = live.state()
    ix, iy, iw, ih = state["panels"]["input"]
    try:
        xclient.click(ix + 10, iy + ih // 2, button=2)
        xclient.click(ix + iw // 2, iy + ih // 2, button=2)
        time.sleep(0.3)
        assert live.state()["input"]["text"] == "paste-me"
    finally:
        xclient.tap("a", ctrl=True)
        xclient.tap("BackSpace")


@pytest.mark.x11
@pytest.mark.subprocess
def test_real_hovering_shows_no_tooltip_and_no_details(live, xclient):
    state = live.state()
    windows = len(xclient.viewable_windows(live.window_pid))
    for panel in ("network", "chart"):
        x, y, w, h = state["panels"][panel]
        xclient.move(x + 10, y + h // 2)
        xclient.glide(x + 10, y + h // 2, x + w - 10, y + h // 3, steps=40, pause=0.02)
        time.sleep(0.8)  # longer than a tooltip delay
    after = live.state()
    assert after["network"]["selected"] is None
    assert after["network"]["readout"] is None
    assert after["chart"]["probe"] is None
    assert after["popups"] == []
    assert len(xclient.viewable_windows(live.window_pid)) == windows


@pytest.mark.x11
@pytest.mark.subprocess
def test_training_and_keyboard_still_work_after_all_that_mouse_input(live):
    before = live.epochs_reported()
    live.focus()
    live.enter("ping", expect="pong")
    live.wait_epochs(2)
    assert live.epochs_reported() >= before + 2

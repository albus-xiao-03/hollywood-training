"""Pointer guard (SPEC R2.1, R2.2, R2.6): no pointer and no pointer input in the window only."""

from __future__ import annotations

import pytest
from PySide6.QtCore import QEvent, QMimeData, QPoint, QPointF, Qt
from PySide6.QtGui import QContextMenuEvent, QDragEnterEvent, QDropEvent, QWheelEvent
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QPushButton, QSplitterHandle, QToolTip, QWidget

from hollywood_training.commands import create_registry
from hollywood_training.gui import MainWindow
from hollywood_training.gui.pointer_guard import POINTER_EVENTS, PointerGuard
from hollywood_training.gui.widgets import LossChart, NetworkView
from hollywood_training.sync import EpochRecord, SessionState

BLANK = Qt.CursorShape.BlankCursor


def activate(qtbot, widget: QWidget) -> None:
    widget.activateWindow()
    qtbot.waitUntil(lambda: QApplication.activeWindow() is widget)


@pytest.fixture
def window(qtbot):
    session = SessionState()
    session.total_epochs = 10
    win = MainWindow(session, completer=create_registry().complete)
    qtbot.addWidget(win)
    win.resize(1400, 900)
    with qtbot.waitExposed(win):
        win.show()
    activate(qtbot, win)
    for epoch, loss in enumerate((0.9, 0.5, 0.3), start=1):  # something to hover and click
        session.history.append(EpochRecord(epoch, loss, loss + 0.1))
    session.history_version += 1
    for k in range(200):
        win.terminal.write(f"line {k}")  # something to scroll and select
    return win


def widgets(window: MainWindow) -> list[QWidget]:
    return [window, *window.findChildren(QWidget)]


def centre(window: MainWindow, widget: QWidget) -> QPoint:
    return widget.mapTo(window, widget.rect().center())


def wheel(target, x: int, y: int) -> QWheelEvent:
    """A wheel turn at ``(x, y)`` of ``target`` (a widget or a native window)."""
    return QWheelEvent(
        QPointF(x, y),
        QPointF(target.mapToGlobal(QPoint(x, y))),
        QPoint(),
        QPoint(0, 360),
        Qt.MouseButton.NoButton,
        Qt.KeyboardModifier.NoModifier,
        Qt.ScrollPhase.NoScrollPhase,
        False,
    )


# --------------------------------------------------------------------- R2.1
def test_no_pointer_is_shown_anywhere_in_the_window(window):
    assert all(widget.cursor().shape() == BLANK for widget in widgets(window))
    native = window.windowHandle()
    assert isinstance(window.pointer, PointerGuard) and window.pointer.native_window is native
    for widget in (window.terminal.input, window.chart, window.network, window.terminal.output):
        QTest.mouseMove(native, centre(window, widget))  # over every kind of widget
        assert native.cursor().shape() == BLANK


def test_widgets_cannot_bring_their_own_cursor_back(window):
    window.chart.setCursor(Qt.CursorShape.CrossCursor)
    window.terminal.input.setCursor(Qt.CursorShape.IBeamCursor)
    assert window.chart.cursor().shape() == BLANK
    assert window.terminal.input.cursor().shape() == BLANK


# --------------------------------------------------------------------- R2.2
def test_mouse_input_from_the_window_system_is_swallowed(window):
    native = window.windowHandle()
    QTest.mouseClick(native, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier,
                     centre(window, window.chart))  # fmt: skip
    assert not window.chart.log_scale  # R1: a click switched the axis
    bar = window.terminal.output.verticalScrollBar()
    bar.setValue(bar.maximum() // 2)
    before = bar.value()
    where = centre(window, window.terminal.output)
    QApplication.sendEvent(native, wheel(native, where.x(), where.y()))
    assert bar.value() == before
    QTest.mouseMove(native, centre(window, window.network))  # R1: a neuron tooltip
    assert not QToolTip.isVisible()


def test_mouse_events_sent_to_widgets_are_swallowed(window, qtbot):
    qtbot.mouseClick(window.chart, Qt.MouseButton.LeftButton)
    assert not window.chart.log_scale
    output = window.terminal.output
    qtbot.mouseDClick(output.viewport(), Qt.MouseButton.LeftButton)
    assert not output.textCursor().hasSelection()
    bar = output.verticalScrollBar()
    before = bar.value()
    QApplication.sendEvent(output.viewport(), wheel(output.viewport(), 5, 5))
    assert bar.value() == before
    for handle in window.findChildren(QSplitterHandle):
        splitter = handle.splitter()
        sizes = splitter.sizes()
        start = handle.rect().center()
        qtbot.mousePress(handle, Qt.MouseButton.LeftButton, pos=start)
        qtbot.mouseMove(handle, start + QPoint(-90, -90))
        qtbot.mouseRelease(handle, Qt.MouseButton.LeftButton, pos=start + QPoint(-90, -90))
        assert splitter.sizes() == sizes


def test_no_context_menu_tooltip_or_drop(window, qtbot):
    line = window.terminal.input
    for reason in (QContextMenuEvent.Reason.Mouse, QContextMenuEvent.Reason.Keyboard):
        event = QContextMenuEvent(reason, QPoint(5, 5), line.mapToGlobal(QPoint(5, 5)))
        QApplication.sendEvent(line, event)
    assert QApplication.activePopupWidget() is None
    QApplication.sendEvent(window.chart, QEvent(QEvent.Type.ToolTip))
    assert not QToolTip.isVisible()
    data = QMimeData()
    data.setText("dropped")
    actions = Qt.DropAction.CopyAction
    enter = QDragEnterEvent(QPointF(5, 5), actions, data, Qt.MouseButton.LeftButton,
                            Qt.KeyboardModifier.NoModifier)  # fmt: skip
    QApplication.sendEvent(line, enter)
    drop = QDropEvent(QPointF(5, 5), actions, data, Qt.MouseButton.LeftButton,
                      Qt.KeyboardModifier.NoModifier)  # fmt: skip
    QApplication.sendEvent(line, drop)
    assert not enter.isAccepted() and line.text() == ""


def test_every_widget_is_switched_to_keyboard_only(window):
    for widget in widgets(window):
        assert not widget.hasMouseTracking(), widget
        assert not widget.testAttribute(Qt.WidgetAttribute.WA_Hover), widget
        assert not widget.acceptDrops(), widget
        assert widget.contextMenuPolicy() == Qt.ContextMenuPolicy.PreventContextMenu, widget


def test_widgets_added_later_are_guarded(window, qtbot):
    clicks: list[bool] = []
    button = QPushButton("press me", window.status)
    button.clicked.connect(clicks.append)
    button.show()
    assert button.cursor().shape() == BLANK and not button.hasMouseTracking()
    qtbot.mouseClick(button, Qt.MouseButton.LeftButton)
    assert clicks == []


MOUSE_HANDLERS = (
    "mousePressEvent",
    "mouseReleaseEvent",
    "mouseDoubleClickEvent",
    "mouseMoveEvent",
    "wheelEvent",
    "enterEvent",
    "leaveEvent",
    "contextMenuEvent",
)


@pytest.mark.parametrize("panel", [LossChart, NetworkView])
def test_panels_have_no_mouse_code(panel):
    for name in MOUSE_HANDLERS:
        assert getattr(panel, name) is getattr(QWidget, name), f"{panel.__name__}.{name}"


def test_the_loss_chart_has_no_tooltip(window):
    assert window.chart.toolTip() == ""


def test_every_pointer_device_is_covered():
    for kind in ("MouseButtonPress", "MouseMove", "Wheel", "TouchBegin", "TabletPress",
                 "NativeGesture", "ContextMenu", "DragEnter", "ToolTip", "HoverMove"):  # fmt: skip
        assert getattr(QEvent.Type, kind) in POINTER_EVENTS
    assert QEvent.Type.KeyPress not in POINTER_EVENTS
    assert QEvent.Type.NonClientAreaMouseButtonPress not in POINTER_EVENTS  # the title bar


def test_the_keyboard_keeps_working(window):
    for key in (Qt.Key.Key_O, Qt.Key.Key_K):
        QTest.keyClick(window.windowHandle(), key)
    assert window.terminal.input.text() == "ok"


def test_the_guard_attaches_when_the_window_is_shown(qtbot):
    win = MainWindow(SessionState())
    qtbot.addWidget(win)
    assert win.pointer.native_window is None  # not created yet
    with qtbot.waitExposed(win):
        win.show()
    assert win.pointer.native_window is win.windowHandle()
    assert win.windowHandle().cursor().shape() == BLANK


# --------------------------------------------------------------------- R2.6
def test_outside_the_window_nothing_changes(window, qtbot):
    events: list[QEvent.Type] = []

    class Host(QWidget):  # e.g. a host application's window (in-process frontend)
        def mousePressEvent(self, event):
            events.append(event.type())

        def wheelEvent(self, event):
            events.append(event.type())

    host = Host()
    qtbot.addWidget(host)
    host.resize(300, 200)
    with qtbot.waitExposed(host):
        host.show()
    activate(qtbot, host)
    native = host.windowHandle()
    QTest.mouseClick(native, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier,
                     QPoint(40, 40))  # fmt: skip
    QApplication.sendEvent(native, wheel(native, 40, 40))
    assert events == [QEvent.Type.MouseButtonPress, QEvent.Type.Wheel]
    assert not host.testAttribute(Qt.WidgetAttribute.WA_SetCursor)
    assert host.cursor().shape() == Qt.CursorShape.ArrowCursor
    assert native.cursor().shape() != BLANK
    assert QApplication.overrideCursor() is None
    assert QWidget.mouseGrabber() is None and QWidget.keyboardGrabber() is None

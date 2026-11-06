"""``PointerGuard``: no mouse pointer and no pointer input inside the window.

SPEC R2 requirements 1, 2 and 6: the mouse pointer is not displayed anywhere inside the
Hollywood Training window, all mouse-based interactions inside it are disabled, and the
pointer and mouse interactions outside the window are not affected. Every function of the
window has a keyboard path instead: the terminal, its keys and the ``view`` commands.

One guard object filters the events of the window:

* it is installed on the window's native ``QWindow``, where all real pointer input of the
  window arrives, and on every widget of the window (widgets added later as well), so
  pointer events also stop when they are sent to a widget directly;
* it swallows mouse buttons, moves and double clicks, the wheel, enter / leave / hover,
  tooltips and what's-this, context menus, drag and drop, tablet (stylus), touch and native
  gestures (``POINTER_EVENTS``);
* it gives the window and every widget a blank cursor (``Qt.BlankCursor``) and blanks a
  widget's cursor again whenever the widget changes it; mouse tracking, hover, drops and
  context menus are switched off on every widget.

Everything is per window: no override cursor, no application-wide event filter, no mouse
or keyboard grab. Over other windows - other applications, and other windows of the same
``QApplication`` (a host application with ``run(process, isolate_gui=False)``) - the
pointer looks and works as usual. The native title bar and frame belong to the window
manager and keep their own pointer and buttons; non-client events are left alone.
"""

from __future__ import annotations

from PySide6.QtCore import QEvent, QObject, Qt
from PySide6.QtGui import QWindow
from PySide6.QtWidgets import QWidget

_T = QEvent.Type
POINTER_EVENTS = frozenset(
    {
        _T.MouseButtonPress,
        _T.MouseButtonRelease,
        _T.MouseButtonDblClick,
        _T.MouseMove,
        _T.Wheel,
        _T.Enter,
        _T.Leave,
        _T.HoverEnter,
        _T.HoverMove,
        _T.HoverLeave,
        _T.ToolTip,
        _T.WhatsThis,
        _T.ContextMenu,
        _T.DragEnter,
        _T.DragMove,
        _T.DragLeave,
        _T.Drop,
        _T.TabletPress,
        _T.TabletMove,
        _T.TabletRelease,
        _T.TouchBegin,
        _T.TouchUpdate,
        _T.TouchEnd,
        _T.TouchCancel,
        _T.NativeGesture,
    }
)


class PointerGuard(QObject):
    """Hides the pointer over ``window`` and swallows all pointer input to it."""

    def __init__(self, window: QWidget) -> None:
        super().__init__(window)
        self._window = window
        self._native: QWindow | None = None
        self._adopt(window)
        self._attach_native_window()

    @property
    def native_window(self) -> QWindow | None:
        """The native window the guard watches (None until the window is created)."""
        return self._native

    def _adopt(self, widget: QWidget) -> None:
        """Guard ``widget`` and its descendants."""
        for each in (widget, *widget.findChildren(QWidget)):
            each.installEventFilter(self)  # installing twice only moves it to the front
            each.setCursor(Qt.CursorShape.BlankCursor)
            each.setMouseTracking(False)
            each.setAttribute(Qt.WidgetAttribute.WA_Hover, False)
            each.setAcceptDrops(False)
            each.setContextMenuPolicy(Qt.ContextMenuPolicy.PreventContextMenu)

    def _attach_native_window(self) -> None:
        """Watch the native window too; it exists once the window has been created."""
        native = self._window.windowHandle()
        if native is None or native is self._native:
            return
        native.installEventFilter(self)
        native.setCursor(Qt.CursorShape.BlankCursor)
        self._native = native

    def eventFilter(self, watched: QObject, event: QEvent) -> bool:
        kind = event.type()
        if kind in POINTER_EVENTS:
            return True
        if kind == QEvent.Type.CursorChange:
            if isinstance(watched, QWidget) and (
                watched.cursor().shape() != Qt.CursorShape.BlankCursor
            ):
                watched.setCursor(Qt.CursorShape.BlankCursor)
        elif kind == QEvent.Type.ChildPolished:
            child = event.child()
            if isinstance(child, QWidget):
                self._adopt(child)
        elif kind in (QEvent.Type.Show, QEvent.Type.WinIdChange) and watched is self._window:
            self._attach_native_window()
        return False

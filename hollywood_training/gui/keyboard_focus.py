"""``KeyboardFocus``: everything typed in the window goes to the terminal's command line.

SPEC R2 requirement 7: while the user stays on the Hollywood Training window - while it
is the active window - all keyboard input is routed to the integrated terminal. The
command line is therefore the window's only focus target:

* every other widget of the window gets ``Qt.NoFocus`` (widgets added later as well), and
  the command line is the window's focus proxy, so activating the window (Alt+Tab, the
  first show, the window manager) puts the focus there;
* when the focus moves to another widget of the window anyway (``QWidget.setFocus()``
  ignores the focus policy), it is handed back at once;
* a key press or release addressed to another widget of the window is forwarded to the
  command line. The command line consumes every key
  (:class:`~hollywood_training.gui.widgets.terminal.CommandLine`), so nothing it receives
  travels up to the panels and back again.

Nothing changes while another window is active: there is no keyboard grab, and the window
never activates itself. Other windows of the same ``QApplication`` (a host application
with ``run(process, isolate_gui=False)``) keep their own focus handling.
"""

from __future__ import annotations

from PySide6.QtCore import QEvent, QObject, Qt
from PySide6.QtWidgets import QApplication, QWidget

FORWARDED = frozenset({QEvent.Type.KeyPress, QEvent.Type.KeyRelease})


class KeyboardFocus(QObject):
    """Keeps the keyboard focus of ``window`` on ``target`` and forwards stray keys to it."""

    def __init__(self, window: QWidget, target: QWidget) -> None:
        super().__init__(window)
        if not window.isAncestorOf(target):
            raise ValueError("the focus target must be a widget inside the window")
        self._window = window
        self._target = target
        target.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        window.setFocusProxy(target)
        self._adopt(window)
        QApplication.instance().focusChanged.connect(self._focus_changed)

    @property
    def target(self) -> QWidget:
        """The widget that receives every key (the terminal's command line)."""
        return self._target

    def _adopt(self, widget: QWidget) -> None:
        """Watch ``widget`` and its descendants and take away their focus."""
        for each in (widget, *widget.findChildren(QWidget)):
            each.installEventFilter(self)  # installing twice only moves it to the front
            if each is not self._target:
                each.setFocusPolicy(Qt.FocusPolicy.NoFocus)

    def _focus_changed(self, _old: QWidget | None, new: QWidget | None) -> None:
        if new is not None and new is not self._target and new.window() is self._window:
            self._target.setFocus(Qt.FocusReason.OtherFocusReason)

    def eventFilter(self, watched: QObject, event: QEvent) -> bool:
        kind = event.type()
        if kind in FORWARDED:
            if watched is self._target:
                return False
            self._target.setFocus(Qt.FocusReason.OtherFocusReason)
            QApplication.sendEvent(self._target, event)
            return True
        if kind == QEvent.Type.ChildPolished:
            child = event.child()
            if isinstance(child, QWidget):
                self._adopt(child)
        elif kind == QEvent.Type.WindowActivate and watched is self._window:
            self._target.setFocus(Qt.FocusReason.ActiveWindowFocusReason)
        return False

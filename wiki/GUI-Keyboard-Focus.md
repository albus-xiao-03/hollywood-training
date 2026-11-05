# Keyboard Focus

`hollywood_training/gui/keyboard_focus.py` · part of [GUI](GUI)

`KeyboardFocus` routes the keyboard to the integrated terminal (SPEC R2 requirement 7):
while the Hollywood Training window is the active window, every key goes to the
terminal's command line, wherever the window would otherwise have delivered it. The
command line is the window's only focus target, so typing works at once after Alt+Tab or
when the window opens, with no click needed.
[Main Window](GUI-Main-Window) creates one instance as `window.keyboard`.

## Responsibilities

* Make the command line the window's only focus target: `Qt.NoFocus` for every other
  widget, including widgets added later, and the command line as the window's focus proxy.
* Hand the focus back to the command line whenever it moves to another widget of the
  window, and when the window is activated.
* Forward key presses and releases addressed to any other widget of the window to the
  command line.
* Leave everything outside the window alone: other applications, and other windows of the
  same `QApplication`.

## API

```python
FORWARDED = frozenset({QEvent.Type.KeyPress, QEvent.Type.KeyRelease})

class KeyboardFocus(QObject):
    def __init__(self, window: QWidget, target: QWidget) -> None

    @property
    def target(self) -> QWidget
```

* `window`: the top-level window to lock, which becomes the parent of the `KeyboardFocus`
  object (it lives as long as the window).
* `target`: the widget that receives every key, here `window.terminal.input`. It must be
  inside `window`, otherwise `ValueError` is raised. It gets `Qt.StrongFocus`.
* `FORWARDED`: the event types forwarded to the target.

## How it works

At construction the lock installs itself as an event filter on the window and on every
widget in it, sets `Qt.NoFocus` on all of them except the target, makes the target the
window's focus proxy, and connects to `QApplication.focusChanged`.

| Situation | What the lock does |
|---|---|
| The window is activated (Alt+Tab, the first show, a click on the frame) | `WindowActivate` on the window: the target gets the focus (`ActiveWindowFocusReason`). The focus proxy and Qt's own focus restore lead there too. |
| The focus moves to another widget of the window (`setFocus()` ignores the focus policy) | `focusChanged`: the focus is handed back to the target at once. |
| A key press or release is addressed to another widget of the window (no focus widget yet, or an event sent to a panel directly) | The target gets the focus, the event is sent to the target, and the original receiver never sees it. |
| A widget is added to the window later | `ChildPolished` (sent before the widget is first shown): the widget and its children are adopted, with filter and `Qt.NoFocus`. |
| Another window is active | Nothing. Key events only reach the active window, and the lock never grabs the keyboard or activates its window. |

**No forwarding loops.** A key event that a widget leaves unhandled travels up to its
parent widgets, and the panels around the command line are watched by the lock. The
[command line](GUI-Terminal) therefore consumes every key press and release, including
keys it has no use for, such as F5 or Escape. Nothing that reaches it travels on, so a
forwarded event can never come back.

Tab and Shift+Tab never move the focus: the command line uses Tab for completion and
swallows Shift+Tab. Input-method text (for example CJK input) needs no forwarding,
because it always goes to the focus widget, which is the command line.

## Used by / uses

* Used by [Main Window](GUI-Main-Window), which creates `KeyboardFocus(window,
  window.terminal.input)` after building its panels.
* Uses Qt only. The [Terminal](GUI-Terminal)'s `CommandLine` consumes every key, which
  the lock relies on.

## Example

```python
from PySide6.QtCore import Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from hollywood_training.gui import MainWindow
from hollywood_training.sync import SessionState

app = QApplication([])
window = MainWindow(SessionState())   # creates window.keyboard
window.show()

QTest.keyClicks(window.network, "status")   # typed "into" a panel ...
print(window.terminal.input.text())         # ... arrives in the command line: "status"
window.terminal.output.setFocus()           # focus moved away ...
print(QApplication.focusWidget() is window.terminal.input)   # ... and handed back: True
```

## Notes

* Only the window and its widgets are watched. The `focusChanged` handler ignores widgets
  of other windows, so a host application (`run(process, isolate_gui=False)`) keeps its
  own focus behaviour.
* The OS still decides which window is active. The window manager's own keys (for
  example Alt+F4, or Alt+Tab) act before any key reaches the window.

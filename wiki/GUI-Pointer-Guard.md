# Pointer Guard

`hollywood_training/gui/pointer_guard.py` · part of [GUI](GUI)

`PointerGuard` makes the window keyboard-only (SPEC R2 requirements 1, 2 and 6). Inside
the Hollywood Training window the mouse pointer is not shown, and no mouse-based
interaction has any effect. Outside the window nothing changes. Everything the mouse did
in the first version has a keyboard path: the [terminal](GUI-Terminal) with its keys, and
the window-only commands ([View Actions](GUI-View-Actions)). [Main Window](GUI-Main-Window)
creates one guard as `window.pointer`.

## Responsibilities

* Hide the pointer over the whole window: a blank cursor on the window, on its native
  window and on every widget, including widgets added later and widgets that set a
  cursor of their own.
* Swallow every pointer event addressed to the window or one of its widgets, whether it
  comes from the window system or is sent to a widget directly.
* Switch every widget to keyboard-only settings: no mouse tracking, no hover, no drops,
  no context menu.
* Touch nothing outside the window.

## API

```python
POINTER_EVENTS = frozenset({
    QEvent.Type.MouseButtonPress, QEvent.Type.MouseButtonRelease,
    QEvent.Type.MouseButtonDblClick, QEvent.Type.MouseMove, QEvent.Type.Wheel,
    QEvent.Type.Enter, QEvent.Type.Leave,
    QEvent.Type.HoverEnter, QEvent.Type.HoverMove, QEvent.Type.HoverLeave,
    QEvent.Type.ToolTip, QEvent.Type.WhatsThis, QEvent.Type.ContextMenu,
    QEvent.Type.DragEnter, QEvent.Type.DragMove, QEvent.Type.DragLeave, QEvent.Type.Drop,
    QEvent.Type.TabletPress, QEvent.Type.TabletMove, QEvent.Type.TabletRelease,
    QEvent.Type.TouchBegin, QEvent.Type.TouchUpdate, QEvent.Type.TouchEnd,
    QEvent.Type.TouchCancel, QEvent.Type.NativeGesture,
})

class PointerGuard(QObject):
    def __init__(self, window: QWidget) -> None

    @property
    def native_window(self) -> QWindow | None
```

* `POINTER_EVENTS`: the event types the guard swallows: mouse, wheel, enter / leave /
  hover, tooltips and what's-this, context menus, drag and drop, stylus (tablet), touch
  and native gestures (trackpad pinch and the like).
* `window`: the top-level window to guard. It becomes the guard's parent, so the guard
  lives as long as the window.
* `native_window`: the native `QWindow` the guard watches, or `None` until the window has
  been created (shown).

## How it works

**Event filter.** The guard is one event filter, installed on:

| Object | Why |
|---|---|
| the window's native `QWindow` | all real pointer input of the window arrives here first, before Qt picks the widget under the pointer. Swallowing it here also keeps Qt from switching the cursor to the widget under the pointer |
| the window and every widget in it | catches pointer events sent to a widget directly (tests, accessibility tools, synthetic events) |
| widgets added later | `ChildPolished` (sent before a new widget is first shown) adopts the widget and its children |

The filter returns `True` (event consumed) for every type in `POINTER_EVENTS`, and lets
everything else through: keys, paint, resize, timers. The native window exists only once
the window is created. The guard attaches to it at construction if it exists, and
otherwise on the window's first `Show` (and again on `WinIdChange`).

**Cursor.** The window, its native window and every widget get `Qt.BlankCursor`. A
widget that sets another cursor later (for example `QLineEdit` when it changes to
read-only) sends `CursorChange`, and the guard blanks it again.

**Widget settings**, applied to the window and every widget:

| Setting | Value | Effect |
|---|---|---|
| `setMouseTracking` | `False` | no mouse-move events without a button |
| `WA_Hover` | off | no hover events, so no `:hover` styles |
| `setAcceptDrops` | `False` | dragged text or files are refused |
| `setContextMenuPolicy` | `PreventContextMenu` | no context menu, also not from the keyboard's Menu key, and not passed on to the parent |

**What stays outside.** The guard never calls `QApplication.setOverrideCursor`, never
installs an application-wide filter, and never grabs the mouse or the keyboard. Other
applications, and other windows of the same `QApplication` (a host application with
`run(process, isolate_gui=False)`), keep their pointer and mouse handling. The native
title bar and frame are drawn and handled by the window manager: they keep their pointer
and their buttons, and non-client events (`NonClientAreaMouse*`) are not filtered. When
the window manager activates the window on a click, nothing inside the window reacts.

## Mouse interactions of the first version and their replacements

| Mouse (first version) | Keyboard |
|---|---|
| click the loss chart: linear / logarithmic axis | `scale [lin\|log]` |
| hover the loss chart: an epoch's values | `probe [epoch\|off]`: the values, and the probe line on the chart |
| hover a neuron: index, label, bias | `neuron [column] [index]` |
| drag the splitters: panel sizes | `layout [panel] [percent]`, `layout reset` |
| wheel or scroll bar on the log | PgUp / PgDn, Ctrl+Home / Ctrl+End |
| select and copy log text | `copy [lines]` |
| click into the log or a panel (focus) | not needed: every key goes to the command line ([Keyboard Focus](GUI-Keyboard-Focus)) |
| title bar buttons | still there (window manager); also `window [normal\|max\|full\|min]` and `quit` |

## Used by / uses

* Used by [Main Window](GUI-Main-Window), which creates `PointerGuard(window)` after
  building its panels.
* Uses Qt only.

## Example

```python
from PySide6.QtCore import Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from hollywood_training.gui import MainWindow
from hollywood_training.sync import SessionState

app = QApplication([])
window = MainWindow(SessionState())          # creates window.pointer
window.show()

QTest.mouseClick(window.chart, Qt.MouseButton.LeftButton)
print(window.chart.log_scale)                # False: the click never arrives
print(window.chart.cursor().shape())         # CursorShape.BlankCursor
```

## Notes

* What the pointer looks like is decided by the platform. `Qt.BlankCursor` is supported
  on X11, Wayland, Windows and macOS; the offscreen platform only records it.
* Keyboard input is not filtered. Routing it to the terminal is the job of
  [Keyboard Focus](GUI-Keyboard-Focus).

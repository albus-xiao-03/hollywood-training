"""Key sequences in a small text notation, typed with ``QTest``.

``"pause<Return>"`` types p, a, u, s, e and presses Return. Special keys go in angle
brackets, with optional modifiers: ``<C-Home>`` (Ctrl), ``<S-Tab>`` (Shift), ``<A-x>``
(Alt), ``<M-x>`` (Meta). ``<<`` types a literal ``<``.

The target decides the input path:

* a ``QWindow`` (e.g. ``window.windowHandle()``): the platform input path - Qt delivers the
  keys to the focus widget of that window, exactly as for keys typed on a real keyboard;
* a ``QWidget``: the keys are sent to that widget directly (``QTest`` semantics), as if it
  had the keyboard focus - used to check that keys never stay with another widget.
"""

from __future__ import annotations

import re

from PySide6.QtCore import Qt
from PySide6.QtGui import QWindow
from PySide6.QtTest import QTest

_K = Qt.Key
SPECIAL = {
    "Return": _K.Key_Return,
    "Enter": _K.Key_Enter,
    "Tab": _K.Key_Tab,
    "Backtab": _K.Key_Backtab,
    "Esc": _K.Key_Escape,
    "Escape": _K.Key_Escape,
    "Backspace": _K.Key_Backspace,
    "Delete": _K.Key_Delete,
    "Insert": _K.Key_Insert,
    "Up": _K.Key_Up,
    "Down": _K.Key_Down,
    "Left": _K.Key_Left,
    "Right": _K.Key_Right,
    "Home": _K.Key_Home,
    "End": _K.Key_End,
    "PgUp": _K.Key_PageUp,
    "PgDn": _K.Key_PageDown,
    "Space": _K.Key_Space,
    **{f"F{n}": getattr(_K, f"Key_F{n}") for n in range(1, 13)},
}
MODIFIERS = {
    "C": Qt.KeyboardModifier.ControlModifier,
    "S": Qt.KeyboardModifier.ShiftModifier,
    "A": Qt.KeyboardModifier.AltModifier,
    "M": Qt.KeyboardModifier.MetaModifier,
}
_TOKEN = re.compile(r"<<|<([^<>]+)>|[^<]", re.S)


def parse(sequence: str) -> list[tuple[object, Qt.KeyboardModifier]]:
    """``[(key, modifiers)]``: ``key`` is a one-character string or a ``Qt.Key``."""
    result: list[tuple[object, Qt.KeyboardModifier]] = []
    for match in _TOKEN.finditer(sequence):
        token = match.group(0)
        if token == "<<":
            result.append(("<", Qt.KeyboardModifier.NoModifier))
            continue
        name = match.group(1)
        if name is None:
            result.append((token, Qt.KeyboardModifier.NoModifier))
            continue
        modifiers = Qt.KeyboardModifier.NoModifier
        *mods, key_name = name.split("-") if name != "-" else ["-"]
        for mod in mods:
            modifiers |= MODIFIERS[mod]
        if key_name in SPECIAL:
            result.append((SPECIAL[key_name], modifiers))
        elif len(key_name) == 1:
            result.append((key_name, modifiers))
        else:
            raise ValueError(f"unknown key <{name}>")
    return result


def send(target, sequence: str) -> None:
    """Type ``sequence`` into ``target`` (a QWindow or a QWidget)."""
    for key, modifiers in parse(sequence):
        if isinstance(key, str) and modifiers == Qt.KeyboardModifier.NoModifier:
            QTest.keyClick(target, key)
        elif isinstance(key, str):
            code = Qt.Key(ord(key.upper())) if key.isalpha() else Qt.Key(ord(key))
            QTest.keyClick(target, code, modifiers)
        else:
            QTest.keyClick(target, key, modifiers)


def escape(text: str) -> str:
    """``text`` as literal keys (a ``<`` in it is typed, not read as a special key)."""
    return text.replace("<", "<<")


def native(window) -> QWindow:
    """The native ``QWindow`` of a top-level widget (the platform input path)."""
    handle = window.windowHandle()
    if handle is None:
        raise RuntimeError("the window has no native window yet (show it first)")
    return handle

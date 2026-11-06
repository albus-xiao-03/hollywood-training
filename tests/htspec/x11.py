"""A private Xvfb X server and an X client for real pointer, cursor and keyboard tests.

Safety: the harness never uses the display it was started from. :class:`XServer` starts
its own ``Xvfb`` (a free display number chosen by the server itself via ``-displayfd``),
and :class:`XClient` connects to that display by name only. Pointer moves, clicks and key
presses therefore happen on a virtual screen nobody looks at.

The client talks to the server through ``libX11`` (core requests), ``libXtst`` (XTest:
synthetic input that the server treats like real device input - it is routed exactly like
a physical mouse and keyboard) and ``libXfixes`` (``XFixesGetCursorImage``: the cursor
image the server shows *right now*, which is what a user sees as the pointer).

There is no window manager on the server. Where one would act - giving the keyboard focus
to the window the user switched to - the tests do it themselves with ``XSetInputFocus``.
"""

from __future__ import annotations

import ctypes
import os
import select
import shutil
import subprocess
import time
from dataclasses import dataclass
from typing import ClassVar

SCREEN = (2560, 1440)

_x11 = _xtst = _xfixes = None


class _CursorImage(ctypes.Structure):
    _fields_ = [
        ("x", ctypes.c_short),
        ("y", ctypes.c_short),
        ("width", ctypes.c_ushort),
        ("height", ctypes.c_ushort),
        ("xhot", ctypes.c_ushort),
        ("yhot", ctypes.c_ushort),
        ("cursor_serial", ctypes.c_ulong),
        ("pixels", ctypes.POINTER(ctypes.c_ulong)),
        ("atom", ctypes.c_ulong),
        ("name", ctypes.c_char_p),
    ]


class _WindowAttributes(ctypes.Structure):
    _fields_ = [
        ("x", ctypes.c_int), ("y", ctypes.c_int), ("width", ctypes.c_int),
        ("height", ctypes.c_int), ("border_width", ctypes.c_int), ("depth", ctypes.c_int),
        ("visual", ctypes.c_void_p), ("root", ctypes.c_ulong), ("class", ctypes.c_int),
        ("bit_gravity", ctypes.c_int), ("win_gravity", ctypes.c_int),
        ("backing_store", ctypes.c_int), ("backing_planes", ctypes.c_ulong),
        ("backing_pixel", ctypes.c_ulong), ("save_under", ctypes.c_int),
        ("colormap", ctypes.c_ulong), ("map_installed", ctypes.c_int),
        ("map_state", ctypes.c_int), ("all_event_masks", ctypes.c_long),
        ("your_event_mask", ctypes.c_long), ("do_not_propagate_mask", ctypes.c_long),
        ("override_redirect", ctypes.c_int), ("screen", ctypes.c_void_p),
    ]  # fmt: skip


def _load() -> None:
    global _x11, _xtst, _xfixes
    if _x11 is not None:
        return
    x11 = ctypes.CDLL("libX11.so.6")
    xtst = ctypes.CDLL("libXtst.so.6")
    xfixes = ctypes.CDLL("libXfixes.so.3")
    vp, ul, ci, cu, cl = ctypes.c_void_p, ctypes.c_ulong, ctypes.c_int, ctypes.c_uint, ctypes.c_long
    x11.XOpenDisplay.restype = vp
    x11.XOpenDisplay.argtypes = [ctypes.c_char_p]
    x11.XCloseDisplay.argtypes = [vp]
    x11.XDefaultRootWindow.restype = ul
    x11.XDefaultRootWindow.argtypes = [vp]
    x11.XSync.argtypes = [vp, ci]
    x11.XQueryPointer.argtypes = [
        vp, ul, ctypes.POINTER(ul), ctypes.POINTER(ul), ctypes.POINTER(ci), ctypes.POINTER(ci),
        ctypes.POINTER(ci), ctypes.POINTER(ci), ctypes.POINTER(cu),
    ]  # fmt: skip
    x11.XQueryTree.argtypes = [
        vp, ul, ctypes.POINTER(ul), ctypes.POINTER(ul), ctypes.POINTER(ctypes.POINTER(ul)),
        ctypes.POINTER(cu),
    ]  # fmt: skip
    x11.XFree.argtypes = [vp]
    x11.XFetchName.argtypes = [vp, ul, ctypes.POINTER(ctypes.c_char_p)]
    x11.XTranslateCoordinates.argtypes = [
        vp,
        ul,
        ul,
        ci,
        ci,
        ctypes.POINTER(ci),
        ctypes.POINTER(ci),
        ctypes.POINTER(ul),
    ]
    x11.XGetWindowAttributes.argtypes = [vp, ul, ctypes.c_void_p]
    x11.XSetInputFocus.argtypes = [vp, ul, ci, ul]
    x11.XGetInputFocus.argtypes = [vp, ctypes.POINTER(ul), ctypes.POINTER(ci)]
    x11.XRaiseWindow.argtypes = [vp, ul]
    x11.XGrabPointer.argtypes = [vp, ul, ci, cu, ci, ci, ul, ul, ul]
    x11.XUngrabPointer.argtypes = [vp, ul]
    x11.XGrabKeyboard.argtypes = [vp, ul, ci, ci, ci, ul]
    x11.XUngrabKeyboard.argtypes = [vp, ul]
    x11.XStringToKeysym.restype = ul
    x11.XStringToKeysym.argtypes = [ctypes.c_char_p]
    x11.XKeysymToKeycode.restype = ctypes.c_ubyte
    x11.XKeysymToKeycode.argtypes = [vp, ul]
    x11.XkbKeycodeToKeysym.restype = ul
    x11.XkbKeycodeToKeysym.argtypes = [vp, ctypes.c_ubyte, ci, ci]
    x11.XInternAtom.restype = ul
    x11.XInternAtom.argtypes = [vp, ctypes.c_char_p, ci]
    x11.XGetWindowProperty.argtypes = [
        vp, ul, ul, cl, cl, ci, ul, ctypes.POINTER(ul), ctypes.POINTER(ci),
        ctypes.POINTER(ul), ctypes.POINTER(ul), ctypes.POINTER(ctypes.c_void_p),
    ]  # fmt: skip
    x11.XSetErrorHandler.restype = vp
    x11.XSetErrorHandler.argtypes = [vp]
    xtst.XTestFakeMotionEvent.argtypes = [vp, ci, ci, ci, ul]
    xtst.XTestFakeButtonEvent.argtypes = [vp, cu, ci, ul]
    xtst.XTestFakeKeyEvent.argtypes = [vp, cu, ci, ul]
    xfixes.XFixesGetCursorImage.restype = ctypes.POINTER(_CursorImage)
    xfixes.XFixesGetCursorImage.argtypes = [vp]
    _x11, _xtst, _xfixes = x11, xtst, xfixes


IS_VIEWABLE = 2
GRAB_SUCCESS = 0
CURRENT_TIME = 0
REVERT_TO_PARENT = 2
GRAB_MODE_ASYNC = 1


def available() -> str | None:
    """None when real X11 tests can run here, else the reason they cannot."""
    if shutil.which("Xvfb") is None:
        return "Xvfb is not installed"
    try:
        _load()
    except OSError as exc:
        return f"X11 client libraries missing: {exc}"
    return None


# ---------------------------------------------------------------------- server
class XServer:
    """A private ``Xvfb`` server; ``display`` is its name, e.g. ``":3"``."""

    def __init__(self, width: int = SCREEN[0], height: int = SCREEN[1]) -> None:
        read_fd, write_fd = os.pipe()
        self.process = subprocess.Popen(
            [
                "Xvfb", "-displayfd", str(write_fd), "-screen", "0", f"{width}x{height}x24",
                "-nolisten", "tcp", "-noreset", "+extension", "XFIXES", "+extension", "XTEST",
            ],
            pass_fds=(write_fd,),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )  # fmt: skip
        os.close(write_fd)
        number = b""
        deadline = time.monotonic() + 20
        while not number.endswith(b"\n"):
            remaining = deadline - time.monotonic()
            ready = select.select([read_fd], [], [], max(0.0, min(remaining, 0.5)))[0]
            if not ready:
                if remaining <= 0 or self.process.poll() is not None:
                    os.close(read_fd)
                    self.stop()
                    raise RuntimeError("Xvfb did not start")
                continue
            chunk = os.read(read_fd, 16)
            if not chunk:
                break
            number += chunk
        os.close(read_fd)
        if not number.strip():
            self.stop()
            raise RuntimeError("Xvfb did not report its display number")
        self.display = f":{int(number.strip())}"
        self.size = (width, height)

    def stop(self) -> None:
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(10)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(5)


# ---------------------------------------------------------------------- client
@dataclass(frozen=True)
class Cursor:
    width: int
    height: int
    opaque: int  # pixels with a non-zero alpha
    x: int  # pointer position reported with the image
    y: int
    serial: int

    @property
    def blank(self) -> bool:
        return self.opaque == 0


@dataclass(frozen=True)
class TopLevel:
    window: int
    name: str
    pid: int | None
    x: int
    y: int
    width: int
    height: int
    viewable: bool


_ERROR_HANDLER = None


class XClient:
    """An X connection to the private server (never to another display)."""

    _CHAR_KEYSYMS: ClassVar[dict[str, str]] = {
        " ": "space", ".": "period", "-": "minus", "_": "underscore", "/": "slash",
        ":": "colon", ",": "comma", "=": "equal", "+": "plus", "%": "percent", "?": "question",
        "\n": "Return", "'": "apostrophe", '"': "quotedbl", "<": "less", ">": "greater",
        "(": "parenleft", ")": "parenright", "#": "numbersign", "!": "exclam", "@": "at",
        ";": "semicolon", "*": "asterisk", "[": "bracketleft", "]": "bracketright", "|": "bar",
        "&": "ampersand", "$": "dollar", "~": "asciitilde", "\\": "backslash", "\t": "Tab",
        "^": "asciicircum", "{": "braceleft", "}": "braceright", "`": "grave",
    }  # fmt: skip

    def __init__(self, display: str) -> None:
        _load()
        global _ERROR_HANDLER
        if _ERROR_HANDLER is None:  # X errors (e.g. a window that just vanished) must not abort
            handler_type = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p)
            _ERROR_HANDLER = handler_type(lambda _display, _event: 0)
            _x11.XSetErrorHandler(ctypes.cast(_ERROR_HANDLER, ctypes.c_void_p))
        self.display_name = display
        self.dpy = _x11.XOpenDisplay(display.encode())
        if not self.dpy:
            raise RuntimeError(f"cannot open X display {display}")
        self.root = _x11.XDefaultRootWindow(self.dpy)
        self._net_wm_pid = _x11.XInternAtom(self.dpy, b"_NET_WM_PID", 0)

    def close(self) -> None:
        if self.dpy:
            _x11.XCloseDisplay(self.dpy)
            self.dpy = None

    def sync(self) -> None:
        _x11.XSync(self.dpy, 0)

    # ----------------------------------------------------------------- pointer
    def move(self, x: int, y: int) -> None:
        _xtst.XTestFakeMotionEvent(self.dpy, -1, int(x), int(y), CURRENT_TIME)
        self.sync()

    def glide(self, x0: int, y0: int, x1: int, y1: int, steps: int = 12, pause: float = 0.01):
        for k in range(1, steps + 1):
            self.move(round(x0 + (x1 - x0) * k / steps), round(y0 + (y1 - y0) * k / steps))
            time.sleep(pause)

    def button(self, button: int, press: bool) -> None:
        _xtst.XTestFakeButtonEvent(self.dpy, button, 1 if press else 0, CURRENT_TIME)
        self.sync()

    def click(self, x: int, y: int, button: int = 1, *, count: int = 1) -> None:
        self.move(x, y)
        for _ in range(count):
            self.button(button, True)
            time.sleep(0.02)
            self.button(button, False)
            time.sleep(0.03)

    def drag(self, x0: int, y0: int, x1: int, y1: int, button: int = 1) -> None:
        self.move(x0, y0)
        time.sleep(0.03)
        self.button(button, True)
        time.sleep(0.03)
        self.glide(x0, y0, x1, y1)
        time.sleep(0.03)
        self.button(button, False)

    def wheel(self, x: int, y: int, *, up: bool, notches: int = 3) -> None:
        self.move(x, y)
        for _ in range(notches):
            self.button(4 if up else 5, True)
            self.button(4 if up else 5, False)
            time.sleep(0.02)

    def pointer(self) -> tuple[int, int]:
        root, child = ctypes.c_ulong(), ctypes.c_ulong()
        rx, ry, wx, wy = (ctypes.c_int() for _ in range(4))
        mask = ctypes.c_uint()
        _x11.XQueryPointer(
            self.dpy, self.root, ctypes.byref(root), ctypes.byref(child), ctypes.byref(rx),
            ctypes.byref(ry), ctypes.byref(wx), ctypes.byref(wy), ctypes.byref(mask),
        )  # fmt: skip
        return rx.value, ry.value

    def cursor(self) -> Cursor:
        """The cursor image the server displays now (XFixes)."""
        pointer = _xfixes.XFixesGetCursorImage(self.dpy)
        if not pointer:
            raise RuntimeError("XFixesGetCursorImage failed")
        image = pointer.contents
        count = image.width * image.height
        opaque = sum(1 for k in range(count) if (image.pixels[k] >> 24) & 0xFF)
        result = Cursor(image.width, image.height, opaque, image.x, image.y, image.cursor_serial)
        _x11.XFree(pointer)
        return result

    def cursor_at(self, x: int, y: int, settle: float = 0.12) -> Cursor:
        self.move(x, y)
        time.sleep(settle)
        return self.cursor()

    # ---------------------------------------------------------------- keyboard
    def key(self, name: str, press: bool) -> None:
        keysym = _x11.XStringToKeysym(name.encode())
        if keysym == 0:
            raise ValueError(f"unknown keysym {name!r}")
        code = _x11.XKeysymToKeycode(self.dpy, keysym)
        if code == 0:
            raise ValueError(f"no keycode for {name!r}")
        _xtst.XTestFakeKeyEvent(self.dpy, code, 1 if press else 0, CURRENT_TIME)
        self.sync()

    def tap(self, name: str, *, shift: bool = False, ctrl: bool = False) -> None:
        if ctrl:
            self.key("Control_L", True)
        if shift:
            self.key("Shift_L", True)
        self.key(name, True)
        time.sleep(0.005)
        self.key(name, False)
        if shift:
            self.key("Shift_L", False)
        if ctrl:
            self.key("Control_L", False)
        time.sleep(0.008)

    def type_text(self, text: str) -> None:
        """Type ``text`` like a user (``\\n`` presses Return)."""
        for char in text:
            name = self._CHAR_KEYSYMS.get(char, char)
            keysym = _x11.XStringToKeysym(name.encode())
            code = _x11.XKeysymToKeycode(self.dpy, keysym)
            shift = False
            if code and _x11.XkbKeycodeToKeysym(self.dpy, code, 0, 0) != keysym:
                shift = _x11.XkbKeycodeToKeysym(self.dpy, code, 0, 1) == keysym
            self.tap(name, shift=shift)

    # ----------------------------------------------------------------- windows
    def top_levels(self) -> list[TopLevel]:
        root_return, parent = ctypes.c_ulong(), ctypes.c_ulong()
        children = ctypes.POINTER(ctypes.c_ulong)()
        count = ctypes.c_uint()
        _x11.XQueryTree(
            self.dpy, self.root, ctypes.byref(root_return), ctypes.byref(parent),
            ctypes.byref(children), ctypes.byref(count),
        )  # fmt: skip
        ids = [children[k] for k in range(count.value)]
        if children:
            _x11.XFree(children)
        result = []
        for window in ids:
            attributes = _WindowAttributes()
            if not _x11.XGetWindowAttributes(self.dpy, window, ctypes.byref(attributes)):
                continue
            name_pointer = ctypes.c_char_p()
            name = ""
            if _x11.XFetchName(self.dpy, window, ctypes.byref(name_pointer)) and name_pointer.value:
                name = name_pointer.value.decode("latin-1", "replace")
            x, y = self.origin(window)
            result.append(
                TopLevel(
                    window, name, self.window_pid(window), x, y, attributes.width,
                    attributes.height, attributes.map_state == IS_VIEWABLE,
                )
            )  # fmt: skip
        return result

    def origin(self, window: int) -> tuple[int, int]:
        x, y = ctypes.c_int(), ctypes.c_int()
        child = ctypes.c_ulong()
        _x11.XTranslateCoordinates(
            self.dpy, window, self.root, 0, 0, ctypes.byref(x), ctypes.byref(y),
            ctypes.byref(child),
        )  # fmt: skip
        return x.value, y.value

    def window_pid(self, window: int) -> int | None:
        actual_type, actual_format = ctypes.c_ulong(), ctypes.c_int()
        items, remaining = ctypes.c_ulong(), ctypes.c_ulong()
        data = ctypes.c_void_p()
        status = _x11.XGetWindowProperty(
            self.dpy, window, self._net_wm_pid, 0, 1, 0, 0, ctypes.byref(actual_type),
            ctypes.byref(actual_format), ctypes.byref(items), ctypes.byref(remaining),
            ctypes.byref(data),
        )  # fmt: skip
        pid = None
        if status == 0 and data.value and items.value == 1:
            pid = ctypes.cast(data, ctypes.POINTER(ctypes.c_ulong))[0]
        if data.value:
            _x11.XFree(data)
        return int(pid) if pid is not None else None

    def viewable_windows(self, pid: int | None = None) -> list[TopLevel]:
        return [w for w in self.top_levels() if w.viewable and (pid is None or w.pid == pid)]

    def focus(self, window: int) -> None:
        """Give ``window`` the keyboard focus - what a window manager does when the user
        switches to it."""
        _x11.XRaiseWindow(self.dpy, window)
        _x11.XSetInputFocus(self.dpy, window, REVERT_TO_PARENT, CURRENT_TIME)
        self.sync()

    def input_focus(self) -> int:
        window, revert = ctypes.c_ulong(), ctypes.c_int()
        _x11.XGetInputFocus(self.dpy, ctypes.byref(window), ctypes.byref(revert))
        return window.value

    def try_grab_pointer(self) -> int:
        """Grab and release the pointer; the grab status (0 = success: nobody else holds an
        active pointer grab)."""
        status = _x11.XGrabPointer(
            self.dpy, self.root, 0, 0, GRAB_MODE_ASYNC, GRAB_MODE_ASYNC, 0, 0, CURRENT_TIME
        )
        _x11.XUngrabPointer(self.dpy, CURRENT_TIME)
        self.sync()
        return status

    def try_grab_keyboard(self) -> int:
        """Grab and release the keyboard; 0 = success (no other client holds a grab)."""
        status = _x11.XGrabKeyboard(
            self.dpy, self.root, 0, GRAB_MODE_ASYNC, GRAB_MODE_ASYNC, CURRENT_TIME
        )
        _x11.XUngrabKeyboard(self.dpy, CURRENT_TIME)
        self.sync()
        return status

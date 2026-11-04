"""The native GUI frontend (PySide6 / Qt 6).

* :class:`RemoteFrontend` - default: the window runs in its own process.
* :class:`QtFrontend` - the window runs inside the training process.

Both are imported on first use, so ``import hollywood_training`` never needs a display
and the GUI process (which only needs the window) never imports torch.
"""

from importlib import import_module

from .animation import AnimationClock
from .window import MainWindow

_LAZY = {
    "QtFrontend": ".app",
    "QtPump": ".app",
    "ensure_display": ".app",
    "RemoteFrontend": ".remote",
    "WindowProcess": ".window_process",
}

__all__ = [
    "AnimationClock",
    "MainWindow",
    "QtFrontend",
    "QtPump",
    "RemoteFrontend",
    "WindowProcess",
    "ensure_display",
]


def __getattr__(name: str):
    # See hollywood_training/__init__.py: torch-dependent exports load on first use.
    module = _LAZY.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(module, __name__), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(__all__)

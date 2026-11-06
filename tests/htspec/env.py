"""Process environment of the harness.

:func:`configure_current_process` runs from ``tests/conftest.py`` before Qt or torch load:

* Qt renders **offscreen** on a 1920 x 1080 virtual screen (``htspec/offscreen.json``), so
  the window gets a realistic size and no real display is ever used;
* ``DISPLAY`` / ``WAYLAND_DISPLAY`` are removed: nothing in the test process - or in the
  window processes it starts - can reach the desktop the tests were started from (the X11
  tests use a private Xvfb server instead, see :mod:`htspec.x11`);
* temporary files, caches and bytecode go to ``<project>/.temp/`` (CONSTRICTION.md).

:func:`subprocess_env` builds the same environment for child processes.
"""

from __future__ import annotations

import os
import sys
import tempfile

from . import paths

OFFSCREEN_PLATFORM = f"offscreen:configfile={paths.OFFSCREEN_CONFIG}"

_TEMP_VARIABLES = {
    "TMPDIR": paths.TMPDIR,
    "XDG_CACHE_HOME": paths.XDG_CACHE,
    "XDG_RUNTIME_DIR": paths.XDG_RUNTIME,
    "PYTHONPYCACHEPREFIX": paths.PYCACHE,
}
_DISPLAY_VARIABLES = ("DISPLAY", "WAYLAND_DISPLAY", "QT_QPA_PLATFORMTHEME")


def configure_current_process() -> None:
    paths.ensure_temp_dirs()
    for name, folder in _TEMP_VARIABLES.items():
        os.environ[name] = str(folder)
    tempfile.tempdir = str(paths.TMPDIR)  # the tempfile module may have cached /tmp already
    sys.pycache_prefix = str(paths.PYCACHE)
    os.environ["ONEDNN_JIT_PROFILE"] = "0"  # oneDNN would write /tmp/perf-<pid>.map files
    os.environ["PIP_DISABLE_PIP_VERSION_CHECK"] = "1"
    os.environ["PYTHONUNBUFFERED"] = "1"
    os.environ["QT_QPA_PLATFORM"] = OFFSCREEN_PLATFORM
    os.environ["PYTEST_QT_API"] = "pyside6"
    os.environ.setdefault("QT_LOGGING_RULES", "qt.qpa.*=false")
    for name in _DISPLAY_VARIABLES:
        os.environ.pop(name, None)


def subprocess_env(extra: dict[str, str] | None = None, *, display: str | None = None) -> dict:
    """Environment for a child process: offscreen Qt, unless ``display`` names the private
    X server (then ``QT_QPA_PLATFORM=xcb``)."""
    env = dict(os.environ)
    for name, folder in _TEMP_VARIABLES.items():
        env[name] = str(folder)
    env["ONEDNN_JIT_PROFILE"] = "0"
    env["PYTHONUNBUFFERED"] = "1"
    for name in _DISPLAY_VARIABLES:
        env.pop(name, None)
    if display is None:
        env["QT_QPA_PLATFORM"] = OFFSCREEN_PLATFORM
    else:
        env["QT_QPA_PLATFORM"] = "xcb"
        env["DISPLAY"] = display
    pythonpath = [str(paths.PROJECT_ROOT), str(paths.HARNESS_DIR)]
    if env.get("PYTHONPATH"):
        pythonpath += [p for p in env["PYTHONPATH"].split(os.pathsep) if p not in pythonpath]
    env["PYTHONPATH"] = os.pathsep.join(pythonpath)
    if extra:
        env.update(extra)
    return env

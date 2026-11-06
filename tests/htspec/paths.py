"""Locations used by the harness.

Every temporary file of the harness lives under ``<project>/.temp/`` (CONSTRICTION.md):
``.temp/tmp`` is ``TMPDIR`` (pytest's ``tmp_path`` folders, multiprocessing sockets, torch),
``.temp/tests`` holds the coverage reports and Qt's private runtime folder.
"""

from __future__ import annotations

from pathlib import Path

HARNESS_DIR = Path(__file__).resolve().parents[1]  # <project>/tests
PROJECT_ROOT = HARNESS_DIR.parent
PACKAGE_DIR = PROJECT_ROOT / "hollywood_training"
DEMOS_DIR = PROJECT_ROOT / "demos"
WIKI_DIR = PROJECT_ROOT / "wiki"
SPECS_DIR = PROJECT_ROOT / "specs"
README = PROJECT_ROOT / "README.md"
PYPROJECT = PROJECT_ROOT / "pyproject.toml"

SUPPORT_DIR = HARNESS_DIR / "htspec"
PROBE_DIR = SUPPORT_DIR / "probe"
SCENARIO_DIR = SUPPORT_DIR / "scenarios"
OFFSCREEN_CONFIG = SUPPORT_DIR / "offscreen.json"

TEMP_ROOT = PROJECT_ROOT / ".temp"
TMPDIR = TEMP_ROOT / "tmp"
HARNESS_TEMP = TEMP_ROOT / "tests"
XDG_CACHE = TEMP_ROOT / "xdg-cache"
XDG_RUNTIME = HARNESS_TEMP / "xdg-runtime"
PYCACHE = TEMP_ROOT / "pycache"
REPORTS = HARNESS_TEMP / "reports"


def ensure_temp_dirs() -> None:
    for folder in (TMPDIR, HARNESS_TEMP, XDG_CACHE, PYCACHE, REPORTS):
        folder.mkdir(parents=True, exist_ok=True)
    XDG_RUNTIME.mkdir(parents=True, exist_ok=True)
    XDG_RUNTIME.chmod(0o700)  # Qt insists on a private runtime folder

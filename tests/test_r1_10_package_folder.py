"""R1 requirement 10 - "**ALL** package file **MUST** be located in the package folder
`hollywood_training`."

* the build configuration only packages ``hollywood_training`` (and its subpackages) and
  ships the package data from inside it;
* no package module lives outside the folder: the only Python files of the project outside
  ``hollywood_training/`` are the demos (``demos/``) and this harness (``tests/``);
* the built distribution (a wheel, built from a copy of the project in a temporary folder,
  offline) contains nothing but the package folder and its metadata - and all of it.
"""

from __future__ import annotations

import importlib.util
import shutil
import zipfile

import pytest
from htspec import paths
from htspec.subproc import run_python

pytestmark = pytest.mark.req("R1.10")

OUTSIDE_ALLOWED = {"demos", "tests"}  # demos (R1.11) and the test harness
IGNORED = {".git", ".venv", ".temp", "__pycache__", "build", "dist"}


def project_python_files():
    for path in paths.PROJECT_ROOT.rglob("*.py"):
        relative = path.relative_to(paths.PROJECT_ROOT)
        if not IGNORED.intersection(relative.parts) and not relative.parts[0].endswith(".egg-info"):
            yield relative


def test_the_build_configuration_packages_only_the_package_folder():
    tomllib = pytest.importorskip("tomllib")  # Python 3.11+
    config = tomllib.loads(paths.PYPROJECT.read_text(encoding="utf-8"))
    setuptools = config["tool"]["setuptools"]
    assert setuptools["packages"]["find"]["include"] == ["hollywood_training*"]
    assert set(setuptools.get("package-data", {})) <= {"hollywood_training"}
    for files in setuptools.get("package-data", {}).values():
        for name in files:
            assert (paths.PACKAGE_DIR / name).exists(), name


def test_no_python_file_lives_outside_the_known_folders():
    outside = [
        str(path) for path in project_python_files()
        if path.parts[0] != "hollywood_training" and path.parts[0] not in OUTSIDE_ALLOWED
    ]  # fmt: skip
    assert outside == []


def test_no_package_module_lives_outside_the_package_folder():
    """Nothing outside the folder belongs to the package (by name or by import path)."""
    strays = [
        str(path) for path in project_python_files()
        if path.parts[0] != "hollywood_training" and "hollywood_training" in path.stem
    ]  # fmt: skip
    assert strays == []
    assert (paths.PACKAGE_DIR / "__init__.py").is_file()
    assert (paths.PACKAGE_DIR / "py.typed").is_file()


@pytest.mark.subprocess
@pytest.mark.slow
@pytest.mark.timeout(600)
def test_the_built_wheel_contains_exactly_the_package_folder(tmp_path):
    if importlib.util.find_spec("setuptools") is None:
        pytest.skip("setuptools is not installed in this environment: no offline build")
    copy = tmp_path / "project"
    shutil.copytree(
        paths.PROJECT_ROOT,
        copy,
        ignore=shutil.ignore_patterns(
            ".git", ".venv", ".temp", "__pycache__", "*.egg-info", "build", "dist", "tests"
        ),
    )
    out = tmp_path / "wheel"
    completed = run_python(
        ["-m", "pip", "wheel", "--no-deps", "--no-build-isolation", "--no-cache-dir",
         "--disable-pip-version-check", "-q", "-w", str(out), str(copy)],
        timeout=540,
    )  # fmt: skip
    assert completed.returncode == 0, completed.stderr[-4000:]
    wheels = list(out.glob("hollywood_training-*.whl"))
    assert len(wheels) == 1, list(out.iterdir())
    names = zipfile.ZipFile(wheels[0]).namelist()
    outside = [
        name for name in names
        if not name.startswith("hollywood_training/") and ".dist-info/" not in name
    ]  # fmt: skip
    assert outside == [], f"files outside the package folder: {outside}"
    shipped = {name for name in names if name.startswith("hollywood_training/")}
    expected = {
        path.relative_to(paths.PROJECT_ROOT).as_posix()
        for path in paths.PACKAGE_DIR.rglob("*")
        if path.is_file() and "__pycache__" not in path.parts
    }
    assert expected <= shipped, f"package files missing from the wheel: {expected - shipped}"

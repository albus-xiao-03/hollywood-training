"""R10: all package files live in ``hollywood_training``.
R11: a demo with an illustrative network and dataset; every demo file in ``demos/``,
directly under the project folder and outside the package."""

from __future__ import annotations

import ast
import os
import sys
from pathlib import Path

import pytest

import hollywood_training as ht

from .helpers import DEMOS_DIR, PACKAGE_DIR, PROJECT_ROOT, parse_source, run_python

IGNORED_DIRECTORIES = {".git", ".venv", "venv", ".temp", "__pycache__", "build", "dist"}
DEMO_SCRIPTS = sorted(
    path for path in DEMOS_DIR.glob("*.py") if "__main__" in path.read_text(encoding="utf-8")
)


def _project_files(suffix: str) -> list[Path]:
    """Project files with ``suffix``, relative to the project (environments and caches skipped)."""
    found = []
    for folder, directories, files in os.walk(PROJECT_ROOT):
        directories[:] = [
            d for d in directories if d not in IGNORED_DIRECTORIES and not d.endswith(".egg-info")
        ]
        found += [
            (Path(folder) / name).relative_to(PROJECT_ROOT)
            for name in files
            if name.endswith(suffix)
        ]
    return found


def _toml(path):
    if sys.version_info < (3, 11):
        pytest.skip("tomllib needs Python 3.11+")
    import tomllib

    return tomllib.loads(path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------- R10
@pytest.mark.spec("R10")
def test_all_python_code_outside_the_package_is_demo_or_test_code():
    outside = [
        str(path)
        for path in _project_files(".py")
        if path.parts[0] not in ("hollywood_training", "demos", "tests")
    ]
    assert outside == []


@pytest.mark.spec("R10")
def test_there_is_exactly_one_importable_package_in_the_project():
    packages = sorted(
        path.parent.relative_to(PROJECT_ROOT).parts[0]
        for path in PROJECT_ROOT.glob("*/__init__.py")
        if path.parent.name not in IGNORED_DIRECTORIES
    )
    assert packages in (["hollywood_training"], ["hollywood_training", "tests"])
    assert not (DEMOS_DIR / "__init__.py").exists()


@pytest.mark.spec("R10")
def test_the_distribution_ships_only_the_package_folder():
    config = _toml(PROJECT_ROOT / "pyproject.toml")
    setuptools = config.get("tool", {}).get("setuptools", {})
    include = setuptools.get("packages", {}).get("find", {}).get("include", [])
    assert include and all(pattern.startswith("hollywood_training") for pattern in include)
    assert set(setuptools.get("package-data", {})) <= {"hollywood_training"}
    assert config["project"]["name"] == "hollywood-training"


@pytest.mark.spec("R10")
def test_package_data_files_are_inside_the_package():
    config = _toml(PROJECT_ROOT / "pyproject.toml")
    patterns = config["tool"]["setuptools"].get("package-data", {}).get("hollywood_training", [])
    for pattern in patterns:
        assert list(PACKAGE_DIR.glob(pattern)), f"package data {pattern!r} is not in the package"
    assert (PACKAGE_DIR / "py.typed").is_file()


# ---------------------------------------------------------------------- R11
@pytest.mark.spec("R11")
def test_the_demos_folder_sits_directly_under_the_project_outside_the_package():
    assert DEMOS_DIR.is_dir() and DEMOS_DIR.parent == PROJECT_ROOT
    assert DEMO_SCRIPTS, "no runnable demo script in demos/"
    assert not [p for p in PACKAGE_DIR.rglob("*") if "demo" in p.name.lower()]


@pytest.mark.spec("R11")
@pytest.mark.parametrize("script", DEMO_SCRIPTS, ids=lambda path: path.name)
def test_each_demo_builds_a_network_and_a_dataset_from_public_api(script):
    tree = parse_source(script)
    support = {path.stem: parse_source(path) for path in DEMOS_DIR.glob("*.py") if path != script}
    used_attributes = {
        node.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id == "ht"
    }
    assert used_attributes, "the demo does not use the package as `ht`"
    assert used_attributes <= set(ht.__all__), used_attributes - set(ht.__all__)
    source = script.read_text(encoding="utf-8") + "".join(
        ast.unparse(module) for module in support.values()
    )
    defines_network = "nn.Module" in source or "nn.Sequential" in source
    defines_data = any(word in source for word in ("TensorDataset", "Dataset", "DataLoader"))
    assert defines_network and defines_data


@pytest.mark.spec("R11")
@pytest.mark.parametrize("script", DEMO_SCRIPTS, ids=lambda path: path.name)
def test_each_demo_explains_its_options(script):
    completed = run_python([str(script), "--help"], timeout=120, cwd=PROJECT_ROOT)
    assert completed.returncode == 0, completed.stderr
    assert "--epochs" in completed.stdout and "--no-gui" in completed.stdout


@pytest.mark.spec("R11")
@pytest.mark.slow
@pytest.mark.e2e
@pytest.mark.timeout(300)
@pytest.mark.parametrize("script", DEMO_SCRIPTS, ids=lambda path: path.name)
def test_each_demo_trains_in_the_console(script):
    completed = run_python([str(script), "--epochs", "3", "--no-gui"], cwd=PROJECT_ROOT)
    assert completed.returncode == 0, completed.stderr
    assert "3 epochs" in completed.stdout and "finished" in completed.stdout.lower()


@pytest.mark.spec("R11", "R4")
@pytest.mark.slow
@pytest.mark.e2e
@pytest.mark.gui
@pytest.mark.timeout(300)
@pytest.mark.parametrize("script", DEMO_SCRIPTS, ids=lambda path: path.name)
def test_each_demo_trains_in_the_window(script):
    completed = run_python(
        [str(script), "--epochs", "3", "--close-on-finish"], cwd=PROJECT_ROOT, timeout=240
    )
    assert completed.returncode == 0, completed.stderr
    assert "3 epochs" in completed.stdout and "finished" in completed.stdout.lower()

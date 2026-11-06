"""R1 requirement 11 - "The package **MUST** provide a demo with illustrative network
architecture and training dataset. **ALL** demo related files **MUST** be located in the
folder `demos`, outside of the package folder `hollywood_training`, directly under the
project folder."

* ``demos/`` sits directly in the project folder; no demo file is inside the package (and
  the package does not depend on the demos);
* every demo script defines an illustrative network and a training dataset, and runs: its
  ``--help``, a console run (``--no-gui``) and a run with the window (offscreen) that closes
  by itself at the end.
"""

from __future__ import annotations

import ast
import re

import pytest
from htspec import paths
from htspec.subproc import run_python

pytestmark = pytest.mark.req("R1.11")


def demo_scripts():
    return sorted(
        path for path in paths.DEMOS_DIR.glob("*.py")
        if "__main__" in path.read_text(encoding="utf-8")
    )  # fmt: skip


def local_sources(script) -> str:
    """The script plus the demos' own helper modules it imports (e.g. demo_data)."""
    text = script.read_text(encoding="utf-8")
    for node in ast.walk(ast.parse(text)):
        if isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            helper = paths.DEMOS_DIR / f"{node.module}.py"
            if helper.exists():
                text += "\n" + helper.read_text(encoding="utf-8")
    return text


def test_the_demos_folder_is_directly_in_the_project_folder():
    assert paths.DEMOS_DIR.is_dir()
    assert paths.DEMOS_DIR.parent == paths.PROJECT_ROOT
    assert demo_scripts(), "no runnable demo script in demos/"
    assert not (paths.PACKAGE_DIR / "demos").exists()


def test_no_demo_file_is_inside_the_package():
    inside = [str(p) for p in paths.PACKAGE_DIR.rglob("*") if "demo" in p.name.lower()]
    assert inside == []
    for path in paths.PACKAGE_DIR.rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                names = [a.name for a in node.names] + [getattr(node, "module", "") or ""]
                assert not any(n.split(".")[0] in ("demos", "demo_data") for n in names), path


@pytest.mark.parametrize("script", demo_scripts(), ids=lambda p: p.name)
def test_each_demo_has_an_illustrative_network_and_a_training_dataset(script):
    source = local_sources(script)
    tree = ast.parse(source)
    defines_network = (
        any(
            isinstance(node, ast.ClassDef)
            and any(ast.unparse(base).endswith("Module") for base in node.bases)
            for node in ast.walk(tree)
        )
        or "nn.Sequential(" in source
    )
    assert defines_network, "no network (nn.Module subclass or nn.Sequential) in the demo"
    assert "TensorDataset" in source or "Dataset" in source, "no dataset in the demo"
    assert "DataLoader(" in source, "the dataset is not fed to training"
    assert "ht.run(" in source and "TrainingProcess(" in source


@pytest.mark.subprocess
@pytest.mark.parametrize("script", demo_scripts(), ids=lambda p: p.name)
def test_each_demo_explains_its_options(script):
    completed = run_python([str(script), "--help"], timeout=120)
    assert completed.returncode == 0, completed.stderr[-2000:]
    assert "usage:" in completed.stdout and "--epochs" in completed.stdout


@pytest.mark.subprocess
@pytest.mark.timeout(360)
@pytest.mark.parametrize("script", demo_scripts(), ids=lambda p: p.name)
def test_each_demo_runs_on_the_console(script):
    completed = run_python([str(script), "--no-gui", "--epochs", "3"], timeout=300)
    assert completed.returncode == 0, completed.stderr[-3000:]
    out = completed.stdout
    assert re.search(r"^epoch 1/3 \|", out, re.M) and re.search(r"^epoch 3/3 \|", out, re.M)
    assert re.search(r"^finished: 3 epochs in", out, re.M), out[-2000:]


@pytest.mark.subprocess
@pytest.mark.slow
@pytest.mark.timeout(360)
@pytest.mark.parametrize("script", demo_scripts(), ids=lambda p: p.name)
def test_each_demo_runs_with_the_window(script):
    completed = run_python([str(script), "--epochs", "3", "--close-on-finish"], timeout=300)
    assert completed.returncode == 0, completed.stderr[-3000:]
    assert re.search(r"^finished: 3 epochs in", completed.stdout, re.M), completed.stdout[-2000:]

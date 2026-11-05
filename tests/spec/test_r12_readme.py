"""R12: the package has a README.md for quick start - and its quick start works."""

from __future__ import annotations

import ast
import inspect
import re

import pytest
import torch

import hollywood_training as ht
from hollywood_training.commands import builtin_commands

from .helpers import README, install_child_probe, run_python

pytestmark = pytest.mark.spec("R12")


def readme_text() -> str:
    return README.read_text(encoding="utf-8")


def section(title: str) -> str:
    """The markdown section under the heading ``title`` (case-insensitive)."""
    text = readme_text()
    match = re.search(rf"^(#+)\s+{re.escape(title)}\s*$", text, re.IGNORECASE | re.MULTILINE)
    assert match, f"README has no '{title}' section"
    level = len(match.group(1))
    rest = text[match.end() :]
    end = re.search(rf"^#{{1,{level}}}\s", rest, re.MULTILINE)
    return rest[: end.start()] if end else rest


def quick_start_code() -> str:
    blocks = re.findall(r"```python\n(.*?)```", section("Quick start"), re.DOTALL)
    assert blocks, "the quick start has no python code block"
    return blocks[0]


def test_the_readme_exists_with_installation_and_quick_start():
    assert README.is_file()  # directly in the project folder
    install = section("Installation")
    assert "pip install" in install and "torch" in install.lower()
    assert quick_start_code()


def test_the_quick_start_shows_the_three_steps():
    code = quick_start_code()
    tree = ast.parse(code)  # complete, valid Python
    used = {
        node.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id == "ht"
    }
    assert {"track", "TrainingProcess", "run"} <= used
    assert "train_step" in code  # the native step, wrapped


def test_the_quick_start_code_trains_as_written():
    torch.manual_seed(0)
    namespace: dict = {"__name__": "readme_quick_start"}  # its `if __name__ == "__main__"` waits
    exec(compile(quick_start_code(), "README.md", "exec"), namespace)
    process = namespace["process"]
    assert isinstance(process, ht.TrainingProcess)
    result = ht.run(process, gui=False, log_every=1000)
    assert result.state is ht.TrainingState.FINISHED
    assert result.epochs_completed == process.epochs
    assert result.train_losses[-1] < result.train_losses[0]


@pytest.mark.e2e
@pytest.mark.slow
@pytest.mark.gui
@pytest.mark.spec("R4")
@pytest.mark.timeout(300)
def test_the_quick_start_script_runs_verbatim_with_the_window(monkeypatch, tmp_path):
    """Run the README code unchanged as a script; a scripted user closes the window once
    training is complete (the quick start keeps the window open until it is closed)."""
    probe = install_child_probe(
        monkeypatch,
        tmp_path,
        [{"on": {"event": "StateChanged", "state": "finished", "nth": 1}, "do": [{"close": True}]}],
        deadline=200,
    )
    script = tmp_path / "quick_start.py"
    script.write_text(quick_start_code(), encoding="utf-8")
    completed = run_python([str(script)], cwd=tmp_path, timeout=240)
    assert completed.returncode == 0, completed.stderr
    float(completed.stdout.strip().splitlines()[-1])  # it prints the final training loss
    probe.assert_installed()
    epochs = [row["epoch"] for row in probe.applied("EpochCompleted")]
    assert epochs == list(range(1, len(epochs) + 1)) and len(epochs) >= 1


def test_every_run_option_is_documented():
    documented = readme_text()
    options = [name for name in inspect.signature(ht.run).parameters if name != "process"]
    assert [name for name in options if f"`{name}`" not in documented] == []


def test_every_built_in_command_is_documented():
    documented = readme_text()
    missing = [c.name for c in builtin_commands() if f"`{c.name}" not in documented]
    assert missing == []


def test_relative_links_in_the_readme_resolve():
    text = re.sub(r"```.*?```", "", readme_text(), flags=re.DOTALL)
    targets = re.findall(r"\]\(([^)\s]+)\)", text)
    broken = [
        target
        for target in targets
        if not re.match(r"^[a-z]+:", target)
        and not target.startswith("#")
        and not (README.parent / target.split("#")[0]).exists()
    ]
    assert broken == []

"""The demos run end to end (console and GUI), as a user would start them."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

DEMOS = Path(__file__).resolve().parents[1] / "demos"


def run_demo(name: str, *args: str) -> subprocess.CompletedProcess:
    env = dict(os.environ, QT_QPA_PLATFORM=os.environ.get("QT_QPA_PLATFORM", "offscreen"))
    return subprocess.run(
        [sys.executable, str(DEMOS / name), *args],
        capture_output=True,
        text=True,
        timeout=240,
        env=env,
        check=False,
    )


@pytest.mark.parametrize("demo", ["spiral_classifier.py", "sine_regression.py"])
def test_demo_console(demo):
    completed = run_demo(demo, "--epochs", "4", "--no-gui")
    assert completed.returncode == 0, completed.stderr
    assert "finished: 4 epochs" in completed.stdout


@pytest.mark.parametrize("demo", ["spiral_classifier.py", "sine_regression.py"])
def test_demo_gui(demo):
    completed = run_demo(demo, "--epochs", "3", "--close-on-finish")
    assert completed.returncode == 0, completed.stderr
    assert "finished: 3 epochs" in completed.stdout

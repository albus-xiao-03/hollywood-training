"""R1 requirement 12 - "The package **MUST** have a `README.md` for quick start."

* ``README.md`` is in the project folder, with an installation section and a quick start;
* the quick start's example **runs as written** - extracted from the README and executed
  unmodified, with the real window (offscreen); the harness's window-process probe closes
  the window once training has finished, as the user would;
* every relative link of the README points to a file that exists.
"""

from __future__ import annotations

import re

import pytest
from htspec import paths
from htspec.probe_log import LINGER_CLOSE, ProbeLog, probe_env
from htspec.subproc import run_python

pytestmark = pytest.mark.req("R1.12")


def readme() -> str:
    return paths.README.read_text(encoding="utf-8")


def section(text: str, title: str) -> str:
    match = re.search(rf"^##+ {re.escape(title)}\s*$(.*?)(?=^## |\Z)", text, re.M | re.S | re.I)
    assert match, f"no '{title}' section"
    return match.group(1)


def quick_start_code() -> str:
    blocks = re.findall(r"```python\n(.*?)```", section(readme(), "Quick start"), re.S)
    assert blocks, "the quick start has no Python example"
    return blocks[0]


def test_the_readme_is_in_the_project_folder():
    assert paths.README.is_file()
    text = readme()
    assert text.startswith("# ") and len(text.split()) > 200


def test_the_readme_explains_the_installation():
    install = section(readme(), "Installation")
    assert "pip install" in install


def test_the_quick_start_shows_the_whole_workflow():
    code = quick_start_code()
    for step in ("import hollywood_training as ht", "@ht.track", "ht.TrainingProcess(", "ht.run("):
        assert step in code, step


@pytest.mark.subprocess
@pytest.mark.slow
@pytest.mark.timeout(600)
def test_the_quick_start_example_runs_as_written(tmp_path):
    script = tmp_path / "quick_start.py"
    script.write_text(quick_start_code(), encoding="utf-8")
    log = ProbeLog(tmp_path / "probe.jsonl")
    env = probe_env(log.path, script={"steps": [LINGER_CLOSE]})
    completed = run_python([str(script)], env=env, timeout=540, cwd=tmp_path)
    assert completed.returncode == 0, completed.stderr[-4000:]
    last = completed.stdout.strip().splitlines()[-1]
    assert float(last) > 0  # it prints the last training loss
    assert any(s["visible"] for s in log.states()), "the example did not show the window"


def test_every_relative_link_of_the_readme_exists():
    broken = []
    for target in re.findall(r"\]\(([^)\s]+)\)", readme()):
        if re.match(r"^[a-z]+:", target) or target.startswith("#"):
            continue
        path = (paths.README.parent / target.split("#", 1)[0]).resolve()
        if not path.exists():
            broken.append(target)
    assert broken == []

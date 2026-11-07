# Development

## Setup

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install torch --index-url https://download.pytorch.org/whl/cpu   # or your CUDA build
pip install -e ".[dev]"
```

## Tests

The tests in `tests/` are the SPEC harness. They check the package against the
requirements in `specs/R1.md` and `specs/R2.md`, with one test module per requirement.

```bash
tests/run_tests.sh                                    # everything
tests/run_tests.sh -m "not x11"                       # without the real-X11 tests
tests/run_tests.sh -m "not x11 and not subprocess and not slow"   # the quick part
tests/run_tests.sh tests/test_r1_01_tracking.py       # one requirement
```

`tests/run_tests.sh` activates the project's venv, keeps every temporary file and cache in
`.temp/`, and runs `python -m pytest` with the extra arguments. A plain `pytest` from the
project folder works too, because `pyproject.toml` points `testpaths` at `tests`. At the
end of a run, a table counts the passed, failed and skipped tests per requirement.

* **Headless.** `tests/conftest.py` removes `DISPLAY` and `WAYLAND_DISPLAY`, and Qt renders
  offscreen. The `x11` tests start their own private `Xvfb`, so no `xvfb-run` is needed.
  They are skipped when Xvfb or the X client libraries are missing.
* **Markers.** `req(...)` names the requirement a test verifies, and every test has one.
  `x11` marks the real-X11 tests, `subprocess` the tests that start processes, and `slow`
  the tests that take more than a few seconds.
* Every test of an R1 requirement also counts for R2.5 ("all other R1 features must not
  be affected").

| Requirement | Module |
|---|---|
| R1.1 tracking | `test_r1_01_tracking.py` |
| R1.2 training management | `test_r1_02_training_management.py` |
| R1.3 native workflow | `test_r1_03_native_workflow.py` |
| R1.4 native GUI, animations | `test_r1_04_native_gui.py` |
| R1.5 synchronisation | `test_r1_05_synchronization.py` |
| R1.6 speed | `test_r1_06_speed.py` |
| R1.7 commands | `test_r1_07_commands.py` |
| R1.8 no dark magic | `test_r1_08_no_dark_magic.py` |
| R1.9 modules, architecture | `test_r1_09_modular_architecture.py` |
| R1.10 package folder | `test_r1_10_package_folder.py` |
| R1.11 demos | `test_r1_11_demos.py` |
| R1.12 README | `test_r1_12_readme.py` |
| R1.13 wiki | `test_r1_13_wiki.py` |
| GUI 1–7: network view, status, loss chart, usage, terminal, network name, blue theme | `test_r1_g1_network_view.py` … `test_r1_g7_blue_theme.py` |
| R2.1 no pointer inside the window | `test_r2_1_no_pointer.py` |
| R2.2 mouse interactions disabled | `test_r2_2_mouse_disabled.py` |
| R2.3 everything without the mouse | `test_r2_3_keyboard_only.py` |
| R2.4 replacement commands | `test_r2_4_view_commands.py` |
| R2.5 R1 unaffected | `test_r2_5_r1_unaffected.py`, plus every R1 test |
| R2.6 outside the window unaffected | `test_r2_6_outside_window.py` |
| R2.7 keyboard rerouted to the terminal | `test_r2_7_keyboard_reroute.py` |

`tests/README.md` describes the harness in full: the levels it tests at (core, offscreen
window, real runs, the window process, real X11), its helpers in `tests/htspec/` and its
safety nets.

## Style

```bash
ruff check .            # lint (rules in pyproject.toml)
ruff format .           # formatting
```

Conventions used across the package:

* **Layers depend only downward.** Tracking → sync → training / commands → frontends →
  runner ([Architecture](Architecture)). Only `gui` imports Qt.
* **The data contract is torch-free.** Modules whose objects cross to the window
  process (`tracking.architecture`, `tracking.roles`, `tracking.weights`,
  `monitor.usage`, `sync.*`) must not import torch at runtime. Package `__init__`s
  export torch-dependent names lazily, through an explicit `_LAZY` map and
  `__getattr__` (PEP 562).
* **Data crossing threads or processes is immutable:** frozen slot dataclasses and
  read-only numpy arrays.
* **No "dark magic".** Only public PyTorch APIs (`named_modules`, forward hooks,
  `param_groups`, `DataLoader.batch_size`, `type(obj).__name__`). User code is never
  inspected.
* **GUI drawing performance:** prefer 1 px pens, cache static layers in pixmaps, keep
  per-frame Python work small ([Performance](Performance)).

## Releasing

The version lives in `hollywood_training/_version.py`. Build with
`python -m build` (or `pip wheel . --no-deps`).

## Publishing this wiki

The `wiki/` folder uses GitHub wiki conventions: `Home.md`, `_Sidebar.md`,
`_Footer.md`, and links written as `[Text](Page-Name)`. Images live in
`wiki/images/`. To publish, clone the repository's wiki
(`https://github.com/<owner>/<repo>.wiki.git`), copy the folder's content into it, and
push.

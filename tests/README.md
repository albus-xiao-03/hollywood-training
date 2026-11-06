# SPEC harness

A pytest harness that tests the package against the requirements in
[`specs/R1.md`](../specs/R1.md) and [`specs/R2.md`](../specs/R2.md). The specification
documents are its only ground truth for *what* must hold. The package's code, README and
wiki are used only to learn *how* to drive the package, and to know what each feature is
documented to do (R2.3 requires every feature to keep working without the mouse). The
harness was written from scratch for the R2 version of the package (branch
`feature/keyboard-driven-control`). It replaces the build-phase test suite that was in
`tests/` before.

> **Status:** written, not yet executed. In this step the harness was developed only:
> * its files were checked with `ruff` (lint and format);
> * a parse-only cross-check confirmed that every imported harness or package name and every
>   requested fixture exists, and that every PySide6, torch, numpy and psutil name it uses
>   resolves;
> * two read-only reviews were done, and their findings fixed.
>
> No test has ever run.

## Running it

```bash
tests/run_tests.sh                                    # everything
tests/run_tests.sh -m "not x11"                       # without the real-X11 tests
tests/run_tests.sh -m "not x11 and not subprocess and not slow"   # the quick part
tests/run_tests.sh tests/test_r2_7_keyboard_reroute.py            # one requirement
```

`run_tests.sh` sources `.temp/env_setup/devenv.sh`, which provides the project's venv and
keeps every cache in `.temp/`, then runs `python -m pytest`. A plain `pytest` from the
project folder also works, because `pyproject.toml` already points `testpaths` at `tests`.

At the end of a run, a table shows passed, failed and skipped tests per requirement. It is
also written to `.temp/tests/reports/coverage-<time>.json`. Every test that verifies an R1
requirement also counts for **R2.5** ("all other R1 features must not be affected").

## Safety and environment

* **No real display is used.** `conftest.py` removes `DISPLAY` and `WAYLAND_DISPLAY`, and Qt
  renders offscreen on a virtual 1920 × 1080 screen (`htspec/offscreen.json`). The `x11`
  tests start their own private `Xvfb` on a free display number. All pointer moves, clicks
  and key presses go to that virtual screen only.
* **Temporary files stay in `.temp/`.** This covers `TMPDIR` (pytest's `tmp_path`,
  multiprocessing sockets), the XDG caches, bytecode (`PYTHONPYCACHEPREFIX`), Qt's runtime
  folder and the coverage reports. With pytest-cov (`--cov`), `run_tests.sh` also puts
  coverage's data file there (`COVERAGE_FILE`, default `.temp/coverage/.coverage`). Xvfb
  itself keeps its socket in `/tmp/.X11-unix`, which the X server removes when it stops.
* No test writes into the project tree; files are created only under `tmp_path`.
* Safety nets run after every test:
  * A window process the test left running is ended.
  * A training run still going on its thread is stopped.
* A safety net runs after every module: a live X11 session or a helper process the module
  left running is ended. This can happen when a fixture fails during its setup, because a
  failed setup gets no teardown. Without the net, such a window would stay on the shared
  Xvfb and disturb the modules after it.
* `LiveApp.close()` ends a live X11 session's window process before its script, so no
  window stays on the shared Xvfb.

## How it tests

| Level | Where | Used for |
|---|---|---|
| core | the test process | tracking, process, manager (`htspec.headless.HeadlessRun`: a real `TrainingManager` with a recording frontend), commands, sources, docs |
| window, offscreen | the test process | a real `MainWindow` fed with real-shaped events (`htspec.rig.GuiRig`, `htspec.events.SessionFeeder`); what the panels actually draw (`htspec.painting.PaintCapture` records every painted text, curve and bar); keyboard and mouse input through the native window (`htspec.keys`, `htspec.gui_state`) |
| real in-process run | the test process | `ht.run(process, isolate_gui=False)` driven from the keyboard (`htspec.pilot.Pilot`) |
| real window process | a child process | the default `ht.run(process)`; a read-only probe (`htspec/probe`, loaded through `sitecustomize` and active only in the window process) logs every event, repaint, acknowledgement and window state, and can type keys or send mouse input inside that process |
| real X11 (`x11`) | private Xvfb | a real user script (`htspec/scenarios/train_app.py`) with the window process on Qt's X11 backend: the cursor image the X server really shows (XFixes), real pointer and keyboard input (XTest), and another application next to it (`htspec/scenarios/other_app.py`). `htspec.x11` and `htspec.live` are the client side |

Without a window manager on Xvfb, the tests give a window the keyboard focus themselves,
with `XSetInputFocus`. That is what a window manager does when the user switches to the
window ("when users stay on the window", R2.7).

Markers: `req(...)` names the requirement(s) a test verifies; every test has one, and a
missing or unknown id stops the collection. `x11` marks real-X11 tests (skipped when Xvfb
or the X client libraries are missing), `subprocess` marks tests that start processes, and
`slow` marks tests that take more than a few seconds.

## Requirement map

| Requirement | Module |
|---|---|
| R1.1 tracking | `test_r1_01_tracking.py` |
| R1.2 training management | `test_r1_02_training_management.py` |
| R1.3 native workflow | `test_r1_03_native_workflow.py` |
| R1.4 native GUI, animations | `test_r1_04_native_gui.py` |
| R1.5 synchronisation | `test_r1_05_synchronization.py` |
| R1.6 speed | `test_r1_06_speed.py` |
| R1.7 commands | `test_r1_07_commands.py` (+ `test_r2_4_view_commands.py`) |
| R1.8 no dark magic | `test_r1_08_no_dark_magic.py` |
| R1.9 modules, architecture | `test_r1_09_modular_architecture.py` |
| R1.10 package folder | `test_r1_10_package_folder.py` |
| R1.11 demos | `test_r1_11_demos.py` |
| R1.12 README | `test_r1_12_readme.py` |
| R1.13 wiki | `test_r1_13_wiki.py` |
| GUI 1 network, connections, weights | `test_r1_g1_network_view.py` |
| GUI 2 training status | `test_r1_g2_status_panel.py` |
| GUI 3 loss curves | `test_r1_g3_loss_chart.py` |
| GUI 4 usage | `test_r1_g4_usage.py` |
| GUI 5 terminal | `test_r1_g5_terminal.py` |
| GUI 6 network name | `test_r1_g6_network_name.py` |
| GUI 7 blue based | `test_r1_g7_blue_theme.py` |
| R2.1 no pointer inside the window | `test_r2_1_no_pointer.py` |
| R2.2 mouse interactions disabled | `test_r2_2_mouse_disabled.py` |
| R2.3 everything without the mouse | `test_r2_3_keyboard_only.py` |
| R2.4 replacement commands | `test_r2_4_view_commands.py` |
| R2.5 R1 unaffected | `test_r2_5_r1_unaffected.py`, plus every R1 test |
| R2.6 outside the window unaffected | `test_r2_6_outside_window.py` |
| R2.7 keyboard rerouted to the terminal | `test_r2_7_keyboard_reroute.py` |

That is 415 test functions in 27 modules; parametrization makes the actual number of test
cases larger.

## Strict tests that reading the code suggests will fail

None of this has been executed. Reading the implementation suggests the tests below will
expose deviations from a strict reading of the specification. Their failures are findings
about the package, not harness bugs, and they must not be weakened.

* **R1.1 "the weights of each layer"**:
  `test_r1_01_tracking.py::test_every_weight_matrix_of_the_network_is_tracked`
  * Expected to fail for `Bilinear`, `RNN`, `LSTM`, `GRU`, `MultiheadAttention` and
    `TransformerEncoderLayer`.
  * Tracking only follows modules with a 2-D `weight` (and convolutions and embeddings).
* **GUI 1 "the REAL weights of each layer"**:
  `test_r1_g1_network_view.py::test_the_weights_of_every_layer_are_shown`, for LSTM and
  attention networks. The cause is the same.
* **GUI 6 "its class name"**:
  `test_r1_g6_network_name.py::test_a_compiled_network_is_still_shown_by_its_own_class`.
  The inspector only unwraps `DataParallel` and `DistributedDataParallel`, so a
  `torch.compile`-d network shows as `OptimizedModule`.
* **GUI 1–5 at small window sizes**:
  `test_r1_g1_network_view.py::test_every_panel_stays_fully_visible_at_small_window_sizes`.
  * The right column's minimum panel heights (300 + 200 + 140 px, plus 16 px of splitter
    handles), with the 80 px header and the margins, need about 764 px.
  * The window allows a height of 680 px, which is also where it opens on a 1366 × 768
    screen.

## Layout

```
tests/
  conftest.py          environment, requirement traceability, coverage table, fixtures
  run_tests.sh         the entry point (sources .temp/env_setup/devenv.sh)
  htspec/              support package (no tests)
    env.py paths.py offscreen.json      environment and locations
    models.py events.py                 networks, data, setups; real-shaped session events
    rig.py painting.py gui_state.py keys.py   offscreen window, paint capture, input
    headless.py pilot.py                real manager / real in-process run drivers
    probe/ probe_log.py                 the window-process probe and its log reader
    subproc.py x11.py live.py           subprocesses, private Xvfb + X client, live sessions
    scenarios/train_app.py other_app.py the user script and the other application
  test_r1_*.py, test_r2_*.py            the tests, one module per requirement
```

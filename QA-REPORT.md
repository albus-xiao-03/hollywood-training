# QA report: hollywood-training, R2 (keyboard-driven control)

| | |
|---|---|
| Date | 2026-11-06 (run 20:30:56–20:33:36 UTC) |
| Branch / commit | `feature/keyboard-driven-control` at `eb5a51e` |
| Package code under test | `hollywood_training/`, unchanged since `a20a247` |
| Test harness | `tests/` at `eb5a51e`, executed as committed: no test was modified, added or removed |
| Ground truth | `specs/R1.md`, `specs/R2.md` |
| Raw output | [`TEST-REPORT.md`](TEST-REPORT.md): the exact, unchanged pytest output of this run |
| Package code | not modified (no defect was fixed in this stage) |

This report and `TEST-REPORT.md` replace the earlier reports. Those described a run made before the harness defects were fixed (commits `184f44f`..`eb5a51e`). Everything below comes from this run, the specifications, the package code and the git history.

## 1. Execution

**Command** (from the project root):
```
tests/run_tests.sh --cov=hollywood_training --cov-report=term-missing
```
`run_tests.sh` sources `.temp/env_setup/devenv.sh` (the project's `.venv`, with every cache in `.temp/`) and runs `python -m pytest` with the given arguments. pytest reads `pyproject.toml`: `testpaths = ["tests"]`, `-ra`, a 120 s timeout per test.

**Environment:**
- Linux aarch64, Python 3.13.5 in `.venv`
- pytest 9.2.0, pluggy 1.7.0, pytest-qt 4.5.0, pytest-timeout 2.4.0
- **pytest-cov 7.1.0** with coverage 7.17.2. Both were already installed in `.venv`, so nothing was installed.
- PySide6 / Qt 6.12.0, torch 2.15.0+cpu, numpy 2.5.4, psutil 7.2.2
- Qt renders offscreen on a virtual 1920 × 1080 screen. The `x11` tests used a private Xvfb started by the harness (Qt `xcb` backend). There is no window manager and no GPU. The display the run was started from was never used.
- The coverage data file is `.temp/coverage/.coverage`. The harness's requirement table is also in `.temp/tests/reports/coverage-20261106-153335.json`.

**Exit code: 1.** pytest code 1 means "tests were collected and run, and some failed". **This is a test result, not an execution or environment error.**
- All 1083 items were collected, with no collection error.
- No test errored in setup or teardown, no timeout fired, and no test was skipped (all `x11` tests ran).
- The harness's safety nets reported nothing left running.
- After the run, no tracked file had changed, and no process, X socket or stray file was left behind.

## 2. Results

| Collected | Passed | Failed | Errors | Skipped | Warnings | Duration |
|---|---|---|---|---|---|---|
| 1083 | **1071** | **12** | **0** | **0** | 1 | 158.78 s |

The 12 failures fall into four groups:
| # | Test | Cases | What failed |
|---|---|---|---|
| 1 | `test_r1_01_tracking.py::test_every_weight_matrix_of_the_network_is_tracked` | `Bilinear`, `RNN`, `LSTM`, `GRU`, `MultiheadAttention`, `TransformerEncoderLayer` | Weights not tracked: `layer.weight` (Bilinear); `weight_ih_l0`, `weight_hh_l0` (RNN, LSTM, GRU); `in_proj_weight` (attention; its `out_proj` *is* tracked); `self_attn.in_proj_weight` (transformer layer; `linear1`, `linear2` and `self_attn.out_proj` are tracked) |
| 2 | `test_r1_g1_network_view.py::test_the_weights_of_every_layer_are_shown` | `RecurrentNet`, `AttentionNet` | Not shown as connections: `rnn.weight_ih_l0`, `rnn.weight_hh_l0`; `attn.in_proj_weight` |
| 3 | `test_r1_g1_network_view.py::test_every_panel_stays_fully_visible_at_small_window_sizes` | 960×680, 1202×680, 1366×720 | `status and chart overlap`; no other panel was involved |
| 4 | `test_r1_g6_network_name.py::test_a_compiled_network_is_still_shown_by_its_own_class` | – | The header shows `OptimizedModule` instead of `TinyNet` for a `torch.compile`-d network |

The one warning is a `FutureWarning` from torch (`torch.jit.script` is deprecated). It is raised by the R1.3 test that checks that a tracked network still scripts.

### Coverage

**92 % statement coverage of `hollywood_training`** (4909 statements, 383 missed). This was measured in the pytest process only: pytest-cov 7 does not follow child processes, so these are not counted:
- the isolated window process. **`gui/window_process.py` shows 0 %**, although the isolated-window tests ran it. `gui/remote.py` (80 %) and `gui/link.py` (76 %) are partly in the same position;
- the demo, README and scenario subprocesses.

The figure therefore understates what the tests exercised, and it is no measure of how well the requirements are verified. Other uncovered code:
- the lazy attribute loaders (`__getattr__`) and `__dir__` of the package `__init__` modules (50–73 %);
- GPU probe code in `monitor/gpu.py` (59 %). This machine has no GPU.

## 3. Requirements

The harness's per-requirement table is at the end of `TEST-REPORT.md` (pass / fail / skip per requirement). The verdicts below weigh *what the tests establish*, not only whether they passed. Two counting rules of that table matter:
- A test can carry several requirement ids.
- Every R1 test is also counted under R2.5.

### Passed
| Req. | Basis |
|---|---|
| R1.2 training management | 71 cases: native PyTorch steps → `TrainingProcess` (decorators, constructor or default steps) → `ht.run`; accepted and refused step results, modes, averaging, hooks, schedulers, data sources; invalid input refused before anything starts; user errors reach the caller; console report and result; window runs (in-process and isolated) that close, linger or start paused; repeated runs; `torch_threads` |
| R1.3 native workflow | 14 cases: a managed run gives exactly the plain PyTorch loop's result (console, in-process window, isolated window); tracking and runs add no attribute, parameter, buffer or hook; the network still pickles, copies, saves / loads and scripts; the step decorators return the user's function |
| R1.5 synchronisation | 26 cases at three levels: the hand-off itself, the real manager, and the real windows (in-process and window process). For every epoch the panels are repainted with that epoch, with the model's real weights, **before** the next epoch's first batch. Boundaries: one epoch, many fast epochs, pause / step / resume, extending a finished run, closing mid-training, a failing step. |
| R1.6 speed | 10 cases: no delay by default; the next epoch starts right after the hand-off (headless, in-process, window process); the window acknowledges as soon as it has drawn; batches never wait. The verdict rests on thresholds one to two orders of magnitude above the hand-off cost: they establish that there is no pacing, not that overhead is minimal. R1.5 says efficiency has no priority. |
| R1.7 commands | 200 cases (including the R2.4 commands): registration and help, parsing, refusals that change nothing, boundaries, custom commands, and every built-in command against a real manager |
| R1.8 no dark magic | 63 cases: a syntax-tree scan of every package module for source retrieval, frame / code introspection, tracing, `exec` / `eval`, import hooks and monkeypatching, plus a check that the scan would notice them. Steps **without any source** train normally while source retrieval is booby-trapped. |
| R1.10 package folder | 4 cases: the build configuration and an offline-built wheel contain exactly `hollywood_training/`; no Python file outside the known folders |
| R1.11 demos | 10 cases: `demos/` directly under the project and not inside the package; every demo defines a network and a dataset, explains its options, and runs on the console and with the window. "Illustrative" was judged by the presence of a real network and dataset. |
| R1.12 README | 5 cases: installation, quick start with the whole workflow; **the quick-start example runs as written**; the relative links exist |
| R1.G5 terminal | 15 cases: part of the window; typed commands control a real in-process run and the window-process run; echo, answers, error colour, history, completion, bounded log, `clear`, any characters. The 3 failures counted under R1.G5 belong to group 3 (status / chart overlap), which does not involve the terminal. |
| R1.G7 blue based | 22 cases measured on rendered pixels: ≥ 80 % of the colourful pixels are blue, in the whole window, every panel, the icon and a screenshot of the real window process; the theme's base colours are blue |
| R2.3 keyboard only | 53 cases: every first-version mouse interaction has a working keyboard path (`scale`, `probe`, `neuron`, `layout`, PgUp / PgDn / Ctrl+Home / Ctrl+End, `copy`, Ctrl+V, `window`, `quit`); every command can be typed; the help explains the keys; whole sessions run by keyboard alone, in-process and in the window process |
| R2.4 replacement commands | 130 cases: the six commands are registered and documented; one validated request per form; bad arguments refused before reaching the window; every documented form and boundary in a real window; the probe line clears by itself; the requests cross the process link to the real window process |
| R2.6 outside the window | 14 cases. Offscreen: no override cursor and no grab; another window of the same application keeps its cursors and receives clicks, moves, wheel, hover / tooltips, drops and context menus, also during a real run and after the window closed. On a real X server: the pointer crosses the boundary freely and looks normal outside; another application gets mouse and, with the focus, keyboard input; no grab; all still true after the session ended. |
| R2.7 keyboard reroute | 34 cases. Offscreen: the platform input path; keys addressed to every widget; key releases, focus changes, Tab / Shift+Tab, every printable character, special keys, reactivation, other windows, widgets added later; a real in-process run and the window process. On a real X server: real keys reach the terminal without a click and wherever the pointer is; command and special keys act; another application keeps its keys; no keyboard grab. "Users stay on the window" was produced by giving it the input focus, as a window manager would. |

### Failed
| Req. | Failure (evidence in §2) | What passes |
|---|---|---|
| R1.1 tracking | The weights of `Bilinear`, `RNN` / `LSTM` / `GRU` and `MultiheadAttention` / `TransformerEncoderLayer` are **not tracked** (group 1; cause in F1) | 40 cases: the declaration, the architecture in execution order, and the real weights of Linear, Conv, ConvTranspose and Embedding layers, every epoch |
| R1.G1 network view | The weights of recurrent and attention layers are **not shown** (group 2; consequence of F1) | 13 R1.G1 cases: columns, connections carrying the real weights (drawn per epoch with the model's real weights), drawing that changes with each epoch, conv / reshape / embedding networks, labels, wide layers |
| R1.G2 status | At window sizes the window allows (960×680, its minimum, plus 1202×680 and 1366×720), **the status panel and the loss chart overlap**, so part of their information can be hidden (group 3; F2) | 26 cases: all nine items with the session's real values, both progress bars, the timer, the boundaries |
| R1.G3 loss chart | Same overlap as R1.G2 (group 3; F2) | 11 cases: one point per epoch on both curves, legend, axis, no validation, NaN / inf gaps, negative and constant losses, a 500-epoch run |
| R1.G6 network name | A `torch.compile`-d network is named `OptimizedModule`, not its class (group 4; F3) | 9 cases: the class name in the header, title and greeting; Sequential, DataParallel, display names, custom titles, the window process, unusual names |

### Partially verified
| Req. | What is verified, and what is not |
|---|---|
| R1.4 native GUI, smooth animations | **Verified:** a native Qt top-level window (in-process and window process); no web engine, web channel, HTTP server, browser module, listening socket or web import (20 cases). **Only by proxies:** "smooth" was measured as the clock's frame rate, continuous repaints and changing frames, offscreen; the isolated window keeps animating during long batches. Perceived smoothness on a real display cannot be established by automated tests. |
| R1.9 modules, architecture | **Verified** (152 cases): subpackages by concern, a documented purpose per module, no module over 1000 lines, no import cycle, a core free of the GUI, a lazy composition root, torch-free frontend data, every module importing on its own. **Not established:** whether the architecture is "clear" is a qualitative judgement. |
| R1.13 wiki | **Verified** (72 cases): separate titled pages, the GitHub-wiki layout, every link and image resolving, every page reachable, an architecture page covering every subpackage, every subpackage, module and public name appearing on a page. **Not established:** that each component is *adequately explained*. The tests check coverage, not explanation quality. See also O2: one page is out of date. |
| R1.G4 usage | **Verified:** CPU and memory from real readings; no GPU rows and no GPU probe when there is no GPU. **Only simulated:** the GPU case ("only when applicable"), through a faked CUDA device, because this machine has no GPU. The 3 failures counted under R1.G4 belong to group 3, which does not involve the usage panel. |
| R2.1 no pointer inside the window | **Verified** (37 cases) for the application's whole window area: blank cursors on the window, its native window and every widget (also those set or added later, in every mode and state). On a real X server the cursor image is transparent at every point of a 16×10 grid, over every panel and splitter handle, at the edges, with a button held, when paused, after mode changes and typing, and when the window appears under a resting pointer. **Not covered:** a window manager's title bar and frame. There was no window manager in the test environment, and by documented design (`gui/pointer_guard.py`, wiki `GUI-Pointer-Guard`) they keep the window manager's pointer and buttons. Whether that frame counts as "inside the window" is a scope decision (F4). |
| R2.2 mouse disabled | **Verified** (112 cases) for the application's window: gestures through the native window on every panel and splitter handle, direct events to every widget, drag and drop, touch, focus by click, a real in-process run, the window process, and real X11 clicks, drags, selection, wheel, middle-click and hovering. Nothing changed and no popup opened, and training and keys still worked afterwards. **Not covered:** the window manager's title-bar buttons and frame, which stay mouse-operable by design (F4). |
| R2.5 R1 unaffected | **Verified:** all 26 cases (12 test functions) of `test_r2_5_r1_unaffected.py` pass. They check how the R2 machinery interacts with R1 features: the API, `run()` options, commands, all panels visible and repainted, resizing and closing, the R1 terminal keys, the console frontend. Every other R1 case passes too, except the 12 of §2. **Not established by tests:** that R2 left those 12 unaffected. They fail on this version whatever R2 did. The git history indicates that they predate R2 (F5): R2 did not change the tracking code, the naming code or any size that decides the layout. This was not confirmed by running the harness against the R1 code. |

### Could not be verified
No requirement is wholly unverifiable. These *aspects* could not be verified in this environment:
- GPU usage on real GPU hardware (R1.G4).
- Perceived smoothness on a real display (R1.4).
- Behaviour under a real window manager or desktop: the window frame (R2.1, R2.2), real window activation (R2.7), compositors, Wayland.
- Other operating systems (macOS, Windows), HiDPI displays.
- Subprocess code coverage (§2).

## 4. Findings

### 4.1 Package defects (tests failed on the package's behaviour; nothing was fixed in this stage)
| ID | Severity | Finding | Evidence |
|---|---|---|---|
| F1 | Medium | **Tracking skips the weight matrices of common layer types (R1.1, R1.G1).** `tracking/layers.py` treats as connection layers only Linear, convolution, transposed-convolution and embedding classes, plus any module whose parameter named `weight` is 2-D (`has_weight_matrix`). `Bilinear` (3-D `weight`), recurrent layers (`weight_ih_l*` / `weight_hh_l*`) and attention (`in_proj_weight`) fall outside this rule. Their weights are neither tracked nor drawn, without any notice. | Groups 1 and 2 in §2 |
| F2 | Medium | **Status panel and loss chart overlap at small window sizes (R1.G2, R1.G3).** The window's minimum is 960×680 (`gui/window.py:163`). Its content needs at least **764 px** of height: margins 8 + 14, header 80, spacing 6, and the right column of status 300 + chart 200 + usage 140 minimum, plus two 8 px handles. At 680 px that is 84 px short, at 720 px 44 px short, and the right column's panels overlap. | Group 3 |
| F3 | Low | **A `torch.compile`-d network is named `OptimizedModule` (R1.G6).** `tracking/inspector.py` `unwrap()` looks only through `DataParallel` / `DistributedDataParallel`, then uses `type(network).__name__`. | Group 4 |
| F4 | Scope question | **The window-manager frame (R2.1, R2.2).** By design, the pointer guard leaves the native title bar and frame to the window manager. On a desktop, the pointer is therefore visible over the title bar, and its buttons and dragging work with the mouse. The specification asks for no pointer "**ANYWHERE** inside the window" and for "**ALL** mouse-based interactions" disabled "within the application". Whether the frame is included needs a decision; it was not testable here. | `gui/pointer_guard.py:23-24`; wiki `GUI-Pointer-Guard.md` lines 85–100; commit `0c51562` |
| F5 | Info | **The R1 failures predate R2 (for R2.5).** `git diff b305b99 a20a247` (the R1 code against R2's last package commit) changes nothing in `tracking/` (F1, F3). It changes no minimum size, margin, spacing, handle width or header height (F2): R2 only moved the splitter shares into the `SPLITS` constant and added `setSizes` calls used by the `layout` command. | git history |

### 4.2 Other observations
- **O1, harness health:** no collection error, setup / teardown error, timeout or skip. The safety nets found nothing left running, and no process or file was left behind. The exit code reflects the 12 test failures only.
- **O2, documentation:** `wiki/Development.md` (lines 15–34) still describes the 11 build-phase test modules (`tests/test_tracking.py` … `tests/test_view_commands.py`). They no longer exist, because the SPEC harness replaced them. The R1.13 tests do not catch this, because the names are written as code, not as links.
- **O3, warnings:** torch's `FutureWarning` for `torch.jit.script` (R1.3 scripting test). The failing GUI tests also show the offscreen platform's informational `This plugin does not support raise()`.
- **O4, environment:** a single run on one machine. Timing-based checks (R1.4, R1.5, R1.6) use generous thresholds, and none of them failed in this run. One run cannot rule out flakiness on a slower or busier machine.

## 5. Recommendations (not done in this stage)
1. F1: treat every parameter with 2 or more dimensions as a tracked weight, so recurrent, bilinear and attention layers are tracked, and draw them in the network view.
2. F2: make the window's minimum height at least the content's minimum (≥ 764 px), or lower the right column's minimum panel heights.
3. F3: also unwrap `torch.compile`'s `OptimizedModule` (`_orig_mod`) before taking the class name.
4. F4: decide whether the window-manager frame is in the scope of R2.1 / R2.2; a frameless window would bring it under the guard.
5. O2: update `wiki/Development.md` to describe the SPEC harness.
6. For a coverage figure that includes the window process, enable coverage's subprocess measurement in a future run.

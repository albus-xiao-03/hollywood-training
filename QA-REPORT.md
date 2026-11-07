# QA report: hollywood-training, R2 (keyboard-driven control), after the QA fixes

| | |
|---|---|
| Date | 2026-11-07 (run 05:57:21–06:00:00 UTC) |
| Branch / commit | `feature/keyboard-driven-control` at `c11fd15` |
| Package code under test | `hollywood_training/` including the 10 fix commits `2888dbb`..`c11fd15` (tracking, `TrainingProcess`, `weights` command, network view, README, wiki) |
| Test harness | `tests/`, unchanged since `eb5a51e`, executed as committed: no test was modified, added or removed |
| Ground truth | `specs/R1.md`, `specs/R2.md` |
| Raw output | [`TEST-REPORT.md`](TEST-REPORT.md): the exact, unchanged pytest output of this run |
| Package code | not modified in this stage |

This report and `TEST-REPORT.md` replace the reports committed in `fcb4359`, which describe a run made before the package fixes. Everything below comes from this run, the specifications, the package code and the git history. Where they apply, I take into account two scope decisions the product owner made when the fixes were planned: the status/chart overlap at small window sizes is accepted, and the window manager's title bar and frame are not "inside the window".

## 1. Execution

**Command** (from the project root):
```
COVERAGE_FILE=.temp/qa_r4/.coverage tests/run_tests.sh --cov=hollywood_training --cov-report=term-missing
```
`run_tests.sh` sources `.temp/env_setup/devenv.sh` (the project's `.venv`, every cache in `.temp/`) and runs `python -m pytest` with the given arguments. `COVERAGE_FILE` only keeps this run's coverage data apart from earlier runs. pytest reads `pyproject.toml`: `testpaths = ["tests"]`, `-ra`, a 120 s timeout per test.

**Environment:**
- Linux aarch64, Python 3.13.5 in `.venv`
- pytest 9.2.0, pluggy 1.7.0, pytest-qt 4.5.0, pytest-timeout 2.4.0
- **pytest-cov 7.1.0** with coverage 7.17.2. Both were already installed in `.venv`, so nothing was installed.
- PySide6 / Qt 6.12.0, torch 2.15.0+cpu
- Qt renders offscreen on a virtual 1920 × 1080 screen. The `x11` tests used a private Xvfb started by the harness (Qt `xcb`). There is no window manager and no GPU. The display the run was started from was never used.
- The harness's requirement table is also in `.temp/tests/reports/coverage-20261107-010000.json`.

**Exit code: 1.** pytest code 1 means "tests were collected and run, and some failed". **This is a test result, not an execution or environment error.**
- All 1083 items were collected, with no collection error.
- No test errored in setup or teardown, no timeout fired, and no test was skipped (all `x11` tests ran).
- The harness's safety nets reported nothing left running.
- After the run, no tracked file had changed, and no process, X socket or stray file was left behind.

## 2. Results

| Collected | Passed | Failed | Errors | Skipped | Warnings | Duration |
|---|---|---|---|---|---|---|
| 1083 | **1080** | **3** | **0** | **0** | 1 | 158.48 s |

**The 3 failures** are the three cases of `test_r1_g1_network_view.py::test_every_panel_stays_fully_visible_at_small_window_sizes` (960×680, 1202×680, 1366×720). Each reports `status and chart overlap`, and no other panel is involved. This test carries the ids R1.G2–R1.G5 and sits in the R1.G1 module, so the harness counts its 3 failures under R1.G1–R1.G5 and, as an R1 test, under R2.5.

**Every other test passes.** That includes the strict tests the fixes addressed:
- `test_every_weight_matrix_of_the_network_is_tracked` (all 6 layer types);
- `test_the_weights_of_every_layer_are_shown` (recurrent and attention networks);
- `test_a_compiled_network_is_still_shown_by_its_own_class`.

The one warning is torch's `FutureWarning` that `torch.jit.script` is deprecated. It is raised by the R1.3 test that checks that a tracked network still scripts.

### Coverage

**92 % statement coverage of `hollywood_training`** (5226 statements, 436 missed). This was measured in the pytest process only: pytest-cov 7 does not follow child processes, so these are not counted:
- the isolated window process (**`gui/window_process.py` 0 %**, although the isolated-window tests ran it; `gui/remote.py` 80 % and `gui/link.py` 76 % are partly the same);
- the demo, README and scenario subprocesses.

The figure understates what the tests exercised. It is no measure of requirement verification.

The code the fixes changed is mostly executed: `tracking/` 73–100 % (`inspector.py` 85 %, `layers.py` 92 %, `snapshot.py` 92 %, `weights.py` 96 %), `network_view.py` 97 %, `builtin.py` 94 %, `process.py` 98 %. Its unexecuted parts are listed in V2 (§4.2). Older uncovered code:
- the lazy attribute loaders (`__getattr__`) and `__dir__` of the package `__init__` modules (50–73 %);
- GPU probe code (`monitor/gpu.py` 59 %). This machine has no GPU.

## 3. Requirements

The harness's per-requirement table is at the end of `TEST-REPORT.md`. The verdicts below weigh *what the tests establish*, not only whether they passed.

### Passed
| Req. | Basis |
|---|---|
| R1.2 training management | 71 cases: native PyTorch steps → `TrainingProcess` (decorators, constructor or default steps) → `ht.run`; accepted and refused step results, modes, averaging, hooks, schedulers, data sources; invalid input refused before anything starts; user errors reach the caller; console report and result; window runs (in-process and isolated) that close, linger or start paused; repeated runs; `torch_threads` |
| R1.3 native workflow | 14 cases: a managed run gives exactly the plain PyTorch loop's result (console, in-process window, isolated window); tracking and runs add no attribute, parameter, buffer or hook; the network still pickles, copies, saves / loads and scripts; the step decorators return the user's function |
| R1.5 synchronisation | 26 cases at three levels: the hand-off itself, the real manager, and the real windows (in-process and window process). Every epoch is repainted with that epoch and the model's real weights **before** the next epoch's first batch. Boundaries: one epoch, many fast epochs, pause / step / resume, extending a finished run, closing mid-training, a failing step. |
| R1.6 speed | 10 cases: no delay by default; the next epoch starts right after the hand-off (headless, in-process, window process); the window acknowledges as soon as it has drawn; batches never wait. The checks use thresholds one to two orders of magnitude above the hand-off cost: they establish that there is no pacing, not that overhead is minimal. R1.5 says efficiency has no priority. |
| R1.7 commands | 200 cases (including the R2.4 commands): registration and help, parsing, refusals that change nothing, boundaries, custom commands, and every built-in command against a real manager. The new per-tensor block of `weights <layer>` is not exercised (V2). |
| R1.8 no dark magic | 63 cases: a syntax-tree scan of every package module (source retrieval, frame / code introspection, tracing, `exec` / `eval`, import hooks, monkeypatching), plus a check that the scan would notice them. Steps **without any source** train normally while source retrieval is booby-trapped. |
| R1.10 package folder | 4 cases: the build configuration and an offline-built wheel contain exactly `hollywood_training/`; no Python file outside the known folders |
| R1.11 demos | 10 cases: `demos/` directly under the project, not in the package; every demo defines a network and a dataset, explains its options, and runs on the console and with the window. "Illustrative" was judged by the presence of a real network and dataset. |
| R1.12 README | 5 cases: installation, quick start with the whole workflow; **the quick-start example runs as written**; the relative links exist |
| R1.G5 terminal | 15 cases: part of the window; typed commands control a real in-process run and the window-process run; echo, answers, error colour, history, completion, bounded log, `clear`, any characters. The 3 failures counted here are the status/chart overlap, which does not involve the terminal. |
| R1.G6 network name | 10 cases: the class name in the header, title and greeting, for the user's class, `nn.Sequential`, `DataParallel`, display names, custom titles, the window process and unusual names; **a `torch.compile`-d network is now shown by its own class** |
| R1.G7 blue based | 22 cases measured on rendered pixels: ≥ 80 % of the colourful pixels are blue (whole window, every panel, icon, a screenshot of the real window process); the theme's base colours are blue |
| R2.1 no pointer inside the window | 37 cases. Blank cursors on the window, its native window and every widget (also those set or added later, in every mode and state). On a real X server the cursor image is transparent at every point of a 16×10 grid, over every panel and splitter handle, at the edges, with a button held, when paused, after mode changes and typing, and when the window appears under a resting pointer. By the product owner's decision the window manager's frame is not part of the window. There was no window manager here. |
| R2.2 mouse disabled | 112 cases: gestures through the native window on every panel and splitter handle, direct events to every widget, drag and drop, touch, focus by click, a real in-process run, the window process, and real X11 clicks, drags, selection, wheel, middle-click and hovering. Nothing changed and no popup opened, and training and keys still worked afterwards. The window-manager frame is out of scope (decision above). |
| R2.3 keyboard only | 53 cases: every first-version mouse interaction has a working keyboard path (`scale`, `probe`, `neuron`, `layout`, PgUp / PgDn / Ctrl+Home / Ctrl+End, `copy`, Ctrl+V, `window`, `quit`); every command can be typed; the help explains the keys; whole sessions run by keyboard alone, in-process and in the window process |
| R2.4 replacement commands | 130 cases: the six commands are registered and documented; one validated request per form; bad arguments refused before reaching the window; every documented form and boundary in a real window; the probe line clears by itself; the requests reach the real window process |
| R2.6 outside the window | 14 cases. Offscreen: no override cursor and no grab; another window of the same application keeps its cursors and receives clicks, moves, wheel, hover / tooltips, drops and context menus, also during a real run and after the window closed. On a real X server: the pointer crosses the boundary freely and looks normal outside; another application gets mouse and (with the focus) keyboard input; no grab, also after the session ended. |
| R2.7 keyboard reroute | 34 cases. Offscreen: the platform input path; keys addressed to every widget; key releases, focus changes, Tab / Shift+Tab, every printable character, special keys, reactivation, other windows, widgets added later; a real in-process run and the window process. On a real X server: keys reach the terminal without a click and wherever the pointer is; command and special keys act; another application keeps its keys; no keyboard grab. "Users stay on the window" was produced by giving it the input focus, as a window manager would. |

### Failed
| Req. | Failure | What passes |
|---|---|---|
| R1.G2 status | At window sizes the window allows (960×680, its minimum, plus 1202×680 and 1366×720), **the status panel and the loss chart overlap**, so part of their information can be hidden (D1). **Accepted by the product owner: no fix planned.** | 26 cases: all nine items with the session's real values, both progress bars, the timer, the boundaries |
| R1.G3 loss chart | The same overlap (D1, accepted) | 11 cases: one point per epoch on both curves, legend, axis, no validation, NaN / inf gaps, negative and constant losses, a 500-epoch run |

### Partially verified
| Req. | What is verified, and what is not |
|---|---|
| R1.1 tracking | **Verified** (46 cases): the declaration; the architecture in execution order; for Linear, Conv, ConvTranspose and Embedding layers, the real weights, their statistics, sampling, read-only copies and per-epoch changes; and that **every weight matrix of `Bilinear`, `RNN`, `LSTM`, `GRU`, `MultiheadAttention` and `TransformerEncoderLayer` now belongs to a tracked layer**. **Not established:** for those six types the tests check *that* a layer is tracked, not *what* is tracked: the values, statistics and per-epoch updates of their weight tensors. Stacked, bidirectional or projected recurrent layers, recurrent cells and attention with separate key/value sizes are not exercised (V1). |
| R1.G1 network view | **Verified** (15 cases): an input column and one column per connection layer, titles, summaries, operations; connections carrying the real weights and biases, redrawn when an epoch arrives (per-epoch equality with the model in R1.5); conv, reshape, embedding networks, labels, wide layers; **the weight matrices of a recurrent and an attention network are now drawn as connections**. **Not established:** that what is drawn for those layers is right, beyond the presence of a connection; the new recurrence arcs and the `ATTENTION` bridge are not checked (V1, V2). The 3 failures counted here are the status/chart overlap, which does not involve the network view. |
| R1.4 native GUI, smooth animations | **Verified:** a native Qt top-level window (in-process and window process); no web engine, web channel, HTTP server, browser module, listening socket or web import (20 cases). **Only by proxies:** "smooth" was measured as the clock's frame rate, continuous repaints and changing frames, offscreen; the isolated window keeps animating during long batches. Perceived smoothness on a real display cannot be established by automated tests. |
| R1.9 modules, architecture | **Verified** (152 cases): subpackages by concern, a documented purpose per module, no module over 1000 lines, no import cycle, a core free of the GUI, a lazy composition root, torch-free frontend data, every module importing on its own. **Not established:** whether the architecture is "clear" is a qualitative judgement. |
| R1.13 wiki | **Verified** (72 cases): separate titled pages, the GitHub-wiki layout, every link and image resolving, every page reachable, an architecture page covering every subpackage, every subpackage, module and public name appearing on a page. **Not established:** that each component is *adequately explained*. The tests check coverage, not explanation quality. |
| R1.G4 usage | **Verified:** CPU and memory from real readings; no GPU rows and no GPU probe without a GPU. **Only simulated:** the GPU case ("only when applicable"), through a faked CUDA device, because this machine has no GPU. The 3 failures counted here are the status/chart overlap, which does not involve the usage panel. |
| R2.5 R1 unaffected | **Verified:** all 26 cases of `test_r2_5_r1_unaffected.py` pass. They check how the R2 machinery interacts with R1 features: the API, `run()` options, commands, all panels visible and repainted, resizing and closing, the R1 terminal keys, the console frontend. Every R1 case passes, except the 3 overlap cases. **Not established by tests:** that R2 left those 3 unaffected, because they fail whatever R2 did. The git history shows that the overlap predates R2: `git diff b305b99 HEAD` changes no minimum size, margin, spacing, handle width or header height in `gui/` (D1). This was not confirmed by running the harness against the R1 code. |

### Could not be verified
No requirement is wholly unverifiable. These *aspects* could not be verified, in this environment or by this harness:
- GPU usage on real GPU hardware (R1.G4).
- Perceived smoothness on a real display (R1.4).
- A real window manager or desktop: real window activation (R2.7), compositors, Wayland.
- Other operating systems (macOS, Windows), HiDPI displays.
- The fixes' behaviour beyond what the harness asserts (V1, V2), because the harness predates the fixes.
- Subprocess code coverage (§2).

## 4. Findings

### 4.1 Package defects
| ID | Severity | Status | Finding |
|---|---|---|---|
| D1 | Medium | Accepted (no fix) | **The status panel and the loss chart overlap at small window sizes (R1.G2, R1.G3).** The window's minimum is 960×680 (`gui/window.py:163`). Its content needs at least **764 px** of height: margins 8 + 14, header 80, spacing 6, and the right column of status 300 + chart 200 + usage 140 minimum, plus two 8 px handles. At 680 px it is 84 px short, at 720 px 44 px short. The sizes are the same in the R1 code (`b305b99`). |

No other package defect was found. The strict tests for tracking (R1.1), drawing (R1.G1) and naming (R1.G6) pass on the fixed code. What they leave open is in §4.2.

### 4.2 Verification gaps: fixed and new behaviour this harness does not check
The harness was written before the fixes. Its two strict tests for the new layer types check presence only, on single small networks: `LSTM(2, 5)`, `MultiheadAttention(8, 2)` and one layer of each R1.1 type.

| ID | Gap |
|---|---|
| V1 | For `Bilinear`, recurrent and attention layers, nothing asserts the **values** of the tracked tensors, their statistics, or that they change with every epoch. That is checked only for Linear / Conv / ConvTranspose / Embedding (R1.1, R1.5). Not exercised at all: stacked (`num_layers > 1`), bidirectional and projected (`proj_size`) recurrent layers, recurrent cells, attention with separate key / value sizes (`q/k/v_proj_weight`), Bilinear's two inputs. |
| V2 | New code that is **not executed** by the harness: <br>• the `tensors` block of `weights <layer>` (`commands/builtin.py:293-299`); <br>• the recurrence arcs drawn back from a projection (`gui/widgets/network_view.py:552-557`); <br>• the traced-order step that places a weight owner that did not run itself (e.g. attention's `out_proj`) after its parent (`tracking/inspector.py:271-286`); <br>• `TrainingProcess.__repr__` naming the network inside a wrapper (`training/process.py:273-275`). <br>New behaviour that is executed but **not asserted**: the recurrence arcs of the network view, and the `ATTENTION` bridge label. |

### 4.3 Other observations
- **Harness health:** no collection error, setup / teardown error, timeout or skip. The safety nets found nothing left running, and no process or file was left behind. The exit code reflects the 3 overlap failures only.
- **Warnings:** torch's `FutureWarning` for `torch.jit.script` (R1.3 scripting test). The failing GUI tests also show the offscreen platform's informational `This plugin does not support raise()`.
- **Documentation:** no document names the removed build-phase test modules any more. `tests/README.md` (the harness's own guide) described the failures of the code before the fixes. It is brought up to date with this run in a separate commit of this stage.
- **Environment:** a single run on one machine. The timing-based checks (R1.4, R1.5, R1.6) use generous thresholds and none failed here. One run cannot rule out flakiness on a slower or busier machine.

## 5. Recommendations (not done in this stage)
1. V1, V2: extend the harness (a harness change, a separate step) with tests for the tracked tensor values and per-epoch updates of recurrent, attention and bilinear layers; stacked, bidirectional and projected recurrent layers; separate q/k/v projections; the recurrence arcs; the `ATTENTION` bridge; `weights <layer>`'s `tensors` block; and `repr(TrainingProcess)`.
2. D1 is accepted. If it is reconsidered: make the window's minimum height at least the content's minimum (≥ 764 px), or lower the right column's minimum panel heights.
3. For a coverage figure that includes the window process, enable coverage's subprocess measurement in a future run.

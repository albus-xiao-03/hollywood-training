# QA report: hollywood-training, R2 (keyboard-driven control)

| | |
|---|---|
| Date | 2026-11-06 |
| Branch / commit under test | `feature/keyboard-driven-control` at `4a4e877` (package code identical to `a20a247`) |
| Ground truth | `specs/R1.md`, `specs/R2.md` |
| Test harness | `tests/` (SPEC harness, executed as committed: no test was modified, added or removed) |
| Raw output | [`TEST-REPORT.md`](TEST-REPORT.md): the exact, unchanged pytest output of run 2 |
| Package code | not modified |

## 1. Execution

**Environment:**
- Linux aarch64, Python 3.13.5 in `.venv`
- pytest 9.2.0, pytest-qt 4.5.0, pytest-timeout 2.4.0, PySide6 6.12.0
- **pytest-cov 7.1.0** and coverage 7.17.2, installed into `.venv` for this QA
- Qt renders offscreen. The `x11` tests used a private Xvfb started by the harness, and Xvfb plus the X client libraries were present, so no `x11` test was skipped.
- All temporary files and the coverage data file are in `.temp/`.

| Run | Command | Exit code | Meaning |
|---|---|---|---|
| 1 | `tests/run_tests.sh --cov=hollywood_training --cov-report=term-missing` | **2** | **Execution error, not a test result.** Collection was interrupted by an error in `tests/test_r2_4_view_commands.py`, so **no test ran**. The output is in the appendix, verbatim. |
| 2 | `tests/run_tests.sh --continue-on-collection-errors --cov=hollywood_training --cov-report=term-missing` | **1** | **Test failures.** Every test that can be collected ran. That output is `TEST-REPORT.md`. |

Run 2 differs from run 1 only by pytest's own `--continue-on-collection-errors` option. It does not touch any test. It lets pytest run the 959 collectable items instead of stopping before any of them. `COVERAGE_FILE` pointed at `.temp/qa_r2/.coverage` in both runs.

### Result of run 2 (129.65 s)

| Collected | Passed | Failed | Errors | Skipped | Warnings | Not collected |
|---|---|---|---|---|---|---|
| 959 items + 1 module error | **932** | **20** | **8** | **0** | 11 | all of `test_r2_4_view_commands.py` (48 test functions) |

What the 20 failures and 8 errors are:
- **12 failures are package findings.** These are the strict tests that `tests/README.md` predicted would fail (§3.1).
- **14 failures and errors come from one harness defect (H2 below).**
  - 7 errors: setup of the `live` fixture in `test_r2_2_mouse_disabled.py`.
  - 7 failures: 2 in R2.1, 1 in R2.6 and 4 in R2.7.
  - All 14 stop with `TypeError: LiveApp.wait() got an unexpected keyword argument 'kind'`, raised in `tests/htspec/live.py:136` before anything is checked. **They are not test results about the package.**
- **1 error** is the collection error of `test_r2_4_view_commands.py` (harness defect H1).
- **1 failure is inconclusive** (finding P5).

### Coverage

The statement coverage of `hollywood_training` is **91 %** (4909 statements, 418 missed), measured in the pytest process.

This measures in-process code only. pytest-cov 7 does not measure subprocesses, so these are not counted:
- the isolated window process: `gui/window_process.py` shows **0 %**, although the isolated-window tests ran it;
- the demo, README and scenario subprocesses.

`gui/remote.py` (80 %) and `gui/link.py` (76 %) are low for the same reason. The figure therefore understates what the tests exercised. It is not a measure of how well the requirements are verified.

## 2. Requirements

The harness's per-requirement table is at the end of `TEST-REPORT.md`. The verdicts below weigh *what the tests establish*, not only whether they passed.

### Passed
| Req. | Basis |
|---|---|
| R1.2 training management | 71 passing cases: native steps → `TrainingProcess` → `ht.run`, refusals, errors, linger / start paused, in-process and isolated windows |
| R1.3 native workflow | A managed run gives exactly the plain loop's result (console, in-process and isolated windows). Tracking and runs add nothing to the network. |
| R1.5 synchronisation | Every epoch is drawn completely, with the real weights, before the next epoch's first batch starts. This was measured headless, in-process and in the window process, including the boundaries. |
| R1.6 speed | No pacing; the next epoch starts right after the hand-off. The verdict rests on thresholds one to two orders of magnitude above the hand-off cost. |
| R1.7 commands | Parsing, validation, every built-in command against a real manager, custom commands |
| R1.8 no dark magic | A static AST scan of every module (and a check that the scan works), plus steps without any source train while source retrieval is booby-trapped |
| R1.10 package folder | Build configuration and a wheel built offline contain exactly the package folder |
| R1.11 demos | `demos/` location; both demos run (`--help`, console, window) |
| R1.12 README | The quick start runs as written; the links resolve |
| R1.G7 blue based | Pixel measurements of the window, every panel, the icon and a window-process screenshot |
| R2.3 keyboard only | All 53 cases pass: every first-version mouse interaction has a working keyboard path, and whole sessions run by keyboard alone (in-process and isolated) |

### Failed
| Req. | Failure |
|---|---|
| R1.1 tracking | The weights of `Bilinear`, `RNN`, `LSTM`, `GRU`, `MultiheadAttention` and `TransformerEncoderLayer` are **not tracked** (e.g. `Bilinear: weights not tracked: ['layer.weight'] (tracked: [])`). The other 40 cases pass: Linear, Conv, ConvTranspose and Embedding are tracked correctly. (P1) |
| R1.G1 network view | For a recurrent and an attention network the real weights are not shown (consequence of P1; P2). At the window sizes it allows, the panels overlap (P4). The architecture and weight drawing for Linear, Conv and Embedding networks passes. |
| R1.G6 network name | A `torch.compile`-d network is shown as `OptimizedModule`, not its class name (P3). The other 8 cases pass. |
| R1.G2 status, R1.G3 loss chart | All their own tests pass, but **the status panel and the loss chart overlap** at 960×680, 1202×680 and 1366×720, which the window allows. Part of both panels can therefore be hidden (P4). |

### Partially verified
| Req. | What is verified, and what is not |
|---|---|
| R1.4 native GUI, smooth animations | Native Qt window and no web technology: verified. "Smooth": verified only as a measurable proxy (clock rate, continuous repaints, differing frames). Perceived smoothness on a real display cannot be judged by automated tests. |
| R1.9 modular architecture | Structural proxies pass: subpackages, documented modules, no god module, no import cycles, a GUI-free core, every module imports on its own. Whether the architecture is "clear" remains a qualitative judgement. |
| R1.13 wiki | Layout, links, reachability and naming of every subpackage, module and public name pass. The tests show that every component is *named* on a page, not that the explanation is adequate. |
| R1.G4 usage | CPU and memory: verified. GPU: only through a faked CUDA device, because this machine has no GPU. |
| R1.G5 terminal | All of the terminal's own cases pass, offscreen and with the window process. The 3 failing cases counted under R1.G5 (and under R1.G4) are the small-window test at its 3 sizes, whose failure (status/chart overlap, P4) does not involve the terminal or the usage panel. |
| R2.1 no pointer | Qt level (in-process and window process) and most real-X11 cursor checks pass. 2 X11 cases did not run their checks because of harness defect H2: cursor after pause / window-mode changes, and after typing. |
| R2.2 mouse disabled | Offscreen gestures, direct events, drag and drop, touch and the real window process pass. **All 7 real-X11 tests errored in fixture setup (H2) and verified nothing.** |
| R2.5 R1 unaffected | The 12 interplay tests in `test_r2_5_r1_unaffected.py` pass. The 12 failing R1 cases are counted here too. They are the P1–P4 findings in tracking, inspector and layout code, which this run cannot attribute to R2 or to R1. |
| R2.6 outside the window | Offscreen (same-application windows), the pointer across the boundary, the normal cursor outside, and another application receiving input all pass. 1 X11 case was cut short by H2, and 1 is inconclusive (P5). |
| R2.7 keyboard reroute | All offscreen and isolated-window cases pass, as do real keys with the pointer anywhere and no keyboard grab. 4 real-X11 cases did not run their checks (H2). |

### Could not be verified
| Req. | Reason |
|---|---|
| R2.4 replacement commands | `test_r2_4_view_commands.py` (48 test functions) **could not be collected** (H1). Its registration, validation, boundary and cross-process tests did not run. The 6 R2.4-tagged passes come from other modules. Indirect evidence: R2.3 shows that `scale`, `probe`, `neuron`, `layout`, `copy` and `window` work from the keyboard. Their well-definedness and boundaries are unverified. |

## 3. Findings

### 3.1 Package (tests failed on the package's behaviour; code not changed, as instructed)
| ID | Severity | Finding |
|---|---|---|
| P1 | Medium | **R1.1:** The tracker only follows modules with a 2-D `weight` (plus convolutions and embeddings). The weight matrices of `Bilinear`, recurrent layers (`RNN`, `LSTM`, `GRU`) and attention / transformer layers are silently untracked. |
| P2 | Medium | **GUI 1:** Consequently, the network view does not show the real weights of recurrent or attention networks. |
| P3 | Low | **GUI 6:** The inspector unwraps only `DataParallel` and `DistributedDataParallel`. A `torch.compile`-d network is named `OptimizedModule` instead of the user's class. |
| P4 | Medium | **GUI layout (G1–G3):** At 960×680 (the window's minimum), 1202×680 and 1366×720, **the status panel and the loss chart overlap**. The right column's minimum heights do not fit the window's minimum height. A 1366×768 screen opens the window at 680 px. |
| P5 | Inconclusive | **R2.6:** `test_outside_stays_normal_after_the_session_ended` found a blank cursor 0.3 s after `quit` at the centre of the closed window. The previous test had already been cut short by H2. Either the window was not yet gone, or this is a test-timing artefact; this run cannot tell which. |

### 3.2 Test harness (defects in the tests; reported, not fixed, as instructed)
| ID | Severity | Finding |
|---|---|---|
| H1 | High | `tests/test_r2_4_view_commands.py`: `test_each_form_hands_the_window_one_validated_request` uses `request` as a `@pytest.mark.parametrize` argument, which pytest reserves. Collecting the module fails, so all of R2.4's 48 test functions are lost, and a plain `tests/run_tests.sh` stops with exit code 2 before running anything. |
| H2 | High | `tests/htspec/live.py:136`: `LiveApp.enter()` calls `self.wait(..., kind=...)`, but `LiveApp.wait()` has no `kind` parameter. The intended call is to the probe log's `wait`. Every real-X11 test that enters a command fails before its check (7 errors, 7 failures), so large parts of the real-X11 verification of R2.1, R2.2, R2.6 and R2.7 never ran. |
| H3 | Info | 10 `DeprecationWarning`s: the `QDragEnterEvent` / `QDragMoveEvent` constructors used in `test_r2_2` / `test_r2_6` are deprecated in PySide6 6.12. They are harmless for now. |
| H4 | Info | The harness had been validated only statically before this run. H1 and H2 are runtime-only defects that this first execution exposed. |

### 3.3 Other observations
- **Environment:** no execution or environment errors other than H1. No x11 test was skipped, no timeout fired, and no process or file was left behind (checked after the run).
- **Warnings:** 1 `FutureWarning` from torch (`torch.jit.script` is deprecated), triggered by an R1.3 test.
- **Documentation:** `wiki/Development.md` still describes the build-phase test modules that the harness replaced. They are named only in code spans, so R1.13 is not affected.

## 4. Recommendations (not done in this stage)
1. Fix H1 (rename the parameter) and H2 (`self.log.wait` in `LiveApp.enter`), then rerun R2.1, R2.2, R2.4, R2.6 and R2.7. Only after that can R2.4 be verified, and R2.1, R2.2, R2.6 and R2.7 be verified fully on real X11.
2. P1/P2: track every weight tensor of a layer (e.g. all parameters with 2 or more dimensions), and show them in the network view.
3. P4: make the right column's minimum heights fit the window's minimum height, or raise the window's minimum.
4. P3: unwrap `OptimizedModule` (`_orig_mod`) when naming the network.
5. Measure the window process's coverage too (coverage's subprocess support), for a coverage figure that reflects the isolated GUI.

## Appendix: run 1 output (verbatim, exit code 2)

```
============================= test session starts ==============================
platform linux -- Python 3.13.5, pytest-9.2.0, pluggy-1.7.0
cachedir: .temp/pytest-cache
PySide6 6.12.0 -- Qt runtime 6.12.0 -- Qt compiled 6.12.0
rootdir: /home/agent/workspace/hollywood-training
configfile: pyproject.toml
testpaths: tests
plugins: qt-4.5.0, timeout-2.4.0, cov-7.1.0
timeout: 120.0s
timeout method: signal
timeout func_only: False
collected 959 items / 1 error

==================================== ERRORS ====================================
______________ ERROR collecting tests/test_r2_4_view_commands.py _______________
tests/test_r2_4_view_commands.py::test_each_form_hands_the_window_one_validated_request: 'request' is a reserved name and cannot be used in @pytest.mark.parametrize
=========================== short test summary info ============================
ERROR tests/test_r2_4_view_commands.py - Failed: tests/test_r2_4_view_command...
!!!!!!!!!!!!!!!!!!!! Interrupted: 1 error during collection !!!!!!!!!!!!!!!!!!!!
=============================== 1 error in 2.23s ===============================
```

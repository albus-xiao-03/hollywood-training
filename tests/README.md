# Test harness

Two layers of tests live here:

| Location | What it is |
|---|---|
| `tests/test_*.py` | the regression suite written with the package (99 tests): unit tests of every module |
| `tests/spec/` | the **SPEC harness**: tests derived from `SPEC.md`, one module per requirement, plus failure scenarios and end-to-end runs of the real window process |

Every SPEC-harness test carries `@pytest.mark.spec("R5", "G1", ...)`. The IDs are the
numbered requirements (`R1`–`R13`) and GUI layout items (`G1`–`G7`) of `SPEC.md`; their
text is in [`spec/requirements.py`](spec/requirements.py). After each run a
**SPEC requirement coverage** table (passed / failed / skipped per requirement) and the list
of checks that need a human are printed at the end of the pytest output.

## Running

```bash
tests/run_tests.sh                               # everything; benchmarks are opt-in
tests/run_tests.sh tests/spec                    # only the SPEC harness
tests/run_tests.sh -m "not e2e and not slow"     # fast: no extra processes
tests/run_tests.sh -m "not timing"               # skip wall-clock bounds (loaded machines)
HT_RUN_BENCHMARKS=1 tests/run_tests.sh -m benchmark
QT_QPA_PLATFORM=xcb xvfb-run -a tests/run_tests.sh   # real X11 windows instead of offscreen
```

`run_tests.sh` uses the project's `.venv` and keeps every temporary and cache file inside
`.temp/` (CONSTRICTION.md): `TMPDIR`, pytest's `tmp_path` base and cache, the XDG cache,
no `__pycache__` folders. `tests/conftest.py` applies the same redirections when pytest is
started some other way, except pytest's own cache folder (pass `-o cache_dir=...`).
GUI tests run headless with `QT_QPA_PLATFORM=offscreen` unless another platform is set.

### Markers

| Marker | Meaning |
|---|---|
| `spec(*ids)` | SPEC requirements verified by the test (traceability) |
| `gui` | needs Qt widgets (works headless) |
| `e2e` | starts real processes: the GUI window process, the demos, the README code |
| `slow` | takes more than a few seconds |
| `timing` | asserts wall-clock bounds; may fail on a heavily loaded machine |
| `benchmark` | performance comparison; skipped unless `HT_RUN_BENCHMARKS=1` |
| `gpu` | needs CUDA or Apple MPS; skipped otherwise |

## Layout of `tests/spec/`

| Module | Verifies |
|---|---|
| `test_r01_tracking.py` | R1: `@track` on classes and instances, automatic tracking by `run()`, every layer type, execution order, lazy / reused / wrapped / compiled networks, real weights and biases per epoch, sampling of wide layers, nothing silently dropped |
| `test_r02_training_management.py` | R2: native steps on decorators or constructor arguments, default steps, per-epoch management, loss and metric averaging, schedulers, train / eval modes, data sources, every invalid definition |
| `test_r03_native_workflow.py` | R3: `@track` returns the class unchanged, pickling / copying / saving, numerically identical training to a plain PyTorch loop (console and window), no leftover hooks or settings, imports change nothing in PyTorch |
| `test_r04_native_gui.py` | R4: a Qt main window, no web technology, the default window process, frame rate and continuous animation |
| `test_r05_synchronization.py` | R5: the blocking hand-off, ordering, pickling for the window process, "next epoch only after the frontend finished drawing", slow and vanished frontends, the in-process window, the window process's draw-then-acknowledge protocol |
| `test_r06_speed.py` | R6: no default delay, training resumes right after each hand-off, no per-epoch or per-batch floor, `delay`, thread policy benchmark |
| `test_r07_commands.py` | R7: every documented command and alias, help, argument validation, effects, session lifecycle, custom commands |
| `test_r07_runtime_control.py` | R7 / G5: the commands against a live session (pause at a batch boundary, step, stop, lr between batches, epochs, save, custom actions on the training thread, concurrency), typed into the window |
| `test_r08_no_dark_magic.py` | R8: static scans (no introspection modules, frames, code objects, eval/exec, monkey-patching, global hooks) and behaviour (steps without source, source access forbidden, no tracer, user functions untouched) |
| `test_r09_architecture.py` | R9: subpackages, module docstrings, no import cycles, downward-only dependencies, GUI-free core, torch-free window process, public API |
| `test_r10_r11_layout_and_demos.py` | R10 / R11: package and demo locations, packaging metadata, demos use the public API and run in the console and the window |
| `test_r12_readme.py` | R12: README sections, the quick start runs (in-process and verbatim with the window), every option and command documented, links |
| `test_r13_wiki.py` | R13: home and architecture pages, a page per subpackage, every module and export explained, GitHub-wiki links, no orphan pages |
| `test_g1_network_view.py` … `test_g7_blue_theme.py` | G1–G7: what each panel draws (recorded paint calls and pixel analysis), kept current per epoch |
| `test_failures.py` | errors in every kind of user code, Ctrl+C, closed windows, invalid options, no display, broken commands, huge layers, non-finite losses, repeated runs |
| `test_window_process_e2e.py` | the default run end to end, observed from inside the real window process |
| `test_traceability.py` | every test names known requirements and every requirement is covered |

Support files: `helpers.py` (networks, processes, frontend stand-ins, a scripted GUI user,
paint recorders, image analysis), `conftest.py` (fixtures, coverage summary) and
`child_hooks/` (a probe loaded into the window process through `PYTHONPATH` by the
end-to-end tests only; it records what the window received, drew and acknowledged, and can
type commands into the real terminal).

## How expectations were derived

Expected behaviour comes from `SPEC.md`. Where the SPEC leaves details open (command names,
argument syntax, colours, the torch-free window process), the tests use the contract the
project documents in `README.md` and `wiki/`, and say so in their docstrings. Assertions
were not relaxed to match the implementation. Tests that encode a SPEC expectation the QA
report (`.temp/qa/QA_REPORT.md`) found unmet:

| QA finding | Tests |
|---|---|
| M1: panels overlap at the minimum window size | `test_g2_status_panel.py::test_no_panel_covers_another_at_any_allowed_window_size[minimum]` |
| M3: the default thread cap slows CPU-heavy training | `test_r06_speed.py::test_the_default_thread_policy_does_not_slow_down_cpu_heavy_training` (benchmark, opt-in) |
| M4: `torch.compile` models show `OptimizedModule` | `test_r01_tracking.py::test_compiled_models_are_tracked_as_the_user_network`, `test_g6_network_name.py::test_a_compiled_network_is_shown_with_its_own_class_name` |
| L1: parameters of non-leaf modules are silently untracked | `test_r01_tracking.py::test_no_layer_weights_are_silently_dropped[transformer]` |

## What automated tests cannot fully verify

| Requirement | Limitation |
|---|---|
| R4 | Perceived smoothness and the "Hollywood" look need a person at a real display; the tests bound the frame rate and check that frames change. |
| R6 | Absolute speed depends on the machine. The thread-policy comparison is an opt-in benchmark; GPU training speed is not measured. |
| R8 | Static scans cover known "dark magic" techniques; new code still deserves a reviewer's eye. |
| R9 | "Clear architecture" is partly judgement; the tests check boundaries, cycles and the documented layering. |
| R12, R13 | The tests check that the quick start runs and that every module and export is documented with working links, not how good the explanations are. Publishing to a real GitHub wiki is manual. |
| G4 | GPU rows are checked with synthetic readings; real CUDA / MPS readings need such a device (`gpu` tests skip otherwise). |
| G7 | Colour checks are numeric hue bands; the overall impression is manual. |
| all GUI items | Runs are headless (`offscreen`) or on Xvfb. Windows, macOS, Wayland, desktop compositors, HiDPI and multiple monitors need manual runs. |

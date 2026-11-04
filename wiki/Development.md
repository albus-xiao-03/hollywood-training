# Development

## Setup

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install torch --index-url https://download.pytorch.org/whl/cpu   # or your CUDA build
pip install -e ".[dev]"
```

## Tests

```bash
pytest                                  # everything (~10 s)
pytest tests/test_tracking.py -q        # one area
xvfb-run -a env QT_QPA_PLATFORM=xcb pytest   # GUI tests on a real X11 server
```

`tests/conftest.py` sets `QT_QPA_PLATFORM=offscreen` unless a platform is already
chosen, so the GUI tests run headless and in CI.

| File | Covers |
|---|---|
| `tests/test_tracking.py` | decorator, classification, architecture building, forward trace, lazy modules, snapshots, connection matrices |
| `tests/test_sync.py` | blocking hand-off, `close()`, control requests, session state |
| `tests/test_training.py` | `TrainingProcess` validation, step output, descriptions, the manager loop: event stream, **training blocked while the frontend handles an epoch**, eval / no-grad validation, pause, step, stop, actions, linger, errors, schedulers, architecture refinement, progress throttling |
| `tests/test_commands.py` | parsing, conversion, errors, every built-in command, custom commands |
| `tests/test_monitor_console.py` | usage sampling, formatting, `run(gui=False)`, `TrainingFailedError` |
| `tests/test_gui.py` | widgets, the main window, terminal input and completion, screenshots, the **in-process frontend drawing every epoch before acknowledging**, commands driving a run, the **window process end to end**, IPC commands and screenshots, start-up failure |
| `tests/test_demos.py` | both demos, console and GUI, as subprocesses |

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

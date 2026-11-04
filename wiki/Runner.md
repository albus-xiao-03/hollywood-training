# Runner

`hollywood_training/runner.py` · part of [Architecture](Architecture)

`run()` is the management method (SPEC 2). It hands a [`TrainingProcess`](Training-Process)
to a [`TrainingManager`](Training-Manager) and a frontend, and returns when the session
is over. The frontend is the native window by default. `run()` creates and connects all
the session objects, so a user script needs only this one call.

## Responsibilities

* Check the process and the options before anything starts.
* Create `TrainingControl`, `SyncChannel`, `NetworkTracker` and the command registry.
* Choose the frontend: window process, in-process window, or console.
* In GUI modes, cap PyTorch's intra-op CPU threads for as long as the window is open.
* Turn a `FAILED` session into `TrainingFailedError`.

## API

```python
def run(
    process: TrainingProcess,
    *,
    gui: bool = True,
    isolate_gui: bool = True,
    start_paused: bool = False,
    close_on_finish: bool = False,
    max_neurons: int = DEFAULT_NEURON_LIMIT,
    title: str | None = None,
    log_every: int | None = None,
    torch_threads: int | str | None = "auto",
) -> TrainingResult
```

| Parameter | Default | Applies to | Meaning |
|---|---|---|---|
| `process` | (required) | all | the wrapped process; any other type raises `TypeError` |
| `gui` | `True` | all | `False` prints to the console instead, for machines without a display |
| `isolate_gui` | `True` | GUI | `True` runs the window in its own process. `False` keeps it in this process, which is needed to embed it in an existing Qt application or where a second process cannot be started |
| `start_paused` | `False` | GUI | open paused, so the initial network can be inspected; `resume` or `step` starts training. With `gui=False` it raises `ValueError` |
| `close_on_finish` | `False` | GUI | close the window once training ends. Otherwise the window stays open and the manager lingers (`linger=not close_on_finish`), so `epochs <n>` can continue |
| `max_neurons` | `DEFAULT_NEURON_LIMIT` (20) | all | neurons drawn per layer column; wider layers are sampled evenly. Passed to `NetworkTracker`, which raises `ValueError` below 2 |
| `title` | `None` | GUI | window title; by default `Hollywood Training — <network class>` |
| `log_every` | `None` | console | print every n-th epoch; by default about 25 lines per run. Must be `None` or a positive `int` (checked in every mode) |
| `torch_threads` | `"auto"` | GUI | intra-op threads while the window is open: `"auto"`, `None` (keep the current setting) or a positive `int` (checked in every mode). Restored afterwards |

**Returns** the [`TrainingResult`](Training-Results), with state `FINISHED` or `STOPPED`.

**Raises**

* `TypeError`: `process` is not a `TrainingProcess`.
* `ProcessError`: from `process.resolve_steps()`, before anything starts.
* `ValueError`: an invalid `torch_threads` or `log_every`, checked in every mode before
  `resolve_steps()` (a `bool` is rejected for both); `start_paused` with `gui=False`;
  `max_neurons` < 2; or a custom command name that is already taken (GUI modes).
* `RuntimeError` from the GUI frontends: no display found (Linux and BSD, with
  `DISPLAY`, `WAYLAND_DISPLAY` and `QT_QPA_PLATFORM` all unset), the window process
  failed to start or connect, or the in-process window was not started on the main
  thread.
* `TrainingFailedError`: the session ended `FAILED`. In GUI modes this is raised after
  the window has closed.
* `KeyboardInterrupt`: Ctrl+C ends the session in every mode, and `run()` re-raises the
  interrupt instead of returning.

```python
class TrainingFailedError(RuntimeError):
    def __init__(self, message: str, result: TrainingResult) -> None
```

Raised by `run()` when the session ended `FAILED`: an exception in a step, a hook, the
scheduler, or the manager's checks (see [Training Manager](Training-Manager)). The
message is `training failed: <Type>: <message>`. The original exception is chained as
`__cause__`, and `result` holds the `FAILED` `TrainingResult` with everything that had
completed.

```python
def cpu_cores() -> int
```

The CPU cores this process may run on: the size of its CPU affinity mask
(`os.sched_getaffinity(0)`, which reflects `taskset` or cpusets), or `os.cpu_count()`
where that is not available (macOS, Windows). Always at least 1. CPU quotas such as
`docker --cpus` are not visible to it.

## How it works

```
run(process, ...)
├─ checks: TrainingProcess? → _check_options(torch_threads, log_every) → resolve_steps()
│          → start_paused needs gui
├─ TrainingControl(process.epochs, start_paused) · SyncChannel() · NetworkTracker(model, max_neurons)
├─ gui and isolate_gui → RemoteFrontend(...)  + TrainingManager(linger=not close_on_finish)
│                         with torch thread cap: RemoteFrontend.run(manager)
├─ gui                 → QtFrontend(...)      + TrainingManager(linger=..., pump=frontend.pump)
│                         with torch thread cap: QtFrontend.run(manager)
├─ else                → ConsoleFrontend(channel, log_every) + TrainingManager(...); manager.run()
└─ FAILED with an error → raise TrainingFailedError(...) from error; else return result
```

| Mode | Call | Frontend | Pump | Linger | `run()` returns |
|---|---|---|---|---|---|
| window process (default) | `run(p)` | [GUI Remote Frontend](GUI-Remote-Frontend) | `WaitingPump` (default) | unless `close_on_finish` | once the window has closed |
| in-process window | `run(p, isolate_gui=False)` | [GUI In-Process Frontend](GUI-In-Process-Frontend) | `QtPump` | unless `close_on_finish` | once the window has closed |
| console | `run(p, gui=False)` | [Console Frontend](Console-Frontend) | `WaitingPump` (default) | never | when training ends |

The GUI frontends get `create_registry(process.commands)`: the
[built-in commands](Commands-Builtin) plus the process's custom ones. The GUI modules
are imported only in their branch. Closing the window stops training. Both GUI
frontends then call `control.stop()`, and the remote frontend also closes the channel.

**Torch thread cap.** By default PyTorch runs one spin-waiting intra-op worker per core.
While a window draws on the same machine, any core it takes stalls that pool's
barriers, and small-batch training can become several times slower. In GUI modes the
private context manager `_torch_threads(torch_threads)` therefore wraps the frontend's
`run()`:

| `torch_threads` | Thread count while the window is open |
|---|---|
| `"auto"` | `min(current, max(1, cpu_cores() // 2))`, never more than the current count |
| `None` | unchanged |
| `int` ≥ 1 | that count |
| anything else | `ValueError`, already raised by `_check_options` before anything starts |

The previous `torch.get_num_threads()` value is restored on exit, including after an
error. Nothing is changed if the target equals the current count. For the best speed on
Linux, the docstring recommends starting Python with `OMP_NUM_THREADS` set (e.g. to half
the cores) instead. Measurements are on the [Performance](Performance) page.

## Used by / uses

* Uses: [Training Process](Training-Process), [Training Manager](Training-Manager),
  [Training Results](Training-Results), [Sync Control](Sync-Control),
  [Sync Channel](Sync-Channel), [Tracking Network Tracker](Tracking-Network-Tracker),
  [Commands Builtin](Commands-Builtin) (`create_registry`), and the three frontends.
* Used by: user scripts, as `hollywood_training.run` (loaded on first access, see
  [API Reference](API-Reference)), and the [Demos](Demos).

## Example

```python
import hollywood_training as ht

if __name__ == "__main__":
    result = ht.run(process)                          # window in its own process
    # ht.run(process, start_paused=True)              # inspect first, then type `resume`
    # ht.run(process, close_on_finish=True)           # close the window when training ends
    # ht.run(process, isolate_gui=False)              # window inside this process
    # ht.run(process, gui=False, log_every=10)        # console, every 10th epoch
    # ht.run(process, torch_threads=None)             # keep PyTorch's thread setting
    print(result.state.value, result.epochs_completed)
```

```python
try:
    ht.run(process, gui=False)
except ht.TrainingFailedError as exc:
    print(exc)                          # training failed: RuntimeError: bad batch
    print(repr(exc.__cause__))          # RuntimeError('bad batch')
    print(exc.result.epochs_completed)
```

## Notes

* `title`, `close_on_finish`, `isolate_gui` and `torch_threads` have no effect with
  `gui=False`, and `log_every` has no effect in GUI modes. `torch_threads` and
  `log_every` are still validated in every mode.
* `start_paused` is rejected without a GUI because the console frontend has no terminal
  to resume from.
* The thread cap is already in force when the session starts, so the status panel shows
  the capped count, e.g. `cpu (5 threads)`.

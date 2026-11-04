# Tracking Network Tracker

`hollywood_training/tracking/tracker.py` · part of [Tracking](Tracking)

`NetworkTracker` follows one model instance through training. It owns the model's
[NetworkArchitecture](Tracking-Architecture-Model), built statically at construction and
refined once by a forward trace. On request it produces a
[WeightSnapshot](Tracking-Weight-Data), remembering the last baseline so each snapshot
carries the per-connection change since then. It is the only tracking object the
[Training Manager](Training-Manager) talks to.

## Responsibilities

* Resolve the display options: explicit, else declared with `@track` on the model or on
  the network inside a wrapper, else defaults.
* Hold the current architecture and run the one-time forward trace.
* Take snapshots with the configured neuron limit, and keep the baseline used for
  `delta`.

## API

```python
class NetworkTracker:
    def __init__(
        self,
        model: nn.Module,
        *,
        max_neurons: int = DEFAULT_NEURON_LIMIT,
        options: TrackOptions | None = None,
    ) -> None
```

* `model`: the network to follow. Wrappers such as `nn.DataParallel` are looked through
  when the architecture is built.
* `max_neurons`: neurons kept per column in snapshots. Wider layers are sampled. It is
  stored as `int(max_neurons)`.
* `options`: display options. When `None`, the tracker uses `track_options(model) or
  track_options(unwrap(model)) or TrackOptions()`. A wrapped model therefore keeps the
  `@track` options of the network inside. Options declared on the wrapper itself take
  precedence.
* **Raises** `TypeError` when `model` is not an `nn.Module`, and `ValueError` when
  `max_neurons < 2`.
* The static architecture is built in the constructor.

| Property | Type | Value |
|---|---|---|
| `model` | `nn.Module` | the object passed in (not unwrapped) |
| `options` | `TrackOptions` | the resolved display options |
| `max_neurons` | `int` | neuron limit per side |
| `architecture` | `NetworkArchitecture` | the current architecture (live objects) |
| `tracing` | `bool` | `True` between `begin_trace()` and `end_trace()` |

```python
def refresh(self) -> NetworkArchitecture
def begin_trace(self) -> None
def end_trace(self) -> bool
def snapshot(self, epoch: int, *, baseline: bool = True) -> WeightSnapshot
```

* `refresh()`: rebuilds the architecture from the static order (e.g. after modules were
  replaced) and returns it. The result has `traced=False`. The baseline is kept.
* `begin_trace()`: creates and starts a [ForwardTrace](Tracking-Inspector), unless one is
  already running.
* `end_trace()`: stops the trace. It returns `False` and keeps the current architecture
  when no trace is running, when nothing was recorded, or when the pass did not complete
  (the forward raised, or the model's own `__call__` never ran). Otherwise it adopts the
  architecture built from the recorded order and returns `True` when its
  `architecture_signature` differs. The signature includes `traced`, so the first
  complete trace after construction or `refresh()` always returns `True`.
* `snapshot(epoch, baseline=True)`: `capture(architecture, epoch=epoch,
  limit=max_neurons, previous=<baseline>)`. `epoch` is the number of completed epochs
  the snapshot belongs to. With `baseline=True`, the snapshot becomes the reference for
  the next `delta`. Pass `False` for intermediate looks, e.g. while paused. Their `delta`
  is still measured against the last baseline, and the baseline stays.

```python
def inspect_network(model: nn.Module) -> NetworkArchitecture
```

Convenience: `NetworkTracker(model).architecture`, the static architecture with the
options declared by `@track`. Raises like the constructor.

## How it works

The tracker keeps three pieces of state: the current `NetworkArchitecture`, the running
`ForwardTrace` (or `None`), and the baseline `WeightSnapshot` (or `None`). Under `run()`,
the tracker is created as `NetworkTracker(process.model,
max_neurons=max_neurons)` and handed to the `TrainingManager`. When no tracker is given,
the manager creates `NetworkTracker(process.model)` itself. It then calls:

| When | Call | Goes to |
|---|---|---|
| session start | `architecture.describe()`, `snapshot(0)` | `SessionStarted` (`SessionInfo.architecture`, `weights`), the first baseline |
| first training batch only | `begin_trace()` before `train_step(batch)`, `end_trace()` in `finally` | the "architecture changed" flag |
| every completed epoch | `snapshot(epoch)` (baseline), plus `architecture.describe()` once when the flag is set | `EpochCompleted.weights`, `EpochCompleted.architecture` |
| a pause takes effect | `snapshot(epoch, baseline=False)` | `TrainingPaused.weights` |
| session end | `snapshot(epoch, baseline=False)`, with any exception turned into `None` | `SessionFinished.weights`, `TrainingResult.final_weights` |

Because the signature includes `traced`, a complete trace always sets the flag, and the
first `EpochCompleted` carries the refined `ArchitectureInfo` (`traced=True`).
`SessionInfo.neuron_limit` is `tracker.max_neurons`. The
[Network View](GUI-Network-View) uses it for layers that have no weights yet.

## Used by / uses

* Uses [Tracking Decorator](Tracking-Decorator) (`TrackOptions`, `track_options`),
  [Tracking Inspector](Tracking-Inspector) (`build_architecture`, `ForwardTrace`,
  `architecture_signature`) and [Tracking Snapshots](Tracking-Snapshots) (`capture`).
* Created by [Runner](Runner) (`run()`) and driven by [Training Manager](Training-Manager)
  (`TrainingManager(tracker=...)`, `manager.tracker`).
* Exported lazily as `hollywood_training.tracking.NetworkTracker` / `inspect_network` and
  at top level as `ht.NetworkTracker` / `ht.inspect_network`.

## Example

```python
import torch
from torch import nn
import hollywood_training as ht

model = nn.Sequential(nn.LazyLinear(5), nn.ReLU(), nn.Linear(5, 2))
tracker = ht.NetworkTracker(model, max_neurons=8)
tracker.snapshot(0).names()             # ('2',): the lazy layer has no weights yet

tracker.begin_trace()
model(torch.randn(3, 7))                # the first forward pass is recorded
tracker.end_trace()                     # True: the pass completed, shapes became known
tracker.architecture.layers[0].summary  # 'Linear 7→5'
tracker.snapshot(1).names()             # ('0', '2')
```

## Notes

* **Thread:** use the tracker from the thread that trains the model, between
  optimisation steps. Nothing is locked.
* The manager traces only the first training batch, even when that pass is not adopted,
  and it never calls `refresh()`, which would discard the traced order. A step that calls
  only submodules (e.g. `model.encoder(x)`) never completes a trace, so the static
  architecture stays in use.

# Tracking Inspector

`hollywood_training/tracking/inspector.py` · part of [Tracking](Tracking)

The inspector groups the leaf modules of a network into connection layers and the
operations applied to their outputs, producing a
[NetworkArchitecture](Tracking-Architecture-Model). The module order comes from one of
two official PyTorch APIs. The **static** order is the registration order of
`named_modules()`, available at once. The **traced** order is the order in which leaf
modules actually ran in one forward pass, recorded by `ForwardTrace` with
`register_forward_hook`.

## Responsibilities

* Look through `nn.DataParallel` / `DistributedDataParallel` wrappers, list leaf modules
  and count initialised parameters.
* Group a module sequence into `TrackedLayer`s, `LayerOperation`s and unsupported names.
* Record the execution order of the first complete forward pass.
* Fingerprint an architecture, so a rebuilt one can be compared with the previous one.

## API

```python
ModuleSequence = Sequence[tuple[str, nn.Module]]

def unwrap(model: nn.Module) -> nn.Module
def leaf_modules(model: nn.Module) -> list[tuple[str, nn.Module]]
def count_parameters(module: nn.Module, *, trainable_only: bool = False) -> int
```

* `unwrap`: follows `.module` while the model is an `nn.DataParallel` or
  `nn.parallel.DistributedDataParallel`, so nested wrappers are handled too.
* `leaf_modules`: `(qualified_name, module)` for every module of `unwrap(model)` without
  children, in `named_modules()` order. The root (name `""`) is listed only when it has no
  children. A module registered under two names is listed once, because
  `named_modules()` removes duplicates.
* `count_parameters`: sum of `numel()` over `module.parameters()`, skipping uninitialised
  lazy parameters. With `trainable_only=True`, only parameters with `requires_grad`
  count.

```python
def build_architecture(
    model: nn.Module,
    options: TrackOptions | None = None,
    order: ModuleSequence | None = None,
) -> NetworkArchitecture
```

* `model`: the network. Wrappers are looked through. `options`: display options declared
  with `@track`, where `None` means `TrackOptions()`.
* `order`: explicit `(name, module)` execution order, e.g. from `ForwardTrace.stop()`.
  Defaults to `leaf_modules(model)`. The result has `traced=True` exactly when `order` is
  given.

```python
def architecture_signature(architecture: NetworkArchitecture) -> tuple
```

A hashable fingerprint. It holds the input operation summaries, then for each layer its
`(name, summary, in_features, out_features, operation summaries)`, then `unsupported`,
`total_parameters` and `traced`. Display options, roles and `trainable_parameters` are not
part of it. Because `traced` is included, a traced build never equals a static one.

```python
class ForwardTrace:
    def __init__(self, model: nn.Module) -> None
    complete: bool                      # instance attribute
    @property
    def active(self) -> bool
    def start(self) -> None
    def stop(self) -> list[tuple[str, nn.Module]]
```

* `start()`: does nothing while hooks are registered. Otherwise it clears earlier calls,
  sets `complete = False` and registers a recording forward hook on every leaf module of
  the unwrapped model, plus one hook on the root module.
* `stop()`: removes all hooks and returns a copy of the recorded `(name, module)` calls,
  even when no pass completed.
* `active` is `True` while hooks are registered. `complete` is `True` once the root
  module's forward has returned.

## How it works

`build_architecture` walks the sequence once, keeping a `current` connection layer:

1. `label = name or type(module).__name__`. Pass-through modules (`nn.Identity`) are
   skipped.
2. `CONNECTION`: the first time a module object is seen (compared by `id`), it becomes a
   new `TrackedLayer`, with `summarize()`, `connection_shape()` and
   `count_parameters(module)`, and is made `current`. When the same object comes again,
   its existing layer becomes `current` again, so it is drawn once.
3. Operation roles: a `LayerOperation` is appended to `current.operations`, or to
   `input_operations` while there is no `current` layer.
4. `OTHER`: the label is appended to `unsupported` unless it is already there. `current`
   is unchanged.

`ForwardTrace` registers the leaf hooks before the root hook, so a model that is itself a
leaf still records its own call. Each leaf hook appends its call while `complete` is
`False`. The root hook sets `complete = True` when the whole forward pass returns, so
later passes add nothing.

| | static (`order=None`) | traced (`order=trace.stop()`) |
|---|---|---|
| source | registration order | order in which leaf forward hooks fired |
| module used twice | listed once | each call. Operations repeat, connection layers are drawn once |
| module that never runs | included | absent |
| lazy module | sizes `None` | sizes known: the build happens after the pass initialised it |

## Used by / uses

* Uses [Tracking Layer Classification](Tracking-Layer-Classification),
  [Tracking Decorator](Tracking-Decorator) (`TrackOptions`) and
  [Tracking Architecture Model](Tracking-Architecture-Model).
* [Tracking Network Tracker](Tracking-Network-Tracker) uses `build_architecture`,
  `ForwardTrace` and `architecture_signature`.
* Lazily exported: `ForwardTrace`, `build_architecture`, `leaf_modules`, `unwrap`.

## Example

```python
import torch
from torch import nn
from hollywood_training.tracking import ForwardTrace, build_architecture

class Net(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.out = nn.Linear(4, 2)   # registered first, runs last
        self.act = nn.ReLU()         # one module, used twice
        self.fc1 = nn.Linear(3, 4)
        self.fc2 = nn.Linear(4, 4)

    def forward(self, x):
        return self.out(self.act(self.fc2(self.act(self.fc1(x)))))

model = Net()
[layer.name for layer in build_architecture(model).layers]      # ['out', 'fc1', 'fc2']
trace = ForwardTrace(model)
trace.start()
model(torch.randn(2, 3))
calls = trace.stop()          # names: fc1, act, fc2, act, out
traced = build_architecture(model, order=calls)
[(layer.name, [op.name for op in layer.operations]) for layer in traced.layers]
# [('fc1', ['act']), ('fc2', ['act']), ('out', [])]
```

## Notes

* Only modules are seen. Functional calls (`torch.relu(x)`, `F.dropout(x)`) and tensor
  operations in `forward` appear in neither order.
* An unsupported module does not end the current layer: in `Linear → GRU → ReLU`, the
  `ReLU` is attached to the `Linear`.
* If the forward pass raises, the root hook never fires. `complete` stays `False`, and
  `stop()` returns the calls recorded up to the error. `NetworkTracker.end_trace()`
  discards such a trace.
* Parameters owned directly by a non-leaf module count in `total_parameters`, but such a
  module is never a layer.

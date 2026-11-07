# Tracking Inspector

`hollywood_training/tracking/inspector.py` · part of [Tracking](Tracking)

The inspector groups the modules of a network into connection layers and the operations
applied to their outputs, producing a [NetworkArchitecture](Tracking-Architecture-Model).
The modules considered are the leaf modules plus every weight owner with children, such
as `nn.MultiheadAttention`, which holds `in_proj_weight` next to its `out_proj` child
(see [Tracking Layer Classification](Tracking-Layer-Classification)). Their order comes
from one of two official PyTorch APIs. The **static** order is the registration order of
`named_modules()`, available at once. The **traced** order is the order in which those
modules actually ran in one forward pass, recorded by `ForwardTrace` with
`register_forward_hook` and `register_forward_pre_hook`.

## Responsibilities

* Look through `nn.DataParallel` / `DistributedDataParallel` wrappers and the
  `OptimizedModule` that `torch.compile` returns, list the modules that can be drawn and
  count initialised parameters.
* Group a module sequence into `TrackedLayer`s, `LayerOperation`s and unsupported names.
* Record the execution order of the first complete forward pass.
* Fingerprint an architecture, so a rebuilt one can be compared with the previous one.

## API

```python
ModuleSequence = Sequence[tuple[str, nn.Module]]

def unwrap(model: nn.Module) -> nn.Module
def leaf_modules(model: nn.Module) -> list[tuple[str, nn.Module]]
def static_order(model: nn.Module) -> list[tuple[str, nn.Module]]
def count_parameters(
    module: nn.Module, *, trainable_only: bool = False, recurse: bool = True
) -> int
```

* `unwrap`: follows `.module` while the model is an `nn.DataParallel` or
  `nn.parallel.DistributedDataParallel`. It also looks through the `OptimizedModule` that
  `torch.compile` returns: that wrapper holds the original network as its only child,
  `_orig_mod`, read through the public `named_children()`. PyTorch has no public unwrap
  API, so the wrapper is recognised by its class name, without importing the compiler.
  Wrappers can nest in any order, e.g. `torch.compile(nn.DataParallel(net))`. The result
  is the user's network, so a compiled network keeps its own class name (`TinyNet`, not
  `OptimizedModule`) and its own layer names (`fc1`, not `_orig_mod.fc1`).
* `leaf_modules`: `(qualified_name, module)` for every module of `unwrap(model)` without
  children, in `named_modules()` order. The root (name `""`) is listed only when it has no
  children. A module registered under two names is listed once, because
  `named_modules()` removes duplicates.
* `static_order`: the modules that can take part in the drawing, in `named_modules()`
  order: every leaf module, plus every module with children that `classify` makes a
  connection layer, i.e. a weight owner such as `nn.MultiheadAttention`. A parent comes
  before its children. Containers such as `nn.Sequential` hold no weight tensor and are
  not listed.
* `count_parameters`: sum of `numel()` over `module.parameters(recurse=recurse)`,
  skipping uninitialised lazy parameters. With `trainable_only=True`, only parameters
  with `requires_grad` count. With `recurse=False`, only the parameters the module holds
  itself count.

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
  Defaults to `static_order(model)`. The result has `traced=True` exactly when `order` is
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
  sets `complete = False` and registers a recording hook on every module of
  `static_order(model)`: a forward hook on each leaf, and a forward pre-hook on each
  weight owner with children. One more forward hook goes on the root module.
* `stop()`: removes all hooks and returns the recorded `(name, module)` calls, even when
  no pass completed. A connection layer that did not run is placed right after its
  parent's first call, when that parent is a weight owner that ran.
  `nn.MultiheadAttention` hands `out_proj.weight` to the functional attention and never
  calls `out_proj`. Without this rule its weights would leave the traced architecture.
* `active` is `True` while hooks are registered. `complete` is `True` once the root
  module's forward has returned.

## How it works

`build_architecture` walks the sequence once, keeping a `current` connection layer:

1. `label = name or type(module).__name__`. Pass-through modules (`nn.Identity`) are
   skipped.
2. `CONNECTION`: the first time a module object is seen (compared by `id`), it becomes a
   new `TrackedLayer`, with `summarize()`, `connection_shape()` and
   `count_parameters(module)` (with `recurse=False` for a weight owner with children:
   the parameters of a child that is a layer count at that child), and is made `current`. When the same object comes again,
   its existing layer becomes `current` again, so it is drawn once.
3. Operation roles: a `LayerOperation` is appended to `current.operations`, or to
   `input_operations` while there is no `current` layer.
4. `OTHER`: the label is appended to `unsupported` unless it is already there. `current`
   is unchanged.

`ForwardTrace` registers the recording hooks before the root hook, so a model that is
itself a leaf still records its own call. Each recording hook appends its call while
`complete` is `False`. A weight owner with children is recorded by its pre-hook, where its
forward begins, so it comes before the children it calls. The root hook sets
`complete = True` when the whole forward pass returns, so later passes add nothing.

| | static (`order=None`) | traced (`order=trace.stop()`) |
|---|---|---|
| source | registration order | order in which the recording hooks fired |
| module used twice | listed once | each call. Operations repeat, connection layers are drawn once |
| module that never runs | included | absent, except a connection layer whose parent is a weight owner that ran: right after that parent |
| lazy module | sizes `None` | sizes known: the build happens after the pass initialised it |

## Used by / uses

* Uses [Tracking Layer Classification](Tracking-Layer-Classification),
  [Tracking Decorator](Tracking-Decorator) (`TrackOptions`) and
  [Tracking Architecture Model](Tracking-Architecture-Model).
* [Tracking Network Tracker](Tracking-Network-Tracker) uses `build_architecture`,
  `ForwardTrace` and `architecture_signature`.
* Lazily exported: `ForwardTrace`, `build_architecture`, `leaf_modules`, `static_order`,
  `unwrap`.

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
* An unsupported module does not end the current layer: in `Linear → ZeroPad1d → ReLU`,
  the `ReLU` is attached to the `Linear`.
* If the forward pass raises, the root hook never fires. `complete` stays `False`, and
  `stop()` returns the calls recorded up to the error. `NetworkTracker.end_trace()`
  discards such a trace.
* A module with children is a layer only when it is a weight owner. Then its layer counts
  only the parameters it holds itself. Parameters held directly by any other module with
  children (vectors and scalars, since a weight tensor would make it a weight owner) count
  in `total_parameters` only.

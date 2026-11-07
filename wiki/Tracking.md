# Tracking

`hollywood_training/tracking/` · subpackage · layer overview in [Architecture](Architecture)

The tracking subpackage answers two questions about the network being trained: what it
is (its connection layers and the operations between them) and what its weights are (a
snapshot of every connection layer, taken at every epoch). It reads the model through
public PyTorch APIs only and turns it into immutable plain data. A frontend can draw that
data on another thread, or in the window process, which imports no torch.

## Data flow

```
 @track / track(instance)            decorator.py   TrackOptions kept in weak registries
              │ track_options(model)
              ▼
 NetworkTracker(model)               tracker.py
   ├─ build_architecture(model)      inspector.py   static order: named_modules()
   │    └─ classify · summarize      layers.py      one LayerRole per module (roles.py)
   ├─ begin_trace() … end_trace()    inspector.py   ForwardTrace: forward hooks, first pass
   │    └─ build_architecture(model, order=calls)   refined once
   └─ snapshot(epoch)                snapshot.py    connection matrices · stats · delta
              │
              ▼
 ArchitectureInfo (architecture.py) · WeightSnapshot (weights.py)   torch-free plain data
              │
              ▼
 SessionStarted · EpochCompleted · TrainingPaused · SessionFinished  →  frontends
```

1. **Declare.** `@track` or `track(instance)` stores `TrackOptions` (display name,
   input/output labels) and returns the target unchanged.
2. **Build.** `NetworkTracker(model)` builds a `NetworkArchitecture` at once, using the
   registration order of `named_modules()`. Each leaf module, and each weight owner
   with children such as `nn.MultiheadAttention`, gets a [role](Tracking-Layer-Roles).
   Connection layers become `TrackedLayer`s, and each operation that follows one is
   attached to it.
3. **Refine.** The [Training Manager](Training-Manager) wraps the first training batch in
   `begin_trace()` / `end_trace()`. A `ForwardTrace` uses forward hooks to record those
   modules in the order they ran during the first complete forward pass. If that
   pass completed, the architecture is rebuilt once from the recorded order (now
   `traced=True`), and the first `EpochCompleted` event carries the refined
   `ArchitectureInfo`. A pass that raised is not adopted.
4. **Snapshot.** `snapshot(epoch)` reduces each connection layer to an `(out, in)`
   connection matrix and keeps at most `max_neurons` evenly spaced neurons per side. It
   computes statistics on the full tensor and on each weight tensor of the layer, plus the
   change since the previous baseline snapshot.
5. **Hand off.** `NetworkArchitecture.describe()` returns an `ArchitectureInfo`. It and
   the `WeightSnapshot` hold no modules or tensors, so [Sync Events](Sync-Events) can
   carry them to any frontend, including across the [link](GUI-Link-Protocol) to the
   window process.

## Terms

| Term | Meaning |
|---|---|
| weight tensor | an initialised parameter with two or more dimensions, held directly by a module (`weight_tensors`) |
| weight owner | a module that holds a weight tensor and is not an operation class: `Bilinear`, `RNN`, `LSTM`, `GRU`, `MultiheadAttention`, custom modules. A weight owner can have children: `MultiheadAttention` holds `in_proj_weight` next to its `out_proj` child |
| connection layer | module with role `CONNECTION`: `Linear`, `Conv*`, `ConvTranspose*`, `Embedding`, `EmbeddingBag` (subclasses and lazy variants included), or any other weight owner. So every weight tensor of the network belongs to a connection layer and is tracked |
| operation | leaf module with an activation, normalization, regularization, pooling or reshape role. It is attached to the connection layer before it, or to the input |
| unsupported | leaf module with role `OTHER`, i.e. without a weight tensor and not an operation (e.g. `nn.ZeroPad2d`). It is listed by name and not drawn |
| connection matrix | `(out, in)` matrix per connection layer: entry `[j, i]` says how strongly input `i` feeds output `j` |
| neuron limit | `max_neurons` / `limit`: neurons kept per side in a snapshot, default `DEFAULT_NEURON_LIMIT = 20` |
| baseline | the last snapshot taken with `baseline=True`. Deltas are measured against it |

## Modules

| Module | Page | Needs torch | Contents |
|---|---|---|---|
| `decorator.py` | [Tracking Decorator](Tracking-Decorator) | yes | `track`, `TrackOptions`, `track_options`, `is_tracked` |
| `roles.py` | [Tracking Layer Roles](Tracking-Layer-Roles) | no | `LayerRole` |
| `layers.py` | [Tracking Layer Classification](Tracking-Layer-Classification) | yes | `classify`, `connection_shape`, `summarize`, `weight_tensors`, class tuples |
| `architecture.py` | [Tracking Architecture Model](Tracking-Architecture-Model) | no (type hints only) | `ArchitectureInfo`, `LayerInfo`, `OperationInfo` and their live counterparts |
| `inspector.py` | [Tracking Inspector](Tracking-Inspector) | yes | `build_architecture`, `ForwardTrace`, `static_order`, `leaf_modules`, `unwrap` |
| `weights.py` | [Tracking Weight Data](Tracking-Weight-Data) | no (numpy) | `WeightSnapshot`, `LayerWeights`, `WeightStats`, `TensorStats`, `display_indices` |
| `snapshot.py` | [Tracking Snapshots](Tracking-Snapshots) | yes | `capture`, `capture_layer`, `connection_matrix`, `weight_stats`, `tensor_stats` |
| `tracker.py` | [Tracking Network Tracker](Tracking-Network-Tracker) | yes | `NetworkTracker`, `inspect_network` |

## Package exports (`__init__.py`)

`hollywood_training.tracking` imports its torch-free modules eagerly and the others on
first use (PEP 562):

| Kind | Names | Source |
|---|---|---|
| eager | `ArchitectureInfo`, `LayerInfo`, `LayerOperation`, `NetworkArchitecture`, `OperationInfo`, `TrackedLayer` | `.architecture` |
| eager | `LayerRole` | `.roles` |
| eager | `DEFAULT_NEURON_LIMIT`, `LayerWeights`, `TensorStats`, `WeightSnapshot`, `WeightStats`, `display_indices` | `.weights` |
| lazy | `TrackOptions`, `is_tracked`, `track`, `track_options` | `.decorator` |
| lazy | `ForwardTrace`, `build_architecture`, `leaf_modules`, `static_order`, `unwrap` | `.inspector` |
| lazy | `classify`, `connection_shape`, `summarize` | `.layers` |
| lazy | `capture`, `capture_layer`, `connection_matrix` | `.snapshot` |
| lazy | `NetworkTracker`, `inspect_network` | `.tracker` |

* The module-level `__getattr__(name)` looks `name` up in the explicit `_LAZY` map. It
  imports the module with `import_module(module, __name__)` and caches the value in the
  package globals, so later lookups skip `__getattr__`. A name outside the map raises
  `AttributeError`.
* `__dir__()` returns `sorted(__all__)`, the 30 names above.
* `import hollywood_training.tracking` and unpickling `ArchitectureInfo` or
  `WeightSnapshot` therefore never import torch. Accessing any lazy name does.
* Other public helpers (`weight_stats`, `tensor_stats`, `count_parameters`,
  `architecture_signature`, `weight_of`, `weight_tensors`, `primary_weight`,
  `is_uninitialized`, ...) are not re-exported. Import them from their module.
* The top-level package re-exports `track`, `is_tracked`, `NetworkTracker` and
  `inspect_network`, lazily as well (see [API Reference](API-Reference)).

## How the model is read

Structure comes from `named_modules()`, `children()`, `parameters()`,
`named_parameters(recurse=False)`, `register_forward_hook`, `register_forward_pre_hook`,
public `torch.nn` classes looked up by name, and `type(module).__name__`. Details come from
public module attributes: `weight`, `bias`, `in_channels`, `out_channels`, `groups`,
`kernel_size`, `in1_features`, `in2_features`, `out_features`, `embed_dim`, `kdim`,
`vdim`, `num_heads`, `in_proj_bias`, `p`, `negative_slope`, `num_features`,
`normalized_shape`, and `.module` of `DataParallel` / `DistributedDataParallel` wrappers.
The network inside `torch.compile`'s `OptimizedModule` is its `_orig_mod` child, read
through `named_children()`. Weights are copied with ordinary tensor operations under
`torch.no_grad()`. User code is never inspected.

## Example

What the manager does, written out by hand:

```python
import torch
from torch import nn
from hollywood_training.tracking import NetworkTracker, track

model = track(nn.Sequential(nn.Linear(2, 8), nn.ReLU(), nn.Linear(8, 1)))
tracker = NetworkTracker(model, max_neurons=4)
info = tracker.architecture.describe()            # ArchitectureInfo: plain data
[layer.summary for layer in info.layers]          # ['Linear 2→8', 'Linear 8→1']

tracker.snapshot(0)                               # initial weights, becomes the baseline
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
tracker.begin_trace()                             # first training batch only
model(torch.randn(16, 2)).pow(2).mean().backward()
optimizer.step()
tracker.end_trace()                               # True: complete pass, traced flips to True

weights = tracker.snapshot(1)
weights.layer("0").matrix.shape                   # (4, 2): 4 of the 8 hidden neurons
weights.layer("0").update_ratio                   # relative change since snapshot 0
```

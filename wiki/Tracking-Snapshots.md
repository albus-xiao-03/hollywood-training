# Tracking Snapshots

`hollywood_training/tracking/snapshot.py` · part of [Tracking](Tracking)

This module takes the per-epoch copies of the weights a frontend draws. It reduces every
connection layer to a **connection matrix** shaped `(out, in)`, keeps at most `limit`
evenly spaced neurons per side, and copies the result into read-only CPU numpy arrays. It
also computes statistics on the full weight tensor and on every weight tensor of the
layer, and the change since a previous snapshot. The result is a [WeightSnapshot](Tracking-Weight-Data) that can be handed to
another thread while training continues.

## Responsibilities

* Turn any supported weight layout into an `(out, in)` matrix (`connection_matrix`).
* Sample, copy and freeze one layer (`capture_layer`) or all layers (`capture`).
* Compute `WeightStats` on the full tensor (`weight_stats`), and `TensorStats` for every
  weight tensor of a layer (`tensor_stats`).
* Compute `delta` and `update_ratio` against a previous snapshot.

## API

```python
def connection_matrix(
    module: nn.Module, in_index: np.ndarray, out_index: np.ndarray
) -> torch.Tensor | None
```

Returns the connection matrix of `module`, restricted to the given indices, as a float32
tensor of shape `(len(out_index), len(in_index))` on the weight's device. Returns `None`
when the module has no weight tensor (yet). The matrix is read from `primary_weight(module)`
([Tracking Layer Classification](Tracking-Layer-Classification)). Entry `[j, i]` says how
strongly input `in_index[i]` feeds output `out_index[j]`:

| Module (checked in this order) | PyTorch weight layout | Connection matrix |
|---|---|---|
| `Embedding`, `EmbeddingBag` | `(num_embeddings, embedding_dim)` | the transposed table, `(embedding_dim, num_embeddings)` |
| `ConvTranspose1d/2d/3d` | `(in, out / groups, *kernel)` | signed mean over the kernel window, block-diagonal for groups, transposed back to `(out, in)` |
| `Conv1d/2d/3d` | `(out, in / groups, *kernel)` | signed mean over the kernel window. Channel pairs in different groups are `0` |
| `Bilinear` | `(out, in1, in2)` | inputs x1 then x2, `(out, in1 + in2)`: an x1 feature gets the mean over x2, an x2 feature the mean over x1 |
| any other weight owner (`Linear`, linear-like, and any module with a weight tensor) | `(out, in, ...)` | the tensor itself when it is 2-D, otherwise its signed mean over the dimensions after the first two |

```python
def capture_layer(
    layer: TrackedLayer, limit: int, previous: LayerWeights | None = None
) -> LayerWeights | None
```

Snapshots one layer. Returns `None` when the module has no initialised weight tensor,
when the `TrackedLayer` has no `in_features` / `out_features`, or when
`connection_matrix` returns `None`. When the weight and both sizes are known, `limit < 1`
raises `ValueError` from `display_indices`. `previous` is this layer in an earlier
snapshot; the tracker passes its baseline.

```python
def capture(
    architecture: NetworkArchitecture,
    *,
    epoch: int,
    limit: int = DEFAULT_NEURON_LIMIT,
    previous: WeightSnapshot | None = None,
) -> WeightSnapshot
```

Calls `capture_layer` for every layer of `architecture`, passing
`previous.layer(layer.name)` as its previous weights. The result keeps the layers that
returned a value, in architecture order.

```python
def weight_stats(weight: torch.Tensor) -> WeightStats
def tensor_stats(
    module: nn.Module, known: dict[int, WeightStats] | None = None
) -> tuple[TensorStats, ...]
```

Statistics of the full tensor: detached, converted to float32 when needed, and reduced
on the tensor's own device. `std` uses `correction=0`, and `norm` is
`torch.linalg.vector_norm`. An empty tensor gives `WeightStats(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0)`.

`tensor_stats` returns one `TensorStats(name, shape, stats)` for every weight tensor the
module holds (`weight_tensors`, in registration order). `known` maps `id(tensor)` to
statistics already computed, which are reused instead of computed again.

## How it works

`capture_layer(layer, limit, previous)`:

1. `in_index = display_indices(in_features, limit)`, and the same for `out_index`.
2. Under `torch.no_grad()`, it calls `connection_matrix`. The selection (`index_select`)
   and the kernel mean run on the weight's device, so only the sampled matrix is copied
   to the CPU (at most `limit × limit` entries). The copy is
   `tensor.detach().to(device="cpu", dtype=torch.float32).numpy().copy()`, then made
   read-only.
3. The bias is read when `module.bias` is an initialised vector with one entry per output
   feature, at `out_index`, and copied the same way. `stats = weight_stats(weight)` uses
   the whole primary weight. `tensors = tensor_stats(module)` adds the statistics of every
   weight tensor of the layer, reusing `stats` for the primary weight.
4. `delta` and `update_ratio` are computed only when `previous` has the same matrix shape
   and identical `in_index` and `out_index`. Then `delta = matrix - previous.matrix`
   (read-only) and `update_ratio = norm(delta) / (norm(previous.matrix) + 1e-12)`.

**Convolutions.** The weight rows of the selected output channels (for transposed
convolutions, the selected input channels) are averaged over every kernel dimension. The
mean keeps its sign, so positive and negative taps cancel. With `groups > 1`, output
channel `j` belongs to group `j // (out_channels // groups)`, and input channel `i` to
group `i // (in_channels // groups)`. The entry is the kernel mean of the local channel
pair when both groups match, and `0` otherwise. A depthwise convolution
(`groups == channels`) therefore yields a diagonal matrix.

**Bilinear layers.** `nn.Bilinear` computes `y[j] = x1 · W[j] · x2 + b[j]`, so the weight
`W[j, i, k]` links x1 feature `i` and x2 feature `k` to output `j`. The drawn inputs are
the in1 features of x1 followed by the in2 features of x2. The entry of x1 feature `i` is
the signed mean of `W[j, i, :]`, and the entry of x2 feature `k` is the signed mean of
`W[j, :, k]`, the same rule as the convolution kernel mean.

## Used by / uses

* Uses [Tracking Layer Classification](Tracking-Layer-Classification) (`primary_weight`,
  `weight_tensors`, `is_uninitialized`, the convolution and embedding class tuples),
  [Tracking Weight Data](Tracking-Weight-Data) (data classes, `display_indices`,
  `DEFAULT_NEURON_LIMIT`) and [Tracking Architecture Model](Tracking-Architecture-Model).
* [Tracking Network Tracker](Tracking-Network-Tracker): `NetworkTracker.snapshot()` calls
  `capture` with its `max_neurons` and its baseline.
* Lazily exported: `capture`, `capture_layer`, `connection_matrix`. `weight_stats` and
  `tensor_stats` are only available from this module.

## Example

```python
import torch
from torch import nn
from hollywood_training.tracking import build_architecture, capture, connection_matrix, display_indices

conv = nn.Conv2d(4, 6, 3, groups=2)
matrix = connection_matrix(conv, display_indices(4, 64), display_indices(6, 64))
matrix.shape                         # torch.Size([6, 4])
bool((matrix[:3, 2:] == 0).all())    # True: outputs 0-2 only see inputs 0-1

model = nn.Sequential(nn.Linear(3, 30), nn.ReLU(), nn.Linear(30, 2))
architecture = build_architecture(model)
first = capture(architecture, epoch=0, limit=8)
with torch.no_grad():
    model[0].weight.add_(0.25)
second = capture(architecture, epoch=1, limit=8, previous=first)
second.layer("0").delta              # about 0.25 everywhere (8 x 3, float32)
second.layer("2").update_ratio       # 0.0: unchanged
```

## Notes

* Call it from the thread that trains the model, between optimisation steps. This is
  the [NetworkTracker](Tracking-Network-Tracker) contract, which the
  [Training Manager](Training-Manager) follows. Nothing synchronises a read with an
  optimizer step running at the same time.
* A layer built while its lazy module was uninitialised has `in_features=None`. It is
  skipped until the architecture is rebuilt, even after the module has been initialised.
* `connection_matrix` returns a torch tensor. Only `capture_layer` converts to numpy.

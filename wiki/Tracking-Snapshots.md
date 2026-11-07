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
* Read the recurrent weights of a recurrent layer as its `Recurrence`
  (`recurrent_matrix`).

## API

```python
def connection_matrix(
    module: nn.Module,
    in_index: np.ndarray,
    out_index: np.ndarray,
    *,
    stage: int = 0,
    projection: bool = False,
) -> torch.Tensor | None
```

Returns the connection matrix of `module`, restricted to the given indices, as a float32
tensor of shape `(len(out_index), len(in_index))` on the weight's device. Returns `None`
when the module has no weight tensor (yet). The matrix is read from `primary_weight(module)`
([Tracking Layer Classification](Tracking-Layer-Classification)). `stage` and
`projection` select a stage of a recurrent layer. Entry `[j, i]` says how strongly input
`in_index[i]` feeds output `out_index[j]`:

| Module (checked in this order) | PyTorch weight layout | Connection matrix |
|---|---|---|
| `RNN`, `LSTM`, `GRU` stage `k`, and their cells | `weight_ih_l<k>` `(G·H, in)` per direction | the signed mean over the `G` gate blocks, one row block per direction (forward first): `(D·H, in)` |
| their projection stage | `weight_hr_l<k>` `(P, H)` per direction | block-diagonal: each direction projects its own hidden state, `(D·P, D·H)` |
| `Embedding`, `EmbeddingBag` | `(num_embeddings, embedding_dim)` | the transposed table, `(embedding_dim, num_embeddings)` |
| `ConvTranspose1d/2d/3d` | `(in, out / groups, *kernel)` | signed mean over the kernel window, block-diagonal for groups, transposed back to `(out, in)` |
| `Conv1d/2d/3d` | `(out, in / groups, *kernel)` | signed mean over the kernel window. Channel pairs in different groups are `0` |
| `MultiheadAttention` | `in_proj_weight` `(3E, E)`, or `q_proj_weight` `(E, E)`, `k_proj_weight` `(E, kdim)`, `v_proj_weight` `(E, vdim)` | the outputs are the query, key and value projections: `in_proj_weight` itself, or the block-diagonal matrix of the three separate weights over the query, key and value features |
| `Bilinear` | `(out, in1, in2)` | inputs x1 then x2, `(out, in1 + in2)`: an x1 feature gets the mean over x2, an x2 feature the mean over x1 |
| any other weight owner (`Linear`, linear-like, and any module with a weight tensor) | `(out, in, ...)` | the tensor itself when it is 2-D, otherwise its signed mean over the dimensions after the first two |

```python
def recurrent_matrix(
    module: nn.Module, out_index: np.ndarray, source_index: np.ndarray, *, stage: int = 0
) -> torch.Tensor | None
```

The recurrent connections of stacked layer `stage` of an `RNN`, `LSTM`, `GRU` or cell,
restricted to the given indices: the signed gate mean of `weight_hh_l<k>`, block-diagonal
per direction, because each direction feeds back only its own state. Rows are the
stage's outputs (`D·H`), columns the outputs that feed back: the stage's own (`D·H`), or
those of its projection stage (`D·P`). `None` for any other module.

```python
def capture_layer(
    layer: TrackedLayer, limit: int, previous: LayerWeights | None = None
) -> LayerWeights | None
```

Snapshots one layer. Returns `None` when the module has no initialised weight tensor,
when the `TrackedLayer` has no `in_features` / `out_features`, or when
`connection_matrix` returns `None`. When the weight and both sizes are known, `limit < 1`
raises `ValueError` from `display_indices`. `previous` is this layer in an earlier
snapshot; the tracker passes its baseline. The stage of a recurrent layer comes from
`layer.stage` and `layer.projection`.

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
    module: nn.Module,
    known: dict[int, WeightStats] | None = None,
    *,
    stage: int = 0,
    projection: bool = False,
) -> tuple[TensorStats, ...]
```

Statistics of the full tensor: detached, converted to float32 when needed, and reduced
on the tensor's own device. `std` uses `correction=0`, and `norm` is
`torch.linalg.vector_norm`. An empty tensor gives `WeightStats(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0)`.

`tensor_stats` returns one `TensorStats(name, shape, stats)` for every weight tensor of a
layer (`stage_weight_tensors`, in registration order): all of the module's, or those of
one stage of a recurrent layer. `known` maps `id(tensor)` to statistics already computed,
which are reused instead of computed again.

## How it works

`capture_layer(layer, limit, previous)`:

1. `in_index = display_indices(in_features, limit)`, and the same for `out_index`.
2. Under `torch.no_grad()`, it calls `connection_matrix`. The selection (`index_select`)
   and the kernel mean run on the weight's device, so only the sampled matrix is copied
   to the CPU (at most `limit × limit` entries). The copy is
   `tensor.detach().to(device="cpu", dtype=torch.float32).numpy().copy()`, then made
   read-only.
3. The bias is read when `module.bias` (`in_proj_bias` for attention) is an initialised
   vector with one entry per output feature, at `out_index`, and copied the same way. A
   recurrent stage's bias is the gate mean of `bias_ih + bias_hh` per direction; a
   projection stage has none. `stats` covers the whole tensor the matrix is read from:
   the primary weight, the separate query, key and value weights taken together, or the
   `weight_ih` (`weight_hr`) of every direction of a recurrent stage. `tensors =
   tensor_stats(...)` adds the statistics of every weight tensor of the layer, reusing
   `stats` when it covers a single tensor.
4. `delta` and `update_ratio` are computed only when `previous` has the same matrix shape
   and identical `in_index` and `out_index`. Then `delta = matrix - previous.matrix`
   (read-only) and `update_ratio = norm(delta) / (norm(previous.matrix) + 1e-12)`.
5. A recurrent stage (not a projection) also gets its `Recurrence`: `recurrent_matrix`
   from the stage's own outputs, or from its `.proj` stage when the layer is projected.
   `source_index = display_indices(source features, limit)`, the same indices the source
   layer's outputs are drawn with. Its `delta` is computed like the layer's, when the
   previous recurrence has the same shape and indices.

**Convolutions.** The weight rows of the selected output channels (for transposed
convolutions, the selected input channels) are averaged over every kernel dimension. The
mean keeps its sign, so positive and negative taps cancel. With `groups > 1`, output
channel `j` belongs to group `j // (out_channels // groups)`, and input channel `i` to
group `i // (in_channels // groups)`. The entry is the kernel mean of the local channel
pair when both groups match, and `0` otherwise. A depthwise convolution
(`groups == channels`) therefore yields a diagonal matrix.

**Attention layers.** `nn.MultiheadAttention` projects its query, key and value inputs
with `in_proj_weight` (rows `0..E-1` make the query, `E..2E-1` the key, `2E..3E-1` the
value) before the attention itself, which has no weights. So its connection matrix is
that projection, and the bias is `in_proj_bias`. When `kdim` or `vdim` differs from `E`,
PyTorch keeps three weights instead. The inputs are then the query, key and value
features (`E + kdim + vdim`), and each projection connects only its own input: the
matrix is block-diagonal, built from the sampled rows and columns only. The output
projection `out_proj` is a `Linear` child and is a layer of its own. `bias_k` and
`bias_v` (with `add_bias_kv=True`) are tracked in `tensors`.

**Recurrent layers.** `nn.RNN`, `nn.LSTM` and `nn.GRU` stack `G` gate blocks of `H`
rows in each weight (`G` = 1, 4, 3). The connection matrix of a stage is the signed mean
over those blocks, as the kernel mean is for convolutions: entry `[u, i]` is the mean of
the `G` weights that link input `i` to hidden unit `u`. With `bidirectional=True` the
forward units come first, then the reverse units, and a stacked layer's input is both
directions' output of the layer below. The hidden-to-hidden weights `weight_hh` are the
layer's `Recurrence`, so they are drawn and tracked too. With `proj_size` P, an LSTM's
output is `weight_hr` times its hidden state: a projection stage `D·H → D·P` follows
each stacked layer, and the recurrence comes from the projection. The cells are read
like a one-layer, one-direction `RNN`.

**Bilinear layers.** `nn.Bilinear` computes `y[j] = x1 · W[j] · x2 + b[j]`, so the weight
`W[j, i, k]` links x1 feature `i` and x2 feature `k` to output `j`. The drawn inputs are
the in1 features of x1 followed by the in2 features of x2. The entry of x1 feature `i` is
the signed mean of `W[j, i, :]`, and the entry of x2 feature `k` is the signed mean of
`W[j, :, k]`, the same rule as the convolution kernel mean.

## Used by / uses

* Uses [Tracking Layer Classification](Tracking-Layer-Classification) (`primary_weight`,
  `stage_weight_tensors`, `recurrent_tensors`, `gate_count`, the class tuples, ...),
  [Tracking Weight Data](Tracking-Weight-Data) (data classes, `display_indices`,
  `DEFAULT_NEURON_LIMIT`) and [Tracking Architecture Model](Tracking-Architecture-Model).
* [Tracking Network Tracker](Tracking-Network-Tracker): `NetworkTracker.snapshot()` calls
  `capture` with its `max_neurons` and its baseline.
* Lazily exported: `capture`, `capture_layer`, `connection_matrix`. `weight_stats`,
  `tensor_stats` and `recurrent_matrix` are only available from this module.

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

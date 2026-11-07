# Tracking Weight Data

`hollywood_training/tracking/weights.py` · part of [Tracking](Tracking)

This module holds what a frontend receives about the weights: one `WeightSnapshot` per
point in time, with one `LayerWeights` per connection layer. The data is small, immutable,
picklable and uses numpy only, never torch, so another thread or the window process can
draw it. The snapshots themselves are produced by [Tracking Snapshots](Tracking-Snapshots)
in the training process.

## Responsibilities

* Define the frozen data classes of a weight snapshot and its statistics.
* Choose which neurons of a wide layer are displayed (`display_indices`).
* Define the default neuron limit (`DEFAULT_NEURON_LIMIT`).

## API

```python
DEFAULT_NEURON_LIMIT = 20
```

The default number of neurons kept per side. It is the default for `capture(limit=...)`,
`NetworkTracker(max_neurons=...)` and `run(max_neurons=...)`, and the
[Network View](GUI-Network-View)'s fallback when no session info is available.

```python
@dataclass(frozen=True, slots=True)
class WeightStats:
    mean: float
    std: float
    minimum: float
    maximum: float
    abs_mean: float
    norm: float
    count: int
```

Summary statistics of the **full** weight tensor, not just the displayed part.
`weight_stats()` computes them in float32: `std` is the population standard deviation
(`correction=0`), `abs_mean` is the mean of `|w|`, `norm` is the L2 norm over all
elements, and `count` is the number of elements. An empty tensor gives all zeros.

```python
@dataclass(frozen=True, slots=True)
class TensorStats:
    name: str
    shape: tuple[int, ...]
    stats: WeightStats
```

One weight tensor of a layer: the parameter's name in its module (`weight`,
`in_proj_weight`, `weight_hh_l0`, ...), its full shape, and the `WeightStats` of its full
content. It compares by value.

```python
@dataclass(frozen=True, slots=True, eq=False)
class LayerWeights:
    name: str
    matrix: np.ndarray  # (len(out_index), len(in_index)) float32
    bias: np.ndarray | None  # (len(out_index),) float32
    delta: np.ndarray | None  # change of `matrix` since the previous *baseline* snapshot
    in_index: np.ndarray  # displayed input feature indices
    out_index: np.ndarray  # displayed output feature indices
    in_features: int
    out_features: int
    stats: WeightStats  # of the full tensor `matrix` is drawn from
    update_ratio: float | None  # ||delta|| / ||previous matrix||
    tensors: tuple[TensorStats, ...] = ()  # every weight tensor of the layer

    @property
    def truncated(self) -> bool: ...
```

| Field | Meaning |
|---|---|
| `name` | the connection layer's name in the architecture (e.g. `fc1`, `0`) |
| `matrix` | sampled connection matrix. `matrix[j, i]` says how strongly input `in_index[i]` feeds output `out_index[j]` |
| `bias` | bias at `out_index`, or `None` when the module has no initialised `bias` tensor |
| `delta` | `matrix` minus the same layer's matrix in the baseline snapshot. `None` when there is no baseline for the layer, or the displayed indices differ |
| `in_index`, `out_index` | sorted `int64` feature indices that are displayed |
| `in_features`, `out_features` | full layer sizes |
| `stats` | `WeightStats` of the full tensor `matrix` is drawn from: the layer's `weight` for `Linear`, convolution and embedding layers, `in_proj_weight` for attention, or its separate query, key and value weights taken together |
| `update_ratio` | `norm(delta) / (norm(previous.matrix) + 1e-12)`, Frobenius norms of the displayed matrices. `None` together with `delta` |
| `tensors` | `TensorStats` of every weight tensor of the layer, in registration order: one (`weight`) for `Linear`, convolution and embedding layers, more for recurrent and attention layers. A snapshot made without them leaves the default `()` |

`truncated` is `True` when fewer indices than features are displayed on either side.

```python
@dataclass(frozen=True, slots=True, eq=False)
class WeightSnapshot:
    epoch: int
    layers: tuple[LayerWeights, ...]

    def layer(self, name: str) -> LayerWeights | None: ...
    def names(self) -> tuple[str, ...]: ...
```

`epoch` is the number of completed epochs (`0` holds the initial weights). `layers`
follows the architecture's layer order and leaves out layers that are still uninitialised.
`layer(name)` returns the first match or `None`. `names()` returns the layer names in
order.

```python
def display_indices(size: int, limit: int) -> np.ndarray
```

| Case (checked in this order) | Result (`int64`) |
|---|---|
| `limit < 1` | raises `ValueError("limit must be at least 1")` |
| `size <= 0` | empty array |
| `size <= limit` | `np.arange(size)` |
| otherwise | `np.unique(np.linspace(0, size - 1, limit).round().astype(np.int64))`: at most `limit` evenly spaced indices, including `0` and `size - 1` when `limit >= 2` |

For example, `display_indices(100, 5)` is `[0, 25, 50, 74, 99]`, and
`display_indices(10, 1)` is `[0]`.

## How it works

* The indices depend only on `(size, limit)`. When a layer's `out_features` equals the
  next layer's `in_features`, its `out_index` equals the next layer's `in_index`, so both
  layers line up on the same column of neurons.
* The arrays come from the snapshot module as CPU copies with `writeable=False`. With
  frozen dataclasses, a snapshot can be shared with another thread while training
  continues.
* With `eq=False`, `LayerWeights` and `WeightSnapshot` keep `object`'s identity-based
  `==` and hash. `WeightStats` compares by value.

## Used by / uses

* Uses numpy only.
* Produced by [Tracking Snapshots](Tracking-Snapshots) through
  [Tracking Network Tracker](Tracking-Network-Tracker).
* Carried by `SessionStarted`, `EpochCompleted`, `TrainingPaused` and `SessionFinished`
  (`weights`, which can be `None` in `SessionFinished`), see [Sync Events](Sync-Events).
  Held by [Sync Session State](Sync-Session-State). Returned as
  `TrainingResult.final_weights` ([Training Results](Training-Results)).
* Read by the `weights [layer]` command ([Built-in Commands](Commands-Builtin)) and the
  [Network View](GUI-Network-View) (line colours from `matrix`, glow from `delta`, neuron
  tint from `bias`, columns from the indices).

## Example

```python
from torch import nn
from hollywood_training.tracking import NetworkTracker

model = nn.Sequential(nn.Linear(3, 30), nn.ReLU(), nn.Linear(30, 2))
hidden, output = NetworkTracker(model, max_neurons=8).snapshot(0).layers
hidden.matrix.shape, output.matrix.shape     # (8, 3), (2, 8)
hidden.out_index.tolist()                    # [0, 4, 8, 12, 17, 21, 25, 29] = output.in_index
hidden.truncated, hidden.stats.count         # True, 90: stats cover all 30 x 3 weights
hidden.delta, hidden.update_ratio            # None, None: no baseline yet
[(t.name, t.shape) for t in hidden.tensors]  # [('weight', (30, 3))]
```

## Notes

* `NetworkTracker` requires `max_neurons >= 2`, so tracked snapshots always include the
  first and the last neuron of each side.
* Unpickling these classes imports numpy, not torch.

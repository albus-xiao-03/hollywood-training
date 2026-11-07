# Tracking Layer Classification

`hollywood_training/tracking/layers.py` · part of [Tracking](Tracking)

This module maps `torch.nn` modules to a [LayerRole](Tracking-Layer-Roles), reads the
`(in, out)` size of connection layers and writes the short summaries shown in the drawing
and in the terminal. Only public `torch.nn` classes are consulted, and they are looked up
by name. A class missing from the installed PyTorch (for example `nn.RMSNorm` in older
releases) is skipped instead of causing an error.

It also defines the **weight tensors** of a module: the initialised parameters with two
or more dimensions that the module holds directly. Any module that holds one and is not
an operation class is a connection layer, a **weight owner**. So every weight matrix of a
network belongs to a connection layer: `Bilinear`, `RNN`, `LSTM`, `GRU`,
`MultiheadAttention` and custom modules included.

A recurrent layer (`RNN`, `LSTM`, `GRU`) is drawn as one **stage** per stacked layer,
each followed by a projection stage when `proj_size` is set. `connection_shape` and
`summarize` take the stage as keyword arguments.

## Responsibilities

* Give every module a role (`classify`). Every weight owner is a connection layer.
* List the weight tensors of a module (`weight_tensors`) and pick the one its connection
  matrix is read from (`primary_weight`).
* Report the input and output features of a connection layer (`connection_shape`).
* Describe a module in a few characters (`summarize`).
* Split a recurrent layer into stages and find each stage's parameters
  (`recurrent_stages`, `stage_name`, `recurrent_parameters`, `recurrent_tensors`,
  `stage_weight_tensors`, `gate_count`).
* Recognise the placeholder parameters of lazy modules (`is_uninitialized`, `weight_of`).

## API

### Class tuples

Each constant is a `tuple[type[nn.Module], ...]`. All except `CONNECTION_LAYERS` are
built by the private `_public_classes(*names)`, which keeps `getattr(nn, name)` when it
exists and is an `nn.Module` subclass. Checks use `isinstance`, so subclasses are covered
too, including the lazy variants (`LazyLinear`, `LazyConv2d`, ...).

| Constant | Classes looked up |
|---|---|
| `LINEAR_LAYERS` | `Linear` |
| `CONVOLUTION_LAYERS` | `Conv1d`, `Conv2d`, `Conv3d` |
| `TRANSPOSED_CONVOLUTION_LAYERS` | `ConvTranspose1d`, `ConvTranspose2d`, `ConvTranspose3d` |
| `EMBEDDING_LAYERS` | `Embedding`, `EmbeddingBag` |
| `BILINEAR_LAYERS` | `Bilinear` |
| `ATTENTION_LAYERS` | `MultiheadAttention` |
| `RECURRENT_LAYERS` | `RNN`, `LSTM`, `GRU` |
| `RECURRENT_CELLS` | `RNNCell`, `LSTMCell`, `GRUCell` |
| `ACTIVATION_LAYERS` | `ReLU`, `ReLU6`, `LeakyReLU`, `PReLU`, `RReLU`, `ELU`, `SELU`, `CELU`, `GELU`, `SiLU`, `Mish`, `Sigmoid`, `LogSigmoid`, `Hardsigmoid`, `Tanh`, `Hardtanh`, `Tanhshrink`, `Hardswish`, `Softplus`, `Softsign`, `Softshrink`, `Hardshrink`, `Threshold`, `GLU`, `Softmax`, `Softmin`, `LogSoftmax`, `Softmax2d` |
| `NORMALIZATION_LAYERS` | `BatchNorm1d`, `BatchNorm2d`, `BatchNorm3d`, `SyncBatchNorm`, `LayerNorm`, `GroupNorm`, `InstanceNorm1d`, `InstanceNorm2d`, `InstanceNorm3d`, `LocalResponseNorm`, `RMSNorm`, `CrossMapLRN2d` |
| `REGULARIZATION_LAYERS` | `Dropout`, `Dropout1d`, `Dropout2d`, `Dropout3d`, `AlphaDropout`, `FeatureAlphaDropout` |
| `POOLING_LAYERS` | `MaxPool1d/2d/3d`, `AvgPool1d/2d/3d`, `AdaptiveMaxPool1d/2d/3d`, `AdaptiveAvgPool1d/2d/3d`, `LPPool1d/2d/3d`, `FractionalMaxPool2d/3d` |
| `RESHAPE_LAYERS` | `Flatten`, `Unflatten`, `Upsample`, `PixelShuffle`, `PixelUnshuffle` |
| `PASS_THROUGH_LAYERS` | `Identity` |

```python
LINEAR_LAYERS = _public_classes("Linear")      # every tuple above is built this way
CONNECTION_LAYERS = (
    LINEAR_LAYERS + CONVOLUTION_LAYERS + TRANSPOSED_CONVOLUTION_LAYERS + EMBEDDING_LAYERS
    + BILINEAR_LAYERS + ATTENTION_LAYERS + RECURRENT_LAYERS + RECURRENT_CELLS
)
OPERATION_ROLES = frozenset({LayerRole.ACTIVATION, LayerRole.NORMALIZATION,
                             LayerRole.REGULARIZATION, LayerRole.POOLING, LayerRole.RESHAPE})
```

### Functions

```python
def is_uninitialized(tensor: object) -> bool
def weight_of(module: nn.Module) -> nn.Parameter | None
def has_weight_matrix(module: nn.Module) -> bool
def weight_tensors(module: nn.Module) -> list[tuple[str, nn.Parameter]]
def primary_weight(module: nn.Module) -> nn.Parameter | None
def attention_projections(module: nn.Module) -> list[tuple[str, nn.Parameter]]
def is_pass_through(module: nn.Module) -> bool
```

* `is_uninitialized`: `True` for `nn.parameter.UninitializedParameter` and
  `UninitializedBuffer`, the placeholders that lazy modules hold until their first forward
  pass.
* `weight_of`: `module.weight` when it is an initialised `nn.Parameter`, otherwise `None`.
  A `weight` that is a plain tensor or a buffer counts as no weight.
* `has_weight_matrix`: `weight_of(module)` exists and is 2-D, i.e. shaped `(out, in)`.
  `classify` does not use it: it uses `weight_tensors`.
* `weight_tensors`: `(name, parameter)` for every initialised parameter with two or more
  dimensions in `module.named_parameters(recurse=False)`, in registration order. For
  example `[('weight_ih_l0', ...), ('weight_hh_l0', ...)]` for `nn.GRU(3, 5)`, and
  `[('in_proj_weight', ...)]` for `nn.MultiheadAttention(8, 2)`: its `out_proj` is a child
  with weights of its own.
* `primary_weight`: the tensor a connection layer is read from: `weight_of(module)` when
  it has two or more dimensions, otherwise the first weight tensor, or `None` when there
  is none (yet).
* `attention_projections`: the query, key and value input projections of an attention
  layer: `[('in_proj_weight', W)]` when they share one `(3E, E)` tensor, otherwise
  `q_proj_weight`, `k_proj_weight` and `v_proj_weight` (PyTorch keeps them separate when
  `kdim` or `vdim` differs from `embed_dim`). Empty for any other module.
* `is_pass_through`: `isinstance(module, PASS_THROUGH_LAYERS)`. Such modules are left out
  of the drawing.

```python
def classify(module: nn.Module) -> LayerRole
```

Checks run in this order, and the first match wins: `CONNECTION_LAYERS` → `CONNECTION`,
`ACTIVATION_LAYERS`, `NORMALIZATION_LAYERS`, `REGULARIZATION_LAYERS`, `POOLING_LAYERS`,
`RESHAPE_LAYERS`, then a non-empty `weight_tensors(module)` → `CONNECTION` (any other
weight owner, e.g. a custom module), and otherwise `OTHER`. It works on any module, with
or without children.

### Recurrent layers

```python
def gate_count(module: nn.Module) -> int
def recurrent_stages(module: nn.Module) -> list[tuple[int, bool]]
def stage_name(name: str, stage: int = 0, projection: bool = False) -> str
def recurrent_parameters(
    module: nn.Module, stage: int = 0, projection: bool = False
) -> list[tuple[str, nn.Parameter]]
def recurrent_tensors(module: nn.Module, kind: str, stage: int = 0) -> list[nn.Parameter]
def stage_weight_tensors(
    module: nn.Module, stage: int = 0, projection: bool = False
) -> list[tuple[str, nn.Parameter]]
```

* `gate_count`: the gate blocks stacked along the rows of a recurrent weight: 4 for
  `LSTM` (input, forget, cell, output), 3 for `GRU` (reset, update, new), 1 for `RNN`,
  their cells alike.
* `recurrent_stages`: `(stacked layer, projection)` of every drawn stage of an `RNN`,
  `LSTM` or `GRU`, in the order data flows: `[(0, False), (1, False)]` for two stacked
  layers, `[(0, False), (0, True), (1, False), (1, True)]` with `proj_size`. Any other
  module, cells included, is the single stage `[(0, False)]`.
* `stage_name`: `rnn` for stacked layer 0, `rnn.l1` for layer 1, plus `.proj` for a
  projection (`rnn.proj`, `rnn.l1.proj`). A recurrent module has no children, so these
  names never collide with module paths.
* `recurrent_parameters`: the parameters of one stage, every direction in registration
  order: `weight_ih`, `weight_hh`, `bias_ih` and `bias_hh` of the stacked layer (with
  the `_l<k>` and `_reverse` suffixes), or its `weight_hr` for a projection stage. All
  parameters of a cell. Empty for other modules.
* `recurrent_tensors`: one kind (`weight_ih`, `weight_hh`, `weight_hr`, `bias_ih`,
  `bias_hh`) of a stacked layer, one parameter per direction, forward first. Empty when
  the module has none, e.g. with `bias=False`.
* `stage_weight_tensors`: the weight tensors of one drawn stage: the stage's parameters
  with two or more dimensions for a recurrent layer or cell, otherwise
  `weight_tensors(module)`.

```python
def connection_shape(
    module: nn.Module, *, stage: int = 0, projection: bool = False
) -> tuple[int | None, int | None]
```

Returns `(in_features, out_features)`. `stage` and `projection` select a stage of a
recurrent layer; every other module has only stage 0. Below, `D` is 2 for a
bidirectional layer and 1 otherwise, `H` is `hidden_size` and `P` is `proj_size`:

| Module | Result |
|---|---|
| no weight tensor (e.g. a lazy module before its first pass) | `(None, None)` |
| `Embedding`, `EmbeddingBag` | `(num_embeddings, embedding_dim)`, from `weight.shape` |
| `Conv*`, `ConvTranspose*` | `(in_channels, out_channels)` |
| `Bilinear` | `(in1_features + in2_features, out_features)`: both inputs, x1 first |
| `MultiheadAttention` | `(embed_dim, 3 * embed_dim)`: the outputs are the query, key and value projections. With separate projections the inputs are the query, key and value features: `(embed_dim + kdim + vdim, 3 * embed_dim)` |
| `RNN`, `LSTM`, `GRU` stage `k` | `(input_size, D·H)` for `k = 0`, `(D·H, D·H)` above it, or `(D·P, D·H)` when the layers are projected: the stacked layer's input and its hidden state, forward units first |
| their projection stage | `(D·H, D·P)` |
| `RNNCell`, `LSTMCell`, `GRUCell` | `(input_size, hidden_size)` |
| any other weight owner: `primary_weight` read as `(out, in, ...)` | `(shape[1], shape[0])`, e.g. `(in, out)` for a 2-D `weight` |

```python
def summarize(module: nn.Module, *, stage: int = 0, projection: bool = False) -> str
```

| Case | Format | Example |
|---|---|---|
| connection layer of unknown shape | `{name} (uninitialised)` | `LazyLinear (uninitialised)` |
| `Bilinear` | `{name} {in1}+{in2}→{out}` | `Bilinear 3+4→2` |
| `MultiheadAttention` | `{name} {in}→{out} h{num_heads}`, with `{in}` written `{embed_dim}+{kdim}+{vdim}` for separate projections | `MultiheadAttention 8→24 h2`, `MultiheadAttention 8+6+4→24 h2` |
| recurrent stage | `{name} {in}→{out}`, a projection `{name} projection {in}→{out}`. Both directions are written `2×H`. ` L{k+1}/{num_layers}` follows when there are several stacked layers | `LSTM 3→5`, `GRU 3→2×5`, `RNN 5→5 L2/3`, `LSTM projection 2×5→2×2 L1/2`, `LSTMCell 3→5` |
| connection layer | `{name} {in}→{out}`, plus ` k{kernel}` when the module has `kernel_size` (tuple parts joined by `x`) | `Linear 2→16`, `Conv2d 3→8 k3x3` |
| regularization | `{name}(p={p:g})` | `Dropout(p=0.2)` |
| `LeakyReLU` | `{name}({negative_slope:g})` | `LeakyReLU(0.1)` |
| normalization | `{name}({size})`, with `size` from `num_features`, else `normalized_shape` (tuple parts joined by `x`). Bare name when neither exists | `BatchNorm1d(16)`, `LayerNorm(4x5)`, `GroupNorm` |
| anything else | class name | `ReLU`, `MaxPool2d` |

## How it works

All tuples are built once, at import time, so the role table matches the installed
PyTorch. `classify` decides by class membership. The only exception is the weight-tensor
fallback, which looks at parameters and runs last. A module with a weight tensor that is
listed elsewhere therefore keeps its listed role: `LayerNorm((4, 5))` and `RMSNorm` stay
`NORMALIZATION`. Their weight scales each feature on its own and connects no neurons.

## Used by / uses

* Uses `torch.nn` and [Tracking Layer Roles](Tracking-Layer-Roles).
* [Tracking Inspector](Tracking-Inspector): `classify`, `connection_shape`, `summarize`,
  `is_pass_through`, `is_uninitialized`, `OPERATION_ROLES`, `recurrent_stages`,
  `stage_name`, `recurrent_parameters`.
* [Tracking Snapshots](Tracking-Snapshots): `primary_weight`, `attention_projections`,
  `gate_count`, `recurrent_stages`, `recurrent_tensors`, `stage_weight_tensors`,
  `connection_shape`, `is_uninitialized`, and the convolution, transposed convolution,
  embedding, bilinear, attention and recurrent tuples.
* [Training Descriptions](Training-Descriptions): `is_uninitialized`, to find the model's
  device.
* Lazily exported: `classify`, `connection_shape`, `summarize`.

## Example

```python
import torch
from torch import nn
from hollywood_training.tracking import classify, connection_shape, summarize

conv = nn.Conv2d(3, 8, 3)
classify(conv), connection_shape(conv), summarize(conv)
# (<LayerRole.CONNECTION: 'connection'>, (3, 8), 'Conv2d 3→8 k3x3')

class Projection(nn.Module):              # custom linear-like module
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.randn(4, 3))

connection_shape(Projection()), summarize(Projection())   # ((3, 4), 'Projection 3→4')

attention = nn.MultiheadAttention(8, 2)   # a weight owner with a child (out_proj)
classify(attention), summarize(attention)
# (<LayerRole.CONNECTION: 'connection'>, 'MultiheadAttention 8→24 h2')

from hollywood_training.tracking.layers import recurrent_stages, stage_name

lstm = nn.LSTM(3, 5, num_layers=2, bidirectional=True)
[(stage_name("rnn", k, p), summarize(lstm, stage=k, projection=p))
 for k, p in recurrent_stages(lstm)]
# [('rnn', 'LSTM 3→2×5 L1/2'), ('rnn.l1', 'LSTM 2×5→2×5 L2/2')]
```

## Notes

* A lazy module becomes its regular class after its first forward pass (PyTorch
  behaviour). For `nn.LazyLinear(5)` fed 7 features, a rebuilt summary reads `Linear 7→5`
  instead of `LazyLinear (uninitialised)`.
* Embedding tables are oriented vocabulary → embedding: `EmbeddingBag(10, 3)` has
  `connection_shape` `(10, 3)` and the summary `EmbeddingBag 10→3`. Any other weight
  owner is read from its primary weight as `(out, in, ...)`, like `Linear`. Further
  dimensions are averaged in the snapshot (see [Tracking Snapshots](Tracking-Snapshots)).

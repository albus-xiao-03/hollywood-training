# Tracking Layer Classification

`hollywood_training/tracking/layers.py` · part of [Tracking](Tracking)

This module maps `torch.nn` modules to a [LayerRole](Tracking-Layer-Roles), reads the
`(in, out)` size of connection layers and writes the short summaries shown in the drawing
and in the terminal. Only public `torch.nn` classes are consulted, and they are looked up
by name. A class missing from the installed PyTorch (for example `nn.RMSNorm` in older
releases) is skipped instead of causing an error.

## Responsibilities

* Give every leaf module a role (`classify`).
* Report the input and output features of a connection layer (`connection_shape`).
* Describe a module in a few characters (`summarize`).
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
)
OPERATION_ROLES = frozenset({LayerRole.ACTIVATION, LayerRole.NORMALIZATION,
                             LayerRole.REGULARIZATION, LayerRole.POOLING, LayerRole.RESHAPE})
```

### Functions

```python
def is_uninitialized(tensor: object) -> bool
def weight_of(module: nn.Module) -> nn.Parameter | None
def has_weight_matrix(module: nn.Module) -> bool
def is_pass_through(module: nn.Module) -> bool
```

* `is_uninitialized`: `True` for `nn.parameter.UninitializedParameter` and
  `UninitializedBuffer`, the placeholders that lazy modules hold until their first forward
  pass.
* `weight_of`: `module.weight` when it is an initialised `nn.Parameter`, otherwise `None`.
  A `weight` that is a plain tensor or a buffer counts as no weight.
* `has_weight_matrix`: `weight_of(module)` exists and is 2-D, i.e. shaped `(out, in)`.
* `is_pass_through`: `isinstance(module, PASS_THROUGH_LAYERS)`. Such modules are left out
  of the drawing.

```python
def classify(module: nn.Module) -> LayerRole
```

Checks run in this order, and the first match wins: `CONNECTION_LAYERS` → `CONNECTION`,
`ACTIVATION_LAYERS`, `NORMALIZATION_LAYERS`, `REGULARIZATION_LAYERS`, `POOLING_LAYERS`,
`RESHAPE_LAYERS`, then `has_weight_matrix(module)` → `CONNECTION` (custom linear-like
modules), and otherwise `OTHER`.

```python
def connection_shape(module: nn.Module) -> tuple[int | None, int | None]
```

Returns `(in_features, out_features)`:

| Module | Result |
|---|---|
| no initialised `weight` parameter (e.g. a lazy module before its first pass) | `(None, None)` |
| `Embedding`, `EmbeddingBag` | `(num_embeddings, embedding_dim)`, from `weight.shape` |
| `Conv*`, `ConvTranspose*` | `(in_channels, out_channels)` |
| any other 2-D `weight` of shape `(out, in)` | `(in, out)` |
| anything else | `(None, None)` |

```python
def summarize(module: nn.Module) -> str
```

| Case | Format | Example |
|---|---|---|
| connection layer of unknown shape | `{name} (uninitialised)` | `LazyLinear (uninitialised)` |
| connection layer | `{name} {in}→{out}`, plus ` k{kernel}` when the module has `kernel_size` (tuple parts joined by `x`) | `Linear 2→16`, `Conv2d 3→8 k3x3` |
| regularization | `{name}(p={p:g})` | `Dropout(p=0.2)` |
| `LeakyReLU` | `{name}({negative_slope:g})` | `LeakyReLU(0.1)` |
| normalization | `{name}({size})`, with `size` from `num_features`, else `normalized_shape` (tuple parts joined by `x`). Bare name when neither exists | `BatchNorm1d(16)`, `LayerNorm(4x5)`, `GroupNorm` |
| anything else | class name | `ReLU`, `MaxPool2d` |

## How it works

All tuples are built once, at import time, so the role table matches the installed
PyTorch. `classify` decides by class membership. The only exception is the 2-D-weight
fallback, which looks at parameters and runs last. A module with a 2-D weight that is
listed elsewhere therefore keeps its listed role: `LayerNorm((4, 5))` stays
`NORMALIZATION`.

## Used by / uses

* Uses `torch.nn` and [Tracking Layer Roles](Tracking-Layer-Roles).
* [Tracking Inspector](Tracking-Inspector): `classify`, `connection_shape`, `summarize`,
  `is_pass_through`, `is_uninitialized`, `OPERATION_ROLES`.
* [Tracking Snapshots](Tracking-Snapshots): `weight_of`, `is_uninitialized`, and the
  convolution, transposed convolution and embedding tuples.
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
```

## Notes

* A lazy module becomes its regular class after its first forward pass (PyTorch
  behaviour). For `nn.LazyLinear(5)` fed 7 features, a rebuilt summary reads `Linear 7→5`
  instead of `LazyLinear (uninitialised)`.
* Embedding tables are oriented vocabulary → embedding: `EmbeddingBag(10, 3)` has
  `connection_shape` `(10, 3)` and the summary `EmbeddingBag 10→3`. A custom module with a
  2-D `weight` is always read as `(out, in)`, like `Linear`.

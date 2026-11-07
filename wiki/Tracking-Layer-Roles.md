# Tracking Layer Roles

`hollywood_training/tracking/roles.py` · part of [Tracking](Tracking)

`LayerRole` names what a module contributes to the drawn network: a weight matrix
between two columns of neurons, an operation listed under a layer, or nothing that can be
drawn. It has its own module that imports only `enum`. `OperationInfo.role` holds a
`LayerRole`, so this lets the window process, which imports no torch, unpickle an
`ArchitectureInfo`.

## Responsibilities

* Define the fixed set of roles shared by classification, the architecture model and the
  frontends.
* Stay torch-free, so plain data that contains a role can cross the process boundary.

## API

```python
class LayerRole(str, Enum):
    CONNECTION = "connection"  # owns a weight matrix linking input features to output features
    ACTIVATION = "activation"
    NORMALIZATION = "normalization"
    REGULARIZATION = "regularization"
    POOLING = "pooling"
    RESHAPE = "reshape"
    OTHER = "other"
```

`LayerRole` mixes in `str`: a member compares equal to its value
(`LayerRole.CONNECTION == "connection"`), and `LayerRole("pooling")` returns
`LayerRole.POOLING`.

| Member | Assigned by `classify()` to ([Tracking Layer Classification](Tracking-Layer-Classification)) | Effect in `build_architecture()` |
|---|---|---|
| `CONNECTION` | `Linear`, `Conv1d/2d/3d`, `ConvTranspose1d/2d/3d`, `Embedding`, `EmbeddingBag`, `Bilinear`, `MultiheadAttention`, `RNN`, `LSTM`, `GRU`, `RNNCell`, `LSTMCell`, `GRUCell` and their subclasses. Also any module outside the lists below that holds a weight tensor (an initialised parameter with two or more dimensions), e.g. a custom module | becomes a `TrackedLayer`, a set of connections in the drawing |
| `ACTIVATION` | `ReLU`, `GELU`, `Tanh`, `Softmax`, ... (28 class names) | `LayerOperation` attached to the current layer |
| `NORMALIZATION` | `BatchNorm*`, `LayerNorm`, `GroupNorm`, `RMSNorm`, ... (12 class names) | `LayerOperation` |
| `REGULARIZATION` | `Dropout`, `Dropout1d/2d/3d`, `AlphaDropout`, `FeatureAlphaDropout` | `LayerOperation` |
| `POOLING` | `MaxPool*`, `AvgPool*`, `AdaptiveMaxPool*`, `AdaptiveAvgPool*`, `LPPool*`, `FractionalMaxPool*` | `LayerOperation` |
| `RESHAPE` | `Flatten`, `Unflatten`, `Upsample`, `PixelShuffle`, `PixelUnshuffle` | `LayerOperation` |
| `OTHER` | everything else: modules without a weight tensor, e.g. `nn.ZeroPad2d`, `nn.CosineSimilarity` | label added to `unsupported`, not drawn |

The five operation roles are collected in `OPERATION_ROLES` (a `frozenset` in
`layers.py`). An operation attaches to the most recent connection layer in the module
order, or to the network input (`input_operations`) when no connection layer comes
before it.

## How it works

* `classify()` assigns the role, using `isinstance` checks against public `torch.nn`
  classes. This module only defines the values.
* `nn.Identity` is a pass-through. `build_architecture()` skips it before classifying, so
  it never shows up as `OTHER` or as unsupported.
* Frontends work with roles through `ArchitectureInfo`:
  * `LayerInfo.activation` is the summary of the first `ACTIVATION` operation.
  * `ArchitectureInfo.normalization_layers()` and `normalization_summary()` collect the
    `NORMALIZATION` operations (see
    [Tracking Architecture Model](Tracking-Architecture-Model)).

## Used by / uses

* Uses only `enum`.
* [Tracking Layer Classification](Tracking-Layer-Classification) (`classify`,
  `OPERATION_ROLES`) and [Tracking Inspector](Tracking-Inspector), which imports
  `LayerRole` through `layers.py`.
* [Tracking Architecture Model](Tracking-Architecture-Model): `OperationInfo.role` and
  `LayerOperation.role`.
* Exported eagerly as `hollywood_training.tracking.LayerRole`.

## Example

```python
from torch import nn
from hollywood_training.tracking import LayerRole, build_architecture, classify

classify(nn.Dropout(0.1)) is LayerRole.REGULARIZATION      # True
LayerRole("pooling") is LayerRole.POOLING                  # True

model = nn.Sequential(nn.Linear(2, 8), nn.Tanh(), nn.Linear(8, 3))
info = build_architecture(model).describe()
[op.name for op in info.operations() if op.role is LayerRole.ACTIVATION]   # ['1']
```

## Notes

* A role states what the module contributes to the drawing: `Softmax` is an activation,
  `Upsample` a reshape.
* A role comes from the module's class, or for other modules from their weight tensors.
  It is never read from `forward`. An activation written as a function (`torch.relu(x)`)
  is not a module, so it has no role and does not appear.

# Tracking Architecture Model

`hollywood_training/tracking/architecture.py` · part of [Tracking](Tracking)

These slot dataclasses describe a tracked network as its connection layers plus the
operations (activation, normalisation, ...) applied between them. The live classes keep
references to the modules, so the training thread can read weights through them. Their
`describe()` methods produce the immutable, module-free `*Info` classes, which are safe to
hand to a frontend on another thread or in another process.

## Responsibilities

* Hold the structure that [build_architecture](Tracking-Inspector) produces.
* Convert it into plain data with `describe()`: `NetworkArchitecture` → `ArchitectureInfo`,
  `TrackedLayer` → `LayerInfo`, `LayerOperation` → `OperationInfo`.
* Answer the small queries frontends need: a layer by name, the first activation of a
  layer, and the normalisation summary.

## API

```python
@dataclass(frozen=True, slots=True)
class OperationInfo:
    name: str
    role: LayerRole
    type_name: str
    summary: str

@dataclass(frozen=True, slots=True)
class LayerInfo:
    name: str
    type_name: str
    summary: str
    in_features: int | None
    out_features: int | None
    parameters: int
    operations: tuple[OperationInfo, ...] = ()
    @property
    def activation(self) -> str | None: ...

@dataclass(frozen=True, slots=True)
class ArchitectureInfo:
    class_name: str
    display_name: str | None
    layers: tuple[LayerInfo, ...]
    input_operations: tuple[OperationInfo, ...] = ()
    unsupported: tuple[str, ...] = ()
    total_parameters: int = 0
    trainable_parameters: int = 0
    input_labels: tuple[str, ...] | None = None
    output_labels: tuple[str, ...] | None = None
    traced: bool = False
    def operations(self) -> tuple[OperationInfo, ...]: ...
    def normalization_layers(self) -> tuple[str, ...]: ...
    def normalization_summary(self) -> str: ...

@dataclass(slots=True)
class LayerOperation:
    name: str
    role: LayerRole
    module: nn.Module
    summary: str
    @property
    def type_name(self) -> str: ...
    def describe(self) -> OperationInfo: ...

@dataclass(slots=True)
class TrackedLayer:
    name: str
    module: nn.Module
    summary: str
    in_features: int | None
    out_features: int | None
    parameters: int
    operations: list[LayerOperation] = field(default_factory=list)
    stage: int = 0                     # recurrent layers: the stacked layer
    projection: bool = False           # recurrent layers: the proj_size projection
    @property
    def type_name(self) -> str: ...
    def describe(self) -> LayerInfo: ...

@dataclass(slots=True)
class NetworkArchitecture:
    class_name: str
    display_name: str | None
    layers: list[TrackedLayer]
    input_operations: list[LayerOperation] = field(default_factory=list)
    unsupported: list[str] = field(default_factory=list)
    total_parameters: int = 0
    trainable_parameters: int = 0
    input_labels: tuple[str, ...] | None = None
    output_labels: tuple[str, ...] | None = None
    traced: bool = False
    def layer(self, name: str) -> TrackedLayer | None: ...
    def describe(self) -> ArchitectureInfo: ...
```

| Member | Returns |
|---|---|
| `LayerInfo.activation` | `summary` of the first operation with role `ACTIVATION`, or `None` |
| `ArchitectureInfo.operations()` | every operation in drawing order, input operations first |
| `ArchitectureInfo.normalization_layers()` | `type_name` of each `NORMALIZATION` operation, in drawing order |
| `ArchitectureInfo.normalization_summary()` | counts of those names in first-seen order, e.g. `BatchNorm1d ×2, LayerNorm`, or `none` when there are none |
| `LayerOperation.type_name`, `TrackedLayer.type_name` | `type(self.module).__name__`, read on every access |
| `NetworkArchitecture.layer(name)` | first connection layer called `name`, or `None` |

## How it works

* torch is imported only under `TYPE_CHECKING`, and annotations are not evaluated
  (`from __future__ import annotations`). Importing the module, or unpickling `*Info`
  objects, therefore needs only [LayerRole](Tracking-Layer-Roles).
* `describe()` copies every field. Lists become tuples, layers and operations are
  described recursively, and `type_name` is read from the module at that moment.
* Values as set by `build_architecture` ([Tracking Inspector](Tracking-Inspector)):

| Field | Value |
|---|---|
| `class_name` | `type(network).__name__` of the unwrapped model |
| `display_name`, `input_labels`, `output_labels` | from `TrackOptions` ([Tracking Decorator](Tracking-Decorator)) |
| `layers`, `input_operations` | connection layers in drawing order, and the operations that come before the first of them |
| `unsupported` | labels of leaf modules with role `OTHER`, without repeats |
| `total_parameters`, `trainable_parameters` | initialised parameter elements of the whole network, and of those with `requires_grad` |
| `traced` | `True` when built from an explicit `order`, i.e. a recorded forward pass |
| `name` (layer, operation) | qualified module name such as `enc.0`, or the class name when the model itself is the leaf or the weight owner. The stages of a recurrent module add `.l<k>` for stacked layer `k ≥ 1` and `.proj` for a projection: `rnn`, `rnn.proj`, `rnn.l1`, `rnn.l1.proj` |
| `stage`, `projection` (live layer) | the stacked layer and whether it is the `proj_size` projection, for the stages of a recurrent module. `0` and `False` for every other layer. Not part of `LayerInfo`: the stage is in the name |
| `summary`, `in_features`, `out_features` | `summarize()` and `connection_shape()` ([Tracking Layer Classification](Tracking-Layer-Classification)) |
| `parameters` (layer) | initialised parameter elements of that module (weight and bias). A weight owner with children counts only the parameters it holds itself, and a recurrent stage only its own (`weight_ih`, `weight_hh` and the biases of its stacked layer, or `weight_hr` for a projection) |
| `operations` (layer) | operations applied to its output, in execution order |

## Used by / uses

* Uses [Tracking Layer Roles](Tracking-Layer-Roles) only.
* Live classes: built by [Tracking Inspector](Tracking-Inspector), held by
  [Tracking Network Tracker](Tracking-Network-Tracker), read by
  [Tracking Snapshots](Tracking-Snapshots).
* `ArchitectureInfo`: carried by `SessionInfo.architecture` and
  `EpochCompleted.architecture` ([Sync Events](Sync-Events)), held by
  [Sync Session State](Sync-Session-State), and read by `describe_normalization`
  ([Training Descriptions](Training-Descriptions)), the `arch` command
  ([Built-in Commands](Commands-Builtin)) and the [Network View](GUI-Network-View).

## Example

```python
from torch import nn
from hollywood_training.tracking import build_architecture

model = nn.Sequential(nn.Flatten(), nn.Linear(4, 8), nn.BatchNorm1d(8), nn.ReLU(),
                      nn.Linear(8, 2), nn.ZeroPad1d(1))
info = build_architecture(model).describe()     # NetworkArchitecture -> ArchitectureInfo
[layer.summary for layer in info.layers]        # ['Linear 4→8', 'Linear 8→2']
[op.summary for op in info.input_operations]    # ['Flatten']
info.layers[0].activation                       # 'ReLU'
info.normalization_summary()                    # 'BatchNorm1d'
info.unsupported                                # ('5',): the padding is not drawn
```

## Notes

* Only `*Info` objects belong in events. The live classes reference modules, and through
  them tensors, so use them on the training thread.
* Live `summary`, feature sizes and `parameters` are fixed at build time. Only `type_name`
  follows the module, so a lazy layer stays `(uninitialised)` until the trace rebuilds it.

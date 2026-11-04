"""Data model of a tracked network.

Slot dataclasses hold the network's connection layers (``Linear``, ``Conv*``, ...) together
with the activation, normalisation and other operations applied between them.

* :class:`NetworkArchitecture`, :class:`TrackedLayer` and :class:`LayerOperation` keep live
  references to the modules; the training thread reads weights through them.
* :meth:`NetworkArchitecture.describe` produces the immutable, module-free
  :class:`ArchitectureInfo` that is safe to hand to a frontend running on another thread.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from .roles import LayerRole

if TYPE_CHECKING:  # the module-free *Info classes must not need torch at runtime
    from torch import nn


@dataclass(frozen=True, slots=True)
class OperationInfo:
    """Module-free description of an operation between connection layers."""

    name: str
    role: LayerRole
    type_name: str
    summary: str


@dataclass(frozen=True, slots=True)
class LayerInfo:
    """Module-free description of one connection layer."""

    name: str
    type_name: str
    summary: str
    in_features: int | None
    out_features: int | None
    parameters: int
    operations: tuple[OperationInfo, ...] = ()

    @property
    def activation(self) -> str | None:
        """Summary of the first activation applied to this layer's output, if any."""
        for operation in self.operations:
            if operation.role is LayerRole.ACTIVATION:
                return operation.summary
        return None


@dataclass(frozen=True, slots=True)
class ArchitectureInfo:
    """Module-free description of a whole network, as shown by frontends."""

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

    def operations(self) -> tuple[OperationInfo, ...]:
        """Every operation in drawing order (input operations first)."""
        ops = list(self.input_operations)
        for layer in self.layers:
            ops.extend(layer.operations)
        return tuple(ops)

    def normalization_layers(self) -> tuple[str, ...]:
        """Type names of the normalisation operations, in drawing order."""
        return tuple(op.type_name for op in self.operations() if op.role is LayerRole.NORMALIZATION)

    def normalization_summary(self) -> str:
        """Compact text such as ``BatchNorm1d ×2, LayerNorm`` (``none`` when absent)."""
        counts = Counter(self.normalization_layers())
        if not counts:
            return "none"
        return ", ".join(f"{name} ×{n}" if n > 1 else name for name, n in counts.items())


@dataclass(slots=True)
class LayerOperation:
    """An activation / normalisation / dropout / pooling / reshape module in the network."""

    name: str
    role: LayerRole
    module: nn.Module
    summary: str

    @property
    def type_name(self) -> str:
        return type(self.module).__name__

    def describe(self) -> OperationInfo:
        return OperationInfo(self.name, self.role, self.type_name, self.summary)


@dataclass(slots=True)
class TrackedLayer:
    """A connection layer and the operations applied to its output, in execution order."""

    name: str
    module: nn.Module
    summary: str
    in_features: int | None
    out_features: int | None
    parameters: int
    operations: list[LayerOperation] = field(default_factory=list)

    @property
    def type_name(self) -> str:
        return type(self.module).__name__

    def describe(self) -> LayerInfo:
        return LayerInfo(
            name=self.name,
            type_name=self.type_name,
            summary=self.summary,
            in_features=self.in_features,
            out_features=self.out_features,
            parameters=self.parameters,
            operations=tuple(op.describe() for op in self.operations),
        )


@dataclass(slots=True)
class NetworkArchitecture:
    """The structure of a tracked network, with live module references."""

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

    def layer(self, name: str) -> TrackedLayer | None:
        """The connection layer called ``name``, if present."""
        return next((layer for layer in self.layers if layer.name == name), None)

    def describe(self) -> ArchitectureInfo:
        return ArchitectureInfo(
            class_name=self.class_name,
            display_name=self.display_name,
            layers=tuple(layer.describe() for layer in self.layers),
            input_operations=tuple(op.describe() for op in self.input_operations),
            unsupported=tuple(self.unsupported),
            total_parameters=self.total_parameters,
            trainable_parameters=self.trainable_parameters,
            input_labels=self.input_labels,
            output_labels=self.output_labels,
            traced=self.traced,
        )

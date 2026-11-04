"""Build a :class:`NetworkArchitecture` from a module.

Two sources of layer order are supported, both through official PyTorch APIs:

* **static**: the registration order reported by ``named_modules()``. Available
  immediately, right for most ``__init__``-then-``forward`` models.
* **traced**: the order in which leaf modules actually ran during one forward pass,
  recorded with ``register_forward_hook`` by :class:`ForwardTrace`. This catches modules
  that are registered in a different order than they run, activation modules that are
  reused, and the shapes of lazy modules.
"""

from __future__ import annotations

from collections.abc import Sequence

from torch import nn

from .architecture import LayerOperation, NetworkArchitecture, TrackedLayer
from .decorator import TrackOptions
from .layers import (
    OPERATION_ROLES,
    LayerRole,
    classify,
    connection_shape,
    is_pass_through,
    is_uninitialized,
    summarize,
)

ModuleSequence = Sequence[tuple[str, nn.Module]]

_WRAPPERS: tuple[type[nn.Module], ...] = (nn.DataParallel, nn.parallel.DistributedDataParallel)


def unwrap(model: nn.Module) -> nn.Module:
    """The network inside ``nn.DataParallel`` / ``DistributedDataParallel`` wrappers."""
    while isinstance(model, _WRAPPERS):
        model = model.module
    return model


def leaf_modules(model: nn.Module) -> list[tuple[str, nn.Module]]:
    """``(qualified_name, module)`` for every module without children, in registration order."""
    return [
        (name, module)
        for name, module in unwrap(model).named_modules()
        if next(module.children(), None) is None
    ]


def count_parameters(module: nn.Module, *, trainable_only: bool = False) -> int:
    """Number of (initialised) parameter elements in ``module``."""
    return sum(
        parameter.numel()
        for parameter in module.parameters()
        if not is_uninitialized(parameter) and (parameter.requires_grad or not trainable_only)
    )


def build_architecture(
    model: nn.Module,
    options: TrackOptions | None = None,
    order: ModuleSequence | None = None,
) -> NetworkArchitecture:
    """Group the leaf modules of ``model`` into connection layers and their operations.

    Args:
        model: the network (wrappers such as ``nn.DataParallel`` are looked through).
        options: display options declared with ``@track``.
        order: explicit ``(name, module)`` execution order, e.g. from :class:`ForwardTrace`.
            Defaults to the static registration order.
    """
    network = unwrap(model)
    options = options or TrackOptions()
    sequence = order if order is not None else leaf_modules(network)

    layers: list[TrackedLayer] = []
    input_operations: list[LayerOperation] = []
    unsupported: list[str] = []
    seen_connections: set[int] = set()
    current: TrackedLayer | None = None

    for name, module in sequence:
        label = name or type(module).__name__
        if is_pass_through(module):
            continue
        role = classify(module)
        if role is LayerRole.CONNECTION:
            if id(module) in seen_connections:  # a module that runs twice is drawn once
                current = next(layer for layer in layers if layer.module is module)
                continue
            seen_connections.add(id(module))
            in_features, out_features = connection_shape(module)
            current = TrackedLayer(
                name=label,
                module=module,
                summary=summarize(module),
                in_features=in_features,
                out_features=out_features,
                parameters=count_parameters(module),
            )
            layers.append(current)
        elif role in OPERATION_ROLES:
            operation = LayerOperation(label, role, module, summarize(module))
            (current.operations if current is not None else input_operations).append(operation)
        elif label not in unsupported:
            unsupported.append(label)

    return NetworkArchitecture(
        class_name=type(network).__name__,
        display_name=options.name,
        layers=layers,
        input_operations=input_operations,
        unsupported=unsupported,
        total_parameters=count_parameters(network),
        trainable_parameters=count_parameters(network, trainable_only=True),
        input_labels=options.input_labels,
        output_labels=options.output_labels,
        traced=order is not None,
    )


def architecture_signature(architecture: NetworkArchitecture) -> tuple:
    """Hashable fingerprint used to detect whether a re-built architecture changed."""
    return (
        tuple(op.summary for op in architecture.input_operations),
        tuple(
            (
                layer.name,
                layer.summary,
                layer.in_features,
                layer.out_features,
                tuple(op.summary for op in layer.operations),
            )
            for layer in architecture.layers
        ),
        tuple(architecture.unsupported),
        architecture.total_parameters,
        architecture.traced,
    )


class ForwardTrace:
    """Record the leaf modules that run, in order, during the next forward pass.

    Uses ``register_forward_hook`` on every leaf module plus one on the root module to
    know when the first complete forward pass has finished; later passes are ignored.
    Call :meth:`start` before the forward pass and :meth:`stop` afterwards.
    """

    def __init__(self, model: nn.Module) -> None:
        self._model = unwrap(model)
        self._handles: list = []
        self._calls: list[tuple[str, nn.Module]] = []
        self.complete = False

    @property
    def active(self) -> bool:
        return bool(self._handles)

    def start(self) -> None:
        if self._handles:
            return
        self._calls.clear()
        self.complete = False
        for name, module in leaf_modules(self._model):
            self._handles.append(module.register_forward_hook(self._recorder(name)))
        self._handles.append(self._model.register_forward_hook(self._on_root_finished))

    def stop(self) -> list[tuple[str, nn.Module]]:
        """Remove the hooks and return the recorded ``(name, module)`` calls."""
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        return list(self._calls)

    def _recorder(self, name: str):
        def record(module: nn.Module, _inputs: object, _output: object) -> None:
            if not self.complete:
                self._calls.append((name, module))

        return record

    def _on_root_finished(self, _module: nn.Module, _inputs: object, _output: object) -> None:
        self.complete = True

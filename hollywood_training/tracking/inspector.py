"""Build a :class:`NetworkArchitecture` from a module.

The modules that can take part in the drawing are the leaf modules plus every **weight
owner** with children - a module such as ``nn.MultiheadAttention`` that holds a weight
tensor itself (:func:`static_order`). Two sources of their order are supported, both
through official PyTorch APIs:

* **static**: the registration order reported by ``named_modules()``. Available
  immediately, right for most ``__init__``-then-``forward`` models.
* **traced**: the order in which those modules actually ran during one forward pass,
  recorded with ``register_forward_hook`` (``register_forward_pre_hook`` for weight owners
  with children) by :class:`ForwardTrace`. This catches modules that are registered in a
  different order than they run, activation modules that are reused, and the shapes of
  lazy modules.

Wrappers are looked through (:func:`unwrap`): ``nn.DataParallel``,
``DistributedDataParallel`` and the ``OptimizedModule`` that ``torch.compile`` returns. The
architecture always describes the user's own network, with its own class and layer names.
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


def _compiled_original(model: nn.Module) -> nn.Module | None:
    """The network inside ``torch.compile``'s ``OptimizedModule`` wrapper, or None.

    PyTorch has no public API that unwraps a compiled model, and importing its compiler
    package for an ``isinstance`` check would load the whole compiler stack. The wrapper is
    therefore recognised by its class name, and the original network is read through the
    public ``named_children()``: the wrapper registers it as its only child, ``_orig_mod``.
    """
    if type(model).__name__ != "OptimizedModule":
        return None
    inner = dict(model.named_children()).get("_orig_mod")
    return inner if isinstance(inner, nn.Module) else None


def unwrap(model: nn.Module) -> nn.Module:
    """The network inside ``nn.DataParallel`` / ``DistributedDataParallel`` wrappers and
    inside the ``OptimizedModule`` that ``torch.compile`` returns, in any nesting."""
    while True:
        if isinstance(model, _WRAPPERS):
            model = model.module
        elif (inner := _compiled_original(model)) is not None:
            model = inner
        else:
            return model


def _has_children(module: nn.Module) -> bool:
    return next(module.children(), None) is not None


def leaf_modules(model: nn.Module) -> list[tuple[str, nn.Module]]:
    """``(qualified_name, module)`` for every module without children, in registration order."""
    return [
        (name, module)
        for name, module in unwrap(model).named_modules()
        if not _has_children(module)
    ]


def static_order(model: nn.Module) -> list[tuple[str, nn.Module]]:
    """``(qualified_name, module)`` for every module that can take part in the drawing, in
    registration order: each leaf module, and each module with children that is a
    connection layer itself - a weight owner such as ``nn.MultiheadAttention``, which holds
    ``in_proj_weight`` next to its ``out_proj`` child. A parent comes before its children."""
    return [
        (name, module)
        for name, module in unwrap(model).named_modules()
        if not _has_children(module) or classify(module) is LayerRole.CONNECTION
    ]


def count_parameters(
    module: nn.Module, *, trainable_only: bool = False, recurse: bool = True
) -> int:
    """Number of (initialised) parameter elements in ``module`` - with ``recurse=False``,
    only those it holds itself, not those of its children."""
    return sum(
        parameter.numel()
        for parameter in module.parameters(recurse=recurse)
        if not is_uninitialized(parameter) and (parameter.requires_grad or not trainable_only)
    )


def build_architecture(
    model: nn.Module,
    options: TrackOptions | None = None,
    order: ModuleSequence | None = None,
) -> NetworkArchitecture:
    """Group the modules of ``model`` into connection layers and their operations.

    Args:
        model: the network (wrappers such as ``nn.DataParallel`` are looked through).
        options: display options declared with ``@track``.
        order: explicit ``(name, module)`` execution order, e.g. from :class:`ForwardTrace`.
            Defaults to the static registration order (:func:`static_order`).
    """
    network = unwrap(model)
    options = options or TrackOptions()
    sequence = order if order is not None else static_order(network)

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
                # a weight owner with children counts its own parameters: those of a child
                # that is a layer itself (MultiheadAttention.out_proj) count there
                parameters=count_parameters(module, recurse=not _has_children(module)),
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
    """Record the modules that run, in order, during the next forward pass.

    Uses ``register_forward_hook`` on every leaf module and ``register_forward_pre_hook``
    on every weight owner with children (recorded where its forward begins, before the
    children it calls), plus one forward hook on the root module to know when the first
    complete forward pass has finished; later passes are ignored. Call :meth:`start`
    before the forward pass and :meth:`stop` afterwards.
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
        for name, module in static_order(self._model):
            if _has_children(module):  # a weight owner: recorded before its children run
                handle = module.register_forward_pre_hook(self._recorder(name))
            else:
                handle = module.register_forward_hook(self._recorder(name))
            self._handles.append(handle)
        self._handles.append(self._model.register_forward_hook(self._on_root_finished))

    def stop(self) -> list[tuple[str, nn.Module]]:
        """Remove the hooks and return the recorded ``(name, module)`` calls, with the
        connection layers a parent used without calling them (:meth:`_with_unrun_layers`)."""
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        return self._with_unrun_layers(list(self._calls))

    def _with_unrun_layers(self, calls: list[tuple[str, nn.Module]]) -> list[tuple[str, nn.Module]]:
        """Place each connection layer that did not run right after its parent's first call,
        when that parent is a weight owner that ran.

        ``nn.MultiheadAttention`` hands ``out_proj.weight`` to the functional attention and
        never calls ``out_proj``, whose weights are used all the same. Every other module
        that did not run stays out of the traced order.
        """
        ran = {id(module) for _name, module in calls}
        modules = dict(self._model.named_modules())
        placed: dict[int, list[tuple[str, nn.Module]]] = {}  # parent -> layers placed after it
        for name, module in static_order(self._model):  # parents before their children
            if not name or id(module) in ran or classify(module) is not LayerRole.CONNECTION:
                continue
            parent = modules.get(name.rpartition(".")[0])
            if parent is None or id(parent) not in ran:
                continue
            if classify(parent) is not LayerRole.CONNECTION:
                continue
            placed.setdefault(id(parent), []).append((name, module))
            ran.add(id(module))
        if not placed:
            return calls
        ordered: list[tuple[str, nn.Module]] = []

        def emit(name: str, module: nn.Module) -> None:
            ordered.append((name, module))
            for item in placed.pop(id(module), ()):  # after the parent's first call only
                emit(*item)

        for name, module in calls:
            emit(name, module)
        return ordered

    def _recorder(self, name: str):
        def record(module: nn.Module, *_hook_arguments: object) -> None:
            if not self.complete:
                self._calls.append((name, module))

        return record

    def _on_root_finished(self, _module: nn.Module, _inputs: object, _output: object) -> None:
        self.complete = True

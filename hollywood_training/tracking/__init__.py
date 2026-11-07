"""Network tracking: architecture discovery and per-epoch weight snapshots.

The data classes (``ArchitectureInfo``, ``WeightSnapshot``, ...) need no torch; the
tracking machinery (``track``, ``NetworkTracker``, ...) is imported on first use.
"""

from importlib import import_module

from .architecture import (
    ArchitectureInfo,
    LayerInfo,
    LayerOperation,
    NetworkArchitecture,
    OperationInfo,
    TrackedLayer,
)
from .roles import LayerRole
from .weights import (
    DEFAULT_NEURON_LIMIT,
    LayerWeights,
    TensorStats,
    WeightSnapshot,
    WeightStats,
    display_indices,
)

_LAZY = {
    "TrackOptions": ".decorator",
    "is_tracked": ".decorator",
    "track": ".decorator",
    "track_options": ".decorator",
    "ForwardTrace": ".inspector",
    "build_architecture": ".inspector",
    "leaf_modules": ".inspector",
    "static_order": ".inspector",
    "unwrap": ".inspector",
    "classify": ".layers",
    "connection_shape": ".layers",
    "summarize": ".layers",
    "capture": ".snapshot",
    "capture_layer": ".snapshot",
    "connection_matrix": ".snapshot",
    "NetworkTracker": ".tracker",
    "inspect_network": ".tracker",
}

__all__ = [
    "DEFAULT_NEURON_LIMIT",
    "ArchitectureInfo",
    "ForwardTrace",
    "LayerInfo",
    "LayerOperation",
    "LayerRole",
    "LayerWeights",
    "NetworkArchitecture",
    "NetworkTracker",
    "OperationInfo",
    "TensorStats",
    "TrackOptions",
    "TrackedLayer",
    "WeightSnapshot",
    "WeightStats",
    "build_architecture",
    "capture",
    "capture_layer",
    "classify",
    "connection_matrix",
    "connection_shape",
    "display_indices",
    "inspect_network",
    "is_tracked",
    "leaf_modules",
    "static_order",
    "summarize",
    "track",
    "track_options",
    "unwrap",
]


def __getattr__(name: str):
    # Exports that need torch are imported on first use (PEP 562), so the torch-free
    # data modules of this package can be used on their own - e.g. by the GUI process.
    module = _LAZY.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(module, __name__), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(__all__)

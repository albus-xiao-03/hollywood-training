"""Network tracking: architecture discovery and per-epoch weight snapshots."""

from .architecture import (
    ArchitectureInfo,
    LayerInfo,
    LayerOperation,
    NetworkArchitecture,
    OperationInfo,
    TrackedLayer,
)
from .decorator import TrackOptions, is_tracked, track, track_options
from .inspector import ForwardTrace, build_architecture, leaf_modules, unwrap
from .layers import LayerRole, classify, connection_shape, summarize
from .snapshot import (
    DEFAULT_NEURON_LIMIT,
    LayerWeights,
    WeightSnapshot,
    WeightStats,
    capture,
    connection_matrix,
    display_indices,
)
from .tracker import NetworkTracker, inspect_network

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
    "TrackOptions",
    "TrackedLayer",
    "WeightSnapshot",
    "WeightStats",
    "build_architecture",
    "capture",
    "classify",
    "connection_matrix",
    "connection_shape",
    "display_indices",
    "inspect_network",
    "is_tracked",
    "leaf_modules",
    "summarize",
    "track",
    "track_options",
    "unwrap",
]

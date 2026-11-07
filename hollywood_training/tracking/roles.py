"""``LayerRole``: what a module contributes to the drawn network (torch-free)."""

from __future__ import annotations

from enum import Enum


class LayerRole(str, Enum):
    """What a module contributes to the drawn network."""

    CONNECTION = "connection"  # owns a weight matrix linking input features to output features
    ACTIVATION = "activation"
    NORMALIZATION = "normalization"
    REGULARIZATION = "regularization"
    POOLING = "pooling"
    RESHAPE = "reshape"
    OTHER = "other"

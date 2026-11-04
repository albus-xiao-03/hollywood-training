"""Classify ``torch.nn`` modules into the roles the visualisation understands.

Only public ``torch.nn`` classes are consulted. They are looked up by name, so classes that
exist only in newer PyTorch releases (for example ``nn.RMSNorm``) are used when present and
skipped otherwise.
"""

from __future__ import annotations

from enum import Enum

from torch import nn


class LayerRole(str, Enum):
    """What a leaf module contributes to the drawn network."""

    CONNECTION = "connection"  # owns a weight matrix linking input features to output features
    ACTIVATION = "activation"
    NORMALIZATION = "normalization"
    REGULARIZATION = "regularization"
    POOLING = "pooling"
    RESHAPE = "reshape"
    OTHER = "other"


def _public_classes(*names: str) -> tuple[type[nn.Module], ...]:
    found = (getattr(nn, name, None) for name in names)
    return tuple(cls for cls in found if isinstance(cls, type) and issubclass(cls, nn.Module))


# Lazy variants (LazyLinear, LazyConv2d, ...) subclass these, so they are covered too.
LINEAR_LAYERS = _public_classes("Linear")
CONVOLUTION_LAYERS = _public_classes("Conv1d", "Conv2d", "Conv3d")
TRANSPOSED_CONVOLUTION_LAYERS = _public_classes(
    "ConvTranspose1d", "ConvTranspose2d", "ConvTranspose3d"
)
EMBEDDING_LAYERS = _public_classes("Embedding")
ACTIVATION_LAYERS = _public_classes(
    "ReLU", "ReLU6", "LeakyReLU", "PReLU", "RReLU", "ELU", "SELU", "CELU", "GELU", "SiLU",
    "Mish", "Sigmoid", "LogSigmoid", "Hardsigmoid", "Tanh", "Hardtanh", "Tanhshrink",
    "Hardswish", "Softplus", "Softsign", "Softshrink", "Hardshrink", "Threshold", "GLU",
    "Softmax", "Softmin", "LogSoftmax", "Softmax2d",
)  # fmt: skip
NORMALIZATION_LAYERS = _public_classes(
    "BatchNorm1d", "BatchNorm2d", "BatchNorm3d", "SyncBatchNorm", "LayerNorm", "GroupNorm",
    "InstanceNorm1d", "InstanceNorm2d", "InstanceNorm3d", "LocalResponseNorm", "RMSNorm",
    "CrossMapLRN2d",
)  # fmt: skip
REGULARIZATION_LAYERS = _public_classes(
    "Dropout", "Dropout1d", "Dropout2d", "Dropout3d", "AlphaDropout", "FeatureAlphaDropout"
)
POOLING_LAYERS = _public_classes(
    "MaxPool1d", "MaxPool2d", "MaxPool3d", "AvgPool1d", "AvgPool2d", "AvgPool3d",
    "AdaptiveMaxPool1d", "AdaptiveMaxPool2d", "AdaptiveMaxPool3d", "AdaptiveAvgPool1d",
    "AdaptiveAvgPool2d", "AdaptiveAvgPool3d", "LPPool1d", "LPPool2d", "LPPool3d",
    "FractionalMaxPool2d", "FractionalMaxPool3d",
)  # fmt: skip
RESHAPE_LAYERS = _public_classes(
    "Flatten", "Unflatten", "Upsample", "PixelShuffle", "PixelUnshuffle"
)
PASS_THROUGH_LAYERS = _public_classes("Identity")

CONNECTION_LAYERS = (
    LINEAR_LAYERS + CONVOLUTION_LAYERS + TRANSPOSED_CONVOLUTION_LAYERS + EMBEDDING_LAYERS
)
OPERATION_ROLES = frozenset(
    {
        LayerRole.ACTIVATION,
        LayerRole.NORMALIZATION,
        LayerRole.REGULARIZATION,
        LayerRole.POOLING,
        LayerRole.RESHAPE,
    }
)


_LEAKY_RELU = _public_classes("LeakyReLU")
_UNINITIALIZED = (nn.parameter.UninitializedParameter, nn.parameter.UninitializedBuffer)


def is_uninitialized(tensor: object) -> bool:
    """True for the placeholder parameters of lazy modules before their first forward pass."""
    return isinstance(tensor, _UNINITIALIZED)


def weight_of(module: nn.Module) -> nn.Parameter | None:
    """The module's initialised ``weight`` parameter, if it has one."""
    weight = getattr(module, "weight", None)
    if isinstance(weight, nn.Parameter) and not is_uninitialized(weight):
        return weight
    return None


def has_weight_matrix(module: nn.Module) -> bool:
    """True when the module owns a 2-D ``weight`` parameter shaped ``(out, in)``."""
    weight = weight_of(module)
    return weight is not None and weight.dim() == 2


def is_pass_through(module: nn.Module) -> bool:
    """Modules that do nothing (``nn.Identity``) and are left out of the drawing."""
    return isinstance(module, PASS_THROUGH_LAYERS)


def classify(module: nn.Module) -> LayerRole:
    """Return the :class:`LayerRole` of a leaf module."""
    if isinstance(module, CONNECTION_LAYERS):
        return LayerRole.CONNECTION
    if isinstance(module, ACTIVATION_LAYERS):
        return LayerRole.ACTIVATION
    if isinstance(module, NORMALIZATION_LAYERS):
        return LayerRole.NORMALIZATION
    if isinstance(module, REGULARIZATION_LAYERS):
        return LayerRole.REGULARIZATION
    if isinstance(module, POOLING_LAYERS):
        return LayerRole.POOLING
    if isinstance(module, RESHAPE_LAYERS):
        return LayerRole.RESHAPE
    if has_weight_matrix(module):  # custom linear-like modules
        return LayerRole.CONNECTION
    return LayerRole.OTHER


def connection_shape(module: nn.Module) -> tuple[int | None, int | None]:
    """``(in_features, out_features)`` of a connection layer; ``None`` while still unknown."""
    weight = weight_of(module)
    if weight is None:
        return None, None
    if isinstance(module, EMBEDDING_LAYERS):
        num_embeddings, embedding_dim = weight.shape
        return int(num_embeddings), int(embedding_dim)
    if isinstance(module, CONVOLUTION_LAYERS + TRANSPOSED_CONVOLUTION_LAYERS):
        return int(module.in_channels), int(module.out_channels)
    if weight.dim() == 2:
        out_features, in_features = weight.shape
        return int(in_features), int(out_features)
    return None, None


def _kernel_text(kernel: object) -> str:
    if isinstance(kernel, tuple):
        return "x".join(str(k) for k in kernel)
    return str(kernel)


def summarize(module: nn.Module) -> str:
    """Short human-readable description such as ``Linear 2→16`` or ``Dropout(p=0.2)``."""
    name = type(module).__name__
    role = classify(module)
    if role is LayerRole.CONNECTION:
        in_features, out_features = connection_shape(module)
        if in_features is None:
            return f"{name} (uninitialised)"
        text = f"{name} {in_features}→{out_features}"
        kernel = getattr(module, "kernel_size", None)
        return f"{text} k{_kernel_text(kernel)}" if kernel is not None else text
    if isinstance(module, REGULARIZATION_LAYERS):
        return f"{name}(p={module.p:g})"
    if isinstance(module, _LEAKY_RELU):
        return f"{name}({module.negative_slope:g})"
    if isinstance(module, NORMALIZATION_LAYERS):
        size = getattr(module, "num_features", None) or getattr(module, "normalized_shape", None)
        if isinstance(size, tuple):
            size = "x".join(str(s) for s in size)
        return f"{name}({size})" if size is not None else name
    return name

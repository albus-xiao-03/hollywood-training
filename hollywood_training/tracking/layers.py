"""Classify ``torch.nn`` modules into the roles the visualisation understands.

Only public ``torch.nn`` classes are consulted. They are looked up by name, so classes that
exist only in newer PyTorch releases (for example ``nn.RMSNorm``) are used when present and
skipped otherwise.

A **weight tensor** is an initialised parameter with two or more dimensions that a module
holds directly. Any module that holds one and is not an operation class (activation,
normalisation, regularisation, pooling, reshape) is a connection layer, a **weight owner**:
``nn.Bilinear``, the recurrent layers, ``nn.MultiheadAttention`` (which also has a child)
and custom modules. So every weight matrix of a network belongs to a connection layer.
"""

from __future__ import annotations

from torch import nn

from .roles import LayerRole


def _public_classes(*names: str) -> tuple[type[nn.Module], ...]:
    found = (getattr(nn, name, None) for name in names)
    return tuple(cls for cls in found if isinstance(cls, type) and issubclass(cls, nn.Module))


# Lazy variants (LazyLinear, LazyConv2d, ...) subclass these, so they are covered too.
LINEAR_LAYERS = _public_classes("Linear")
CONVOLUTION_LAYERS = _public_classes("Conv1d", "Conv2d", "Conv3d")
TRANSPOSED_CONVOLUTION_LAYERS = _public_classes(
    "ConvTranspose1d", "ConvTranspose2d", "ConvTranspose3d"
)
EMBEDDING_LAYERS = _public_classes("Embedding", "EmbeddingBag")
BILINEAR_LAYERS = _public_classes("Bilinear")  # two inputs: x1 (in1) and x2 (in2)
ATTENTION_LAYERS = _public_classes("MultiheadAttention")  # q, k, v; out_proj is a child
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
    LINEAR_LAYERS
    + CONVOLUTION_LAYERS
    + TRANSPOSED_CONVOLUTION_LAYERS
    + EMBEDDING_LAYERS
    + BILINEAR_LAYERS
    + ATTENTION_LAYERS
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


def weight_tensors(module: nn.Module) -> list[tuple[str, nn.Parameter]]:
    """``(name, parameter)`` for every weight tensor the module holds directly: each
    initialised parameter with two or more dimensions, in registration order."""
    return [
        (name, parameter)
        for name, parameter in module.named_parameters(recurse=False)
        if not is_uninitialized(parameter) and parameter.dim() >= 2
    ]


def primary_weight(module: nn.Module) -> nn.Parameter | None:
    """The tensor a connection layer is drawn from: its ``weight`` when that has two or
    more dimensions, otherwise its first weight tensor; ``None`` while it has none."""
    weight = weight_of(module)
    if weight is not None and weight.dim() >= 2:
        return weight
    tensors = weight_tensors(module)
    return tensors[0][1] if tensors else None


def attention_projections(module: nn.Module) -> list[tuple[str, nn.Parameter]]:
    """The q, k, v input projections of an attention layer: ``[("in_proj_weight", W)]``
    when they share one ``(3E, E)`` tensor, otherwise ``q_proj_weight``, ``k_proj_weight``
    and ``v_proj_weight`` (a key or value size other than ``E``). Empty for other modules."""
    if not isinstance(module, ATTENTION_LAYERS):
        return []
    names = ("in_proj_weight", "q_proj_weight", "k_proj_weight", "v_proj_weight")
    return [(name, tensor) for name, tensor in weight_tensors(module) if name in names]


def is_pass_through(module: nn.Module) -> bool:
    """Modules that do nothing (``nn.Identity``) and are left out of the drawing."""
    return isinstance(module, PASS_THROUGH_LAYERS)


def classify(module: nn.Module) -> LayerRole:
    """Return the :class:`LayerRole` of a module (a leaf, or a weight owner with children)."""
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
    if weight_tensors(module):  # any other weight owner: recurrent, attention, custom, ...
        return LayerRole.CONNECTION
    return LayerRole.OTHER


def connection_shape(module: nn.Module) -> tuple[int | None, int | None]:
    """``(in_features, out_features)`` of a connection layer; ``None`` while still unknown."""
    weight = primary_weight(module)
    if weight is None:
        return None, None
    if isinstance(module, EMBEDDING_LAYERS):
        num_embeddings, embedding_dim = weight.shape
        return int(num_embeddings), int(embedding_dim)
    if isinstance(module, CONVOLUTION_LAYERS + TRANSPOSED_CONVOLUTION_LAYERS):
        return int(module.in_channels), int(module.out_channels)
    if isinstance(module, BILINEAR_LAYERS):  # both inputs, x1 then x2
        return int(module.in1_features) + int(module.in2_features), int(module.out_features)
    if isinstance(module, ATTENTION_LAYERS):  # outputs: the query, key and value projections
        embed = int(module.embed_dim)
        if len(attention_projections(module)) == 1:
            return embed, 3 * embed
        return embed + int(module.kdim) + int(module.vdim), 3 * embed  # query, key, value
    # Linear and any other weight owner: the tensor is read as (out, in, ...)
    return int(weight.shape[1]), int(weight.shape[0])


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
        if isinstance(module, BILINEAR_LAYERS):
            return f"{name} {module.in1_features}+{module.in2_features}→{out_features}"
        if isinstance(module, ATTENTION_LAYERS):
            inputs = str(in_features)
            if len(attention_projections(module)) > 1:
                inputs = f"{module.embed_dim}+{module.kdim}+{module.vdim}"
            return f"{name} {inputs}→{out_features} h{module.num_heads}"
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

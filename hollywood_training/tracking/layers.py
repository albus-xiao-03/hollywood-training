"""Classify ``torch.nn`` modules into the roles the visualisation understands.

Only public ``torch.nn`` classes are consulted. They are looked up by name, so classes that
exist only in newer PyTorch releases (for example ``nn.RMSNorm``) are used when present and
skipped otherwise.

A **weight tensor** is an initialised parameter with two or more dimensions that a module
holds directly. Any module that holds one and is not an operation class (activation,
normalisation, regularisation, pooling, reshape) is a connection layer, a **weight owner**:
``nn.Bilinear``, the recurrent layers, ``nn.MultiheadAttention`` (which also has a child)
and custom modules. So every weight matrix of a network belongs to a connection layer.

A recurrent layer (``nn.RNN``, ``nn.LSTM``, ``nn.GRU``) is drawn as one **stage** per
stacked layer, each followed by its projection stage when ``proj_size`` is set
(:func:`recurrent_stages`). The shape and summary functions take the stage as keywords.
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
RECURRENT_LAYERS = _public_classes("RNN", "LSTM", "GRU")  # one drawn stage per stacked layer
RECURRENT_CELLS = _public_classes("RNNCell", "LSTMCell", "GRUCell")
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
    + RECURRENT_LAYERS
    + RECURRENT_CELLS
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
_FOUR_GATES = _public_classes("LSTM", "LSTMCell")  # input, forget, cell, output
_THREE_GATES = _public_classes("GRU", "GRUCell")  # reset, update, new
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


def gate_count(module: nn.Module) -> int:
    """Gate blocks stacked along the rows of a recurrent weight: 4 for ``LSTM``, 3 for
    ``GRU``, 1 for ``RNN`` (their cells alike)."""
    if isinstance(module, _FOUR_GATES):
        return 4
    if isinstance(module, _THREE_GATES):
        return 3
    return 1


def _directions(module: nn.Module) -> tuple[str, ...]:
    """Parameter-name suffixes of the directions: forward, then reverse if bidirectional."""
    return ("", "_reverse") if getattr(module, "bidirectional", False) else ("",)


def _projection_size(module: nn.Module) -> int:
    return int(getattr(module, "proj_size", 0) or 0)


def recurrent_stages(module: nn.Module) -> list[tuple[int, bool]]:
    """``(stacked layer, projection)`` of every drawn stage of a recurrent layer, in the
    order data flows through them: each stacked layer, followed by its projection when
    ``proj_size`` is set. Any other module is a single stage, ``[(0, False)]``."""
    if not isinstance(module, RECURRENT_LAYERS):
        return [(0, False)]
    kinds = (False, True) if _projection_size(module) else (False,)
    return [(layer, projection) for layer in range(int(module.num_layers)) for projection in kinds]


def stage_name(name: str, stage: int = 0, projection: bool = False) -> str:
    """The layer name of a stage: ``rnn`` (stacked layer 0), ``rnn.l1``, ``rnn.proj`` or
    ``rnn.l1.proj``. A recurrent module has no children, so no module path collides."""
    base = name if stage == 0 else f"{name}.l{stage}"
    return f"{base}.proj" if projection else base


def recurrent_parameters(
    module: nn.Module, stage: int = 0, projection: bool = False
) -> list[tuple[str, nn.Parameter]]:
    """``(name, parameter)`` of one stage of a recurrent layer, every direction: the
    ``weight_ih``, ``weight_hh``, ``bias_ih`` and ``bias_hh`` of stacked layer ``stage``,
    or its ``weight_hr`` for the projection stage. Every parameter of a recurrent cell.
    Empty for any other module."""
    kinds = ("weight_hr",) if projection else ("weight_ih", "weight_hh", "bias_ih", "bias_hh")
    if isinstance(module, RECURRENT_CELLS):
        names = [] if projection else list(kinds)
    elif isinstance(module, RECURRENT_LAYERS):
        names = [f"{kind}_l{stage}{suffix}" for suffix in _directions(module) for kind in kinds]
    else:
        return []
    found = dict(module.named_parameters(recurse=False))
    return [(name, found[name]) for name in names if name in found]


def recurrent_tensors(module: nn.Module, kind: str, stage: int = 0) -> list[nn.Parameter]:
    """``kind`` (``weight_ih``, ``weight_hh``, ``weight_hr``, ``bias_ih`` or ``bias_hh``) of
    stacked layer ``stage`` of a recurrent layer or cell: one parameter per direction,
    forward first. Empty when there is none, e.g. without bias."""
    if isinstance(module, RECURRENT_CELLS):
        names = [kind]
    elif isinstance(module, RECURRENT_LAYERS):
        names = [f"{kind}_l{stage}{suffix}" for suffix in _directions(module)]
    else:
        return []
    found = dict(module.named_parameters(recurse=False))
    return [found[name] for name in names if name in found]


def stage_weight_tensors(
    module: nn.Module, stage: int = 0, projection: bool = False
) -> list[tuple[str, nn.Parameter]]:
    """The weight tensors of one drawn stage: those of the stage for a recurrent layer
    (:func:`recurrent_parameters` with two or more dimensions), otherwise every weight
    tensor of the module."""
    if isinstance(module, RECURRENT_LAYERS + RECURRENT_CELLS):
        return [
            (name, tensor)
            for name, tensor in recurrent_parameters(module, stage, projection)
            if not is_uninitialized(tensor) and tensor.dim() >= 2
        ]
    return weight_tensors(module)


def _recurrent_shape(module: nn.Module, stage: int, projection: bool) -> tuple[int, int]:
    hidden = int(module.hidden_size)
    if isinstance(module, RECURRENT_CELLS):
        return int(module.input_size), hidden
    directions = len(_directions(module))
    projected = _projection_size(module)
    if projection:  # weight_hr: the hidden state to the (smaller) projected output
        return directions * hidden, directions * projected
    if stage == 0:
        return int(module.input_size), directions * hidden
    return directions * (projected or hidden), directions * hidden  # the layer below's output


def _recurrent_summary(module: nn.Module, stage: int, projection: bool) -> str:
    in_features, out_features = _recurrent_shape(module, stage, projection)
    directions = len(_directions(module))

    def width(features: int) -> str:  # both directions: "2×5"
        return f"{directions}×{features // directions}" if directions > 1 else str(features)

    inputs = str(in_features) if stage == 0 and not projection else width(in_features)
    kind = f"{type(module).__name__} projection" if projection else type(module).__name__
    text = f"{kind} {inputs}→{width(out_features)}"
    layers = int(getattr(module, "num_layers", 1))
    return f"{text} L{stage + 1}/{layers}" if layers > 1 else text


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


def connection_shape(
    module: nn.Module, *, stage: int = 0, projection: bool = False
) -> tuple[int | None, int | None]:
    """``(in_features, out_features)`` of a connection layer; ``None`` while still unknown.

    ``stage`` and ``projection`` select a stage of a recurrent layer
    (:func:`recurrent_stages`); other modules have only stage 0.
    """
    weight = primary_weight(module)
    if weight is None:
        return None, None
    if isinstance(module, RECURRENT_LAYERS + RECURRENT_CELLS):
        return _recurrent_shape(module, stage, projection)
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


def summarize(module: nn.Module, *, stage: int = 0, projection: bool = False) -> str:
    """Short human-readable description such as ``Linear 2→16`` or ``Dropout(p=0.2)``;
    ``stage`` and ``projection`` select a stage of a recurrent layer."""
    name = type(module).__name__
    role = classify(module)
    if role is LayerRole.CONNECTION:
        in_features, out_features = connection_shape(module, stage=stage, projection=projection)
        if in_features is None:
            return f"{name} (uninitialised)"
        if isinstance(module, RECURRENT_LAYERS + RECURRENT_CELLS):
            return _recurrent_summary(module, stage, projection)
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

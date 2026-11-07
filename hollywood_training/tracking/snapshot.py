"""Per-epoch copies of the weights a frontend draws.

Every connection layer is reduced to a **connection matrix** shaped ``(out, in)``: entry
``[j, i]`` says how strongly input feature (or channel) ``i`` feeds output ``j``.

* ``Linear`` and other 2-D weights: the weight itself.
* ``Conv*``: the signed mean over the kernel window, block-diagonal for grouped
  convolutions (unconnected channel pairs are 0).
* ``ConvTranspose*``: the same, transposed back to ``(out, in)``.
* ``Embedding`` / ``EmbeddingBag``: the transposed table, ``(embedding_dim, num_embeddings)``.
* ``Bilinear``: the inputs are x1's features, then x2's. ``W[j, i, k]`` links x1 feature
  ``i`` and x2 feature ``k`` to output ``j``: an x1 feature gets the mean over ``k``, an x2
  feature the mean over ``i``.
* Any other weight owner: its primary weight tensor read as ``(out, in, ...)``, with the
  signed mean over the remaining dimensions.

At most ``limit`` neurons per side are kept (evenly spaced indices, recorded in the
snapshot) so that very wide layers stay drawable. Statistics (:class:`WeightStats`) are
always computed on the **full** weight tensor, and on every weight tensor of the layer
(:class:`TensorStats`). Arrays in a snapshot are read-only copies on the CPU, so a snapshot
can be handed to another thread while training continues.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import torch
from torch import nn

from .architecture import NetworkArchitecture, TrackedLayer
from .layers import (
    BILINEAR_LAYERS,
    CONVOLUTION_LAYERS,
    EMBEDDING_LAYERS,
    TRANSPOSED_CONVOLUTION_LAYERS,
    is_uninitialized,
    primary_weight,
    weight_tensors,
)
from .weights import (
    DEFAULT_NEURON_LIMIT,
    LayerWeights,
    TensorStats,
    WeightSnapshot,
    WeightStats,
    display_indices,
)


def _readonly(array: np.ndarray) -> np.ndarray:
    array.setflags(write=False)
    return array


def _to_numpy(tensor: torch.Tensor) -> np.ndarray:
    return _readonly(tensor.detach().to(device="cpu", dtype=torch.float32).numpy().copy())


def weight_stats(weight: torch.Tensor) -> WeightStats:
    """Statistics of the full tensor (computed in float32 on the tensor's device)."""
    values = weight.detach()
    if values.dtype != torch.float32:
        values = values.float()
    count = values.numel()
    if count == 0:
        return WeightStats(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0)
    return WeightStats(
        mean=float(values.mean()),
        std=float(values.std(correction=0)),
        minimum=float(values.min()),
        maximum=float(values.max()),
        abs_mean=float(values.abs().mean()),
        norm=float(torch.linalg.vector_norm(values)),
        count=count,
    )


def _kernel_mean(weight: torch.Tensor) -> torch.Tensor:
    weight = weight.float()
    return weight.mean(dim=tuple(range(2, weight.dim()))) if weight.dim() > 2 else weight


def _grouped(
    block: torch.Tensor, rows: torch.Tensor, cols: torch.Tensor, row_pg: int, col_pg: int
) -> torch.Tensor:
    """Gather ``block[:, local col]`` and zero the pairs that belong to different groups.

    ``block`` already holds the selected rows; its columns index channels within a group.
    """
    row_group = rows // row_pg
    col_group = cols // col_pg
    gathered = block[:, cols - col_group * col_pg]
    return gathered * (row_group[:, None] == col_group[None, :]).to(gathered.dtype)


def connection_matrix(
    module: nn.Module, in_index: np.ndarray, out_index: np.ndarray
) -> torch.Tensor | None:
    """Connection matrix of ``module`` restricted to the given indices, shape ``(out, in)``."""
    weight = primary_weight(module)
    if weight is None:
        return None
    weight = weight.detach()
    device = weight.device
    rows_out = torch.as_tensor(out_index, dtype=torch.long, device=device)
    cols_in = torch.as_tensor(in_index, dtype=torch.long, device=device)

    if isinstance(module, EMBEDDING_LAYERS):  # (num_embeddings, dim)
        return weight.index_select(0, cols_in).index_select(1, rows_out).float().T
    if isinstance(module, TRANSPOSED_CONVOLUTION_LAYERS):  # (in, out / groups, *kernel)
        groups = int(module.groups)
        block = _kernel_mean(weight.index_select(0, cols_in))  # (k_in, out_pg)
        in_pg = int(module.in_channels) // groups
        out_pg = int(module.out_channels) // groups
        return _grouped(block, cols_in, rows_out, in_pg, out_pg).T
    if isinstance(module, CONVOLUTION_LAYERS):  # (out, in / groups, *kernel)
        groups = int(module.groups)
        block = _kernel_mean(weight.index_select(0, rows_out))  # (k_out, in_pg)
        out_pg = int(module.out_channels) // groups
        in_pg = int(module.in_channels) // groups
        return _grouped(block, rows_out, cols_in, out_pg, in_pg)
    if isinstance(module, BILINEAR_LAYERS):  # (out, in1, in2): inputs x1, then x2
        rows = weight.index_select(0, rows_out).float()
        both = torch.cat((rows.mean(dim=2), rows.mean(dim=1)), dim=1)  # (k_out, in1 + in2)
        return both.index_select(1, cols_in)
    # Linear and any other weight owner: (out, in, ...), averaged over the other dimensions
    return _kernel_mean(weight.index_select(0, rows_out).index_select(1, cols_in))


def _sampled_bias(module: nn.Module, out_index: np.ndarray, out_features: int) -> np.ndarray | None:
    """The module's ``bias`` at ``out_index``, when it is a vector with one entry per output."""
    bias = getattr(module, "bias", None)
    if not isinstance(bias, torch.Tensor) or is_uninitialized(bias):
        return None
    if bias.dim() != 1 or bias.shape[0] != out_features:
        return None
    index = torch.as_tensor(out_index, dtype=torch.long, device=bias.device)
    return _to_numpy(bias.detach().index_select(0, index))


def tensor_stats(
    module: nn.Module, known: dict[int, WeightStats] | None = None
) -> tuple[TensorStats, ...]:
    """:class:`TensorStats` of every weight tensor ``module`` holds; ``known`` maps
    ``id(tensor)`` to statistics computed already, which are reused."""
    known = known or {}
    return tuple(
        TensorStats(
            name=name,
            shape=tuple(int(size) for size in tensor.shape),
            stats=known[id(tensor)] if id(tensor) in known else weight_stats(tensor),
        )
        for name, tensor in weight_tensors(module)
    )


def capture_layer(
    layer: TrackedLayer, limit: int, previous: LayerWeights | None = None
) -> LayerWeights | None:
    """Snapshot one layer; ``None`` while its parameters are still uninitialised."""
    in_features, out_features = layer.in_features, layer.out_features
    weight = primary_weight(layer.module)
    if weight is None or in_features is None or out_features is None:
        return None
    in_index = display_indices(in_features, limit)
    out_index = display_indices(out_features, limit)
    with torch.no_grad():
        matrix_tensor = connection_matrix(layer.module, in_index, out_index)
        if matrix_tensor is None:
            return None
        matrix = _to_numpy(matrix_tensor)
        bias = _sampled_bias(layer.module, out_index, out_features)
        stats = weight_stats(weight)
        tensors = tensor_stats(layer.module, {id(weight): stats})

    delta = None
    update_ratio = None
    if (
        previous is not None
        and previous.matrix.shape == matrix.shape
        and np.array_equal(previous.in_index, in_index)
        and np.array_equal(previous.out_index, out_index)
    ):
        delta = _readonly(matrix - previous.matrix)
        update_ratio = float(np.linalg.norm(delta) / (np.linalg.norm(previous.matrix) + 1e-12))

    return LayerWeights(
        name=layer.name,
        matrix=matrix,
        bias=bias,
        delta=delta,
        in_index=_readonly(in_index),
        out_index=_readonly(out_index),
        in_features=in_features,
        out_features=out_features,
        stats=stats,
        update_ratio=update_ratio,
        tensors=tensors,
    )


def capture(
    architecture: NetworkArchitecture,
    *,
    epoch: int,
    limit: int = DEFAULT_NEURON_LIMIT,
    previous: WeightSnapshot | None = None,
) -> WeightSnapshot:
    """Snapshot every connection layer of ``architecture``."""
    layers: Iterable[LayerWeights | None] = (
        capture_layer(layer, limit, previous.layer(layer.name) if previous else None)
        for layer in architecture.layers
    )
    return WeightSnapshot(epoch=epoch, layers=tuple(item for item in layers if item is not None))

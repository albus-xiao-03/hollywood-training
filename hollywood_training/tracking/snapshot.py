"""Per-epoch copies of the weights a frontend draws.

Every connection layer is reduced to a **connection matrix** shaped ``(out, in)``: entry
``[j, i]`` says how strongly input feature (or channel) ``i`` feeds output ``j``.

* ``Linear`` and other 2-D weights: the weight itself.
* ``Conv*``: the signed mean over the kernel window, block-diagonal for grouped
  convolutions (unconnected channel pairs are 0).
* ``ConvTranspose*``: the same, transposed back to ``(out, in)``.
* ``Embedding`` / ``EmbeddingBag``: the transposed table, ``(embedding_dim, num_embeddings)``.
* ``MultiheadAttention``: the outputs are the query, key and value projections (3E). With
  one ``in_proj_weight`` it is the matrix itself; with separate ``q/k/v_proj_weight`` the
  inputs are the query, key and value features, and the matrix is block-diagonal.
* ``Bilinear``: the inputs are x1's features, then x2's. ``W[j, i, k]`` links x1 feature
  ``i`` and x2 feature ``k`` to output ``j``: an x1 feature gets the mean over ``k``, an x2
  feature the mean over ``i``.
* ``RNN`` / ``LSTM`` / ``GRU``, per stage (stacked layer): the signed mean over the gate
  blocks of ``weight_ih``, one row block per direction (forward first). The recurrent
  weights ``weight_hh`` become the layer's :class:`Recurrence`, block-diagonal per
  direction. A ``proj_size`` projection stage is ``weight_hr``, block-diagonal. The cells
  (``RNNCell``, ...) are read like a one-layer recurrent layer.
* Any other weight owner: its primary weight tensor read as ``(out, in, ...)``, with the
  signed mean over the remaining dimensions.

At most ``limit`` neurons per side are kept (evenly spaced indices, recorded in the
snapshot) so that very wide layers stay drawable. Statistics (:class:`WeightStats`) are
always computed on the **full** weight tensor, and on every weight tensor of the layer
(:class:`TensorStats`). Arrays in a snapshot are read-only copies on the CPU, so a snapshot
can be handed to another thread while training continues.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np
import torch
from torch import nn

from .architecture import NetworkArchitecture, TrackedLayer
from .layers import (
    ATTENTION_LAYERS,
    BILINEAR_LAYERS,
    CONVOLUTION_LAYERS,
    EMBEDDING_LAYERS,
    RECURRENT_CELLS,
    RECURRENT_LAYERS,
    TRANSPOSED_CONVOLUTION_LAYERS,
    attention_projections,
    connection_shape,
    gate_count,
    is_uninitialized,
    primary_weight,
    recurrent_stages,
    recurrent_tensors,
    stage_weight_tensors,
)
from .weights import (
    DEFAULT_NEURON_LIMIT,
    LayerWeights,
    Recurrence,
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


def _block_diagonal(
    blocks: Sequence[torch.Tensor], rows: torch.Tensor, cols: torch.Tensor
) -> torch.Tensor:
    """``torch.block_diag(*blocks)`` restricted to ``rows`` and ``cols``, without building
    the full matrix: every entry outside the blocks is 0."""
    matrix = torch.zeros(rows.numel(), cols.numel(), device=rows.device)
    row_start = col_start = 0
    for block in blocks:
        height, width = int(block.shape[0]), int(block.shape[1])
        r = torch.nonzero((rows >= row_start) & (rows < row_start + height)).flatten()
        c = torch.nonzero((cols >= col_start) & (cols < col_start + width)).flatten()
        if r.numel() and c.numel():
            piece = block.index_select(0, rows[r] - row_start)
            matrix[r[:, None], c[None, :]] = piece.index_select(1, cols[c] - col_start).float()
        row_start += height
        col_start += width
    return matrix


def _gate_mean(tensor: torch.Tensor, gates: int) -> torch.Tensor:
    """``(G·H, ...)`` -> ``(H, ...)``: the signed mean over the ``G`` gate blocks a recurrent
    tensor stacks along its rows (LSTM: input, forget, cell, output; GRU: reset, update,
    new), as the kernel mean does for convolutions."""
    tensor = tensor.detach().float()
    if gates == 1:
        return tensor
    return tensor.reshape(gates, tensor.shape[0] // gates, *tensor.shape[1:]).mean(dim=0)


def _recurrent_matrix(
    module: nn.Module, stage: int, projection: bool, rows: torch.Tensor, cols: torch.Tensor
) -> torch.Tensor:
    if projection:  # weight_hr (P, H) per direction: each direction projects its own state
        blocks = [tensor.detach() for tensor in recurrent_tensors(module, "weight_hr", stage)]
        return _block_diagonal(blocks, rows, cols)
    gates = gate_count(module)
    directions = recurrent_tensors(module, "weight_ih", stage)  # (G·H, in) each
    full = torch.cat([_gate_mean(tensor, gates) for tensor in directions], dim=0)
    return full.index_select(0, rows).index_select(1, cols)  # (D·H, in): forward rows first


def connection_matrix(
    module: nn.Module,
    in_index: np.ndarray,
    out_index: np.ndarray,
    *,
    stage: int = 0,
    projection: bool = False,
) -> torch.Tensor | None:
    """Connection matrix of ``module`` restricted to the given indices, shape ``(out, in)``;
    ``stage`` and ``projection`` select a stage of a recurrent layer."""
    weight = primary_weight(module)
    if weight is None:
        return None
    weight = weight.detach()
    device = weight.device
    rows_out = torch.as_tensor(out_index, dtype=torch.long, device=device)
    cols_in = torch.as_tensor(in_index, dtype=torch.long, device=device)

    if isinstance(module, RECURRENT_LAYERS + RECURRENT_CELLS):
        return _recurrent_matrix(module, stage, projection, rows_out, cols_in)

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
    if isinstance(module, ATTENTION_LAYERS):  # outputs: the q, k, v projections (3E)
        projections = [tensor.detach() for _name, tensor in attention_projections(module)]
        if len(projections) > 1:  # q (E, E), k (E, kdim), v (E, vdim): one block each
            return _block_diagonal(projections, rows_out, cols_in)
        weight = projections[0]  # in_proj_weight (3E, E), read like a Linear weight
    if isinstance(module, BILINEAR_LAYERS):  # (out, in1, in2): inputs x1, then x2
        rows = weight.index_select(0, rows_out).float()
        both = torch.cat((rows.mean(dim=2), rows.mean(dim=1)), dim=1)  # (k_out, in1 + in2)
        return both.index_select(1, cols_in)
    # Linear and any other weight owner: (out, in, ...), averaged over the other dimensions
    return _kernel_mean(weight.index_select(0, rows_out).index_select(1, cols_in))


def recurrent_matrix(
    module: nn.Module, out_index: np.ndarray, source_index: np.ndarray, *, stage: int = 0
) -> torch.Tensor | None:
    """The recurrent connections of stacked layer ``stage`` of a recurrent layer or cell,
    restricted to the given indices: the signed gate mean of ``weight_hh``, block-diagonal
    per direction. Rows are the stage's outputs (``D·H``), columns the outputs that feed
    back (``D·H``, or ``D·P`` from the projection). ``None`` for other modules."""
    tensors = recurrent_tensors(module, "weight_hh", stage)
    if not tensors:
        return None
    blocks = [_gate_mean(tensor, gate_count(module)) for tensor in tensors]
    device = tensors[0].device
    rows = torch.as_tensor(out_index, dtype=torch.long, device=device)
    cols = torch.as_tensor(source_index, dtype=torch.long, device=device)
    return _block_diagonal(blocks, rows, cols)


def _drawn_tensors(module: nn.Module, stage: int, projection: bool) -> list[torch.Tensor]:
    """The weight tensors the connection matrix of a layer (stage) is read from."""
    if isinstance(module, RECURRENT_LAYERS + RECURRENT_CELLS):
        return list(recurrent_tensors(module, "weight_hr" if projection else "weight_ih", stage))
    projections = attention_projections(module)
    if projections:
        return [tensor for _name, tensor in projections]
    weight = primary_weight(module)
    return [] if weight is None else [weight]


def _stats_of(tensors: Sequence[torch.Tensor]) -> WeightStats:
    """:func:`weight_stats` of one tensor, or of several taken together."""
    if len(tensors) == 1:
        return weight_stats(tensors[0])
    return weight_stats(torch.cat([tensor.detach().reshape(-1) for tensor in tensors]))


def _recurrent_bias(module: nn.Module, stage: int, out_index: np.ndarray) -> np.ndarray | None:
    """The gate mean of ``bias_ih + bias_hh`` per direction, at ``out_index``."""
    inputs = recurrent_tensors(module, "bias_ih", stage)
    hidden = recurrent_tensors(module, "bias_hh", stage)
    if not inputs or len(inputs) != len(hidden):
        return None
    gates = gate_count(module)
    pairs = zip(inputs, hidden, strict=True)  # same length: checked above
    full = torch.cat([_gate_mean(a.detach() + b.detach(), gates) for a, b in pairs])
    index = torch.as_tensor(out_index, dtype=torch.long, device=full.device)
    return _to_numpy(full.index_select(0, index))


def _sampled_bias(module: nn.Module, out_index: np.ndarray, out_features: int) -> np.ndarray | None:
    """The bias at ``out_index`` - ``bias``, or ``in_proj_bias`` for attention - when it is
    a vector with one entry per output."""
    name = "in_proj_bias" if isinstance(module, ATTENTION_LAYERS) else "bias"
    bias = getattr(module, name, None)
    if not isinstance(bias, torch.Tensor) or is_uninitialized(bias):
        return None
    if bias.dim() != 1 or bias.shape[0] != out_features:
        return None
    index = torch.as_tensor(out_index, dtype=torch.long, device=bias.device)
    return _to_numpy(bias.detach().index_select(0, index))


def tensor_stats(
    module: nn.Module,
    known: dict[int, WeightStats] | None = None,
    *,
    stage: int = 0,
    projection: bool = False,
) -> tuple[TensorStats, ...]:
    """:class:`TensorStats` of every weight tensor of a layer - for a recurrent layer, of
    the given stage only; ``known`` maps ``id(tensor)`` to statistics computed already,
    which are reused."""
    known = known or {}
    return tuple(
        TensorStats(
            name=name,
            shape=tuple(int(size) for size in tensor.shape),
            stats=known[id(tensor)] if id(tensor) in known else weight_stats(tensor),
        )
        for name, tensor in stage_weight_tensors(module, stage, projection)
    )


def _recurrence(
    layer: TrackedLayer, out_index: np.ndarray, limit: int, previous: Recurrence | None
) -> Recurrence | None:
    """The :class:`Recurrence` of a recurrent stage (not of a projection stage)."""
    module = layer.module
    if layer.projection or not isinstance(module, RECURRENT_LAYERS + RECURRENT_CELLS):
        return None
    if (layer.stage, True) in recurrent_stages(module):  # fed back from the projection
        source = f"{layer.name}.proj"
        source_features = connection_shape(module, stage=layer.stage, projection=True)[1] or 0
    else:
        source, source_features = layer.name, layer.out_features or 0
    source_index = display_indices(source_features, limit)
    matrix_tensor = recurrent_matrix(module, out_index, source_index, stage=layer.stage)
    if matrix_tensor is None:
        return None
    matrix = _to_numpy(matrix_tensor)
    delta = None
    if (
        previous is not None
        and previous.matrix.shape == matrix.shape
        and np.array_equal(previous.source_index, source_index)
    ):
        delta = _readonly(matrix - previous.matrix)
    return Recurrence(source, matrix, delta, _readonly(source_index))


def capture_layer(
    layer: TrackedLayer, limit: int, previous: LayerWeights | None = None
) -> LayerWeights | None:
    """Snapshot one layer; ``None`` while its parameters are still uninitialised."""
    in_features, out_features = layer.in_features, layer.out_features
    module, stage, projection = layer.module, layer.stage, layer.projection
    drawn = _drawn_tensors(module, stage, projection)
    if not drawn or in_features is None or out_features is None:
        return None
    in_index = display_indices(in_features, limit)
    out_index = display_indices(out_features, limit)
    same_outputs = previous is not None and np.array_equal(previous.out_index, out_index)
    with torch.no_grad():
        matrix_tensor = connection_matrix(
            module, in_index, out_index, stage=stage, projection=projection
        )
        if matrix_tensor is None:
            return None
        matrix = _to_numpy(matrix_tensor)
        if isinstance(module, RECURRENT_LAYERS + RECURRENT_CELLS):
            bias = None if projection else _recurrent_bias(module, stage, out_index)
        else:
            bias = _sampled_bias(module, out_index, out_features)
        stats = _stats_of(drawn)
        known = {id(drawn[0]): stats} if len(drawn) == 1 else {}
        tensors = tensor_stats(module, known, stage=stage, projection=projection)
        earlier = previous.recurrence if previous is not None and same_outputs else None
        recurrence = _recurrence(layer, out_index, limit, earlier)

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
        recurrence=recurrence,
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

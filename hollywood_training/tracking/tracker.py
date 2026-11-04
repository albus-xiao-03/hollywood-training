"""``NetworkTracker``: follows one model instance through training.

It owns the model's :class:`NetworkArchitecture` (refined once by a forward trace) and
produces a :class:`WeightSnapshot` whenever asked, remembering the previous one so each
snapshot carries the per-connection change since the last.

The tracker must be used from the thread that trains the model, between optimisation
steps (the :class:`~hollywood_training.training.manager.TrainingManager` does this).
"""

from __future__ import annotations

from torch import nn

from .architecture import NetworkArchitecture
from .decorator import TrackOptions, track_options
from .inspector import ForwardTrace, architecture_signature, build_architecture, unwrap
from .snapshot import DEFAULT_NEURON_LIMIT, WeightSnapshot, capture


class NetworkTracker:
    """Track the architecture and weights of ``model``.

    Args:
        model: the network to follow.
        max_neurons: neurons kept per column in snapshots (wider layers are sampled).
        options: display options; defaults to what ``@track`` declared, if anything.
    """

    def __init__(
        self,
        model: nn.Module,
        *,
        max_neurons: int = DEFAULT_NEURON_LIMIT,
        options: TrackOptions | None = None,
    ) -> None:
        if not isinstance(model, nn.Module):
            raise TypeError(f"NetworkTracker expects a torch.nn.Module, got {type(model).__name__}")
        if max_neurons < 2:
            raise ValueError("max_neurons must be at least 2")
        self._model = model
        if options is None:  # declared on the model, or on the network inside a wrapper
            options = track_options(model) or track_options(unwrap(model)) or TrackOptions()
        self._options = options
        self._max_neurons = int(max_neurons)
        self._architecture = build_architecture(model, self._options)
        self._trace: ForwardTrace | None = None
        self._previous: WeightSnapshot | None = None

    @property
    def model(self) -> nn.Module:
        return self._model

    @property
    def options(self) -> TrackOptions:
        return self._options

    @property
    def max_neurons(self) -> int:
        return self._max_neurons

    @property
    def architecture(self) -> NetworkArchitecture:
        return self._architecture

    @property
    def tracing(self) -> bool:
        return self._trace is not None

    def refresh(self) -> NetworkArchitecture:
        """Re-read the static structure (e.g. after modules were replaced)."""
        self._architecture = build_architecture(self._model, self._options)
        return self._architecture

    def begin_trace(self) -> None:
        """Start recording the execution order of the next forward pass."""
        if self._trace is None:
            self._trace = ForwardTrace(self._model)
            self._trace.start()

    def end_trace(self) -> bool:
        """Stop recording; adopt the traced order. Returns True if the architecture changed.

        Only a *complete* forward pass is adopted: if it raised (or never went through the
        model's ``__call__``), the current architecture is kept.
        """
        trace, self._trace = self._trace, None
        if trace is None:
            return False
        calls = trace.stop()
        if not calls or not trace.complete:
            return False
        traced = build_architecture(self._model, self._options, order=calls)
        changed = architecture_signature(traced) != architecture_signature(self._architecture)
        self._architecture = traced
        return changed

    def snapshot(self, epoch: int, *, baseline: bool = True) -> WeightSnapshot:
        """Copy the current weights.

        Args:
            epoch: number of completed epochs the snapshot belongs to.
            baseline: when True the snapshot becomes the reference for the next one's
                ``delta``; pass False for intermediate looks (e.g. while paused).
        """
        snapshot = capture(
            self._architecture, epoch=epoch, limit=self._max_neurons, previous=self._previous
        )
        if baseline:
            self._previous = snapshot
        return snapshot


def inspect_network(model: nn.Module) -> NetworkArchitecture:
    """Convenience: the static architecture of ``model``, using options declared by ``@track``."""
    return NetworkTracker(model).architecture

"""``run()``: the management method. Hands a :class:`TrainingProcess` to the manager and a
frontend - the native GUI window by default - and returns when the session is over.
"""

from __future__ import annotations

from .commands.builtin import create_registry
from .console import ConsoleFrontend
from .sync.channel import SyncChannel
from .sync.control import TrainingControl
from .sync.events import TrainingState
from .tracking.snapshot import DEFAULT_NEURON_LIMIT
from .tracking.tracker import NetworkTracker
from .training.history import TrainingResult
from .training.manager import TrainingManager
from .training.process import TrainingProcess


class TrainingFailedError(RuntimeError):
    """Raised by :func:`run` when the user's training code raised an exception.

    The original exception is chained as ``__cause__``; ``result`` holds what completed.
    """

    def __init__(self, message: str, result: TrainingResult) -> None:
        super().__init__(message)
        self.result = result


def run(
    process: TrainingProcess,
    *,
    gui: bool = True,
    start_paused: bool = False,
    close_on_finish: bool = False,
    max_neurons: int = DEFAULT_NEURON_LIMIT,
    title: str | None = None,
    log_every: int | None = None,
) -> TrainingResult:
    """Run ``process`` under management and visualise it.

    Args:
        process: the wrapped training process.
        gui: open the native GUI window (default). ``False`` prints to the console instead,
            for machines without a display.
        start_paused: open the window paused, so the initial network can be inspected;
            ``resume`` (or ``step``) in the terminal starts training. GUI only.
        close_on_finish: close the window by itself once training ends. Otherwise the
            window stays open (more epochs can be requested with ``epochs <n>``).
        max_neurons: neurons drawn per layer column; wider layers are sampled evenly.
        title: window title (default: derived from the network's class name).
        log_every: console only - print every n-th epoch.

    Returns:
        The :class:`TrainingResult` with the per-epoch history.

    Raises:
        TrainingFailedError: when the training code raised (after the window was closed).
        ProcessError: when the process is incomplete (checked before anything starts).
    """
    if not isinstance(process, TrainingProcess):
        raise TypeError(f"run() expects a TrainingProcess, got {type(process).__name__}")
    process.resolve_steps()  # fail fast, before any window opens
    if start_paused and not gui:
        raise ValueError("start_paused needs the GUI: there is no terminal to resume from")

    control = TrainingControl(process.epochs, start_paused=start_paused)
    channel = SyncChannel()
    tracker = NetworkTracker(process.model, max_neurons=max_neurons)
    manager = TrainingManager(
        process,
        channel=channel,
        control=control,
        tracker=tracker,
        linger=gui and not close_on_finish,
    )

    if gui:
        from .gui.app import QtFrontend  # imported lazily: Qt is only needed for a window

        frontend = QtFrontend(
            channel=channel,
            control=control,
            registry=create_registry(process.commands),
            close_on_finish=close_on_finish,
            title=title,
        )
        result = frontend.run(manager)
    else:
        ConsoleFrontend(channel, log_every=log_every)
        result = manager.run()

    if result.state is TrainingState.FAILED and result.error is not None:
        error = result.error
        message = f"training failed: {type(error).__name__}: {error}"
        raise TrainingFailedError(message, result) from error
    return result

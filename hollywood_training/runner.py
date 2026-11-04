"""``run()``: the management method. Hands a :class:`TrainingProcess` to the manager and a
frontend - the native GUI window by default - and returns when the session is over.

=====================  =====================================================================
frontend               when
=====================  =====================================================================
window process         default (``gui=True``): the window runs in its own process
in-process window      ``isolate_gui=False``: e.g. inside an existing Qt application
console                ``gui=False``: no display needed
=====================  =====================================================================
"""

from __future__ import annotations

import contextlib
import os
from collections.abc import Iterator

import torch

from .commands.builtin import create_registry
from .console import ConsoleFrontend
from .sync.channel import SyncChannel
from .sync.control import TrainingControl
from .sync.events import TrainingState
from .tracking.tracker import NetworkTracker
from .tracking.weights import DEFAULT_NEURON_LIMIT
from .training.history import TrainingResult
from .training.manager import TrainingManager
from .training.process import TrainingProcess


def cpu_cores() -> int:
    """CPU cores this process may run on (its CPU affinity mask, e.g. taskset or cpusets;
    CPU *quotas* such as ``docker --cpus`` are not visible here)."""
    try:
        return max(1, len(os.sched_getaffinity(0)))
    except (AttributeError, OSError):  # not available on macOS / Windows
        return max(1, os.cpu_count() or 1)


@contextlib.contextmanager
def _torch_threads(setting: int | str | None) -> Iterator[int]:
    """Temporarily set PyTorch's intra-op CPU threads; restores the previous value.

    ``"auto"`` caps the count at half the CPU cores (never raising it). By default
    PyTorch runs one spin-waiting worker per core; while a window is drawing on the same
    machine, any core it takes stalls that pool's barriers and small-batch training can
    become several times slower. ``None`` leaves the setting alone; an int sets it
    explicitly. Yields the thread count in effect.
    """
    previous = torch.get_num_threads()
    if setting is None:
        yield previous
        return
    if setting == "auto":
        target = min(previous, max(1, cpu_cores() // 2))
    elif isinstance(setting, int) and not isinstance(setting, bool) and setting >= 1:
        target = setting
    else:
        raise ValueError(f"torch_threads must be 'auto', None or a positive int, got {setting!r}")
    if target == previous:
        yield previous
        return
    torch.set_num_threads(target)
    try:
        yield target
    finally:
        torch.set_num_threads(previous)


def _check_options(torch_threads: object, log_every: object) -> None:
    """Validate options up front, whichever frontend ends up using them."""
    if torch_threads not in (None, "auto") and (
        isinstance(torch_threads, bool) or not isinstance(torch_threads, int) or torch_threads < 1
    ):
        raise ValueError(
            f"torch_threads must be 'auto', None or a positive int, got {torch_threads!r}"
        )
    if log_every is not None and (
        isinstance(log_every, bool) or not isinstance(log_every, int) or log_every < 1
    ):
        raise ValueError(f"log_every must be a positive int, got {log_every!r}")


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
    isolate_gui: bool = True,
    start_paused: bool = False,
    close_on_finish: bool = False,
    max_neurons: int = DEFAULT_NEURON_LIMIT,
    title: str | None = None,
    log_every: int | None = None,
    torch_threads: int | str | None = "auto",
) -> TrainingResult:
    """Run ``process`` under management and visualise it.

    Args:
        process: the wrapped training process.
        gui: open the native GUI window (default). ``False`` prints to the console instead,
            for machines without a display.
        isolate_gui: run the window in its own process (default), so that drawing never
            slows training down. ``False`` keeps it in this process - needed to embed it
            in an existing Qt application or where a second process cannot be started.
        start_paused: open the window paused, so the initial network can be inspected;
            ``resume`` (or ``step``) in the terminal starts training. GUI only.
        close_on_finish: close the window by itself once training ends. Otherwise the
            window stays open (more epochs can be requested with ``epochs <n>``).
        max_neurons: neurons drawn per layer column; wider layers are sampled evenly.
        title: window title (default: derived from the network's class name).
        log_every: console only - print every n-th epoch.
        torch_threads: GUI only - PyTorch intra-op CPU threads while the window is open.
            ``"auto"`` (default) caps them at half the CPU cores, so the window's drawing
            cannot stall PyTorch's spin-waiting thread pool; ``None`` keeps your setting;
            an int sets it. Restored afterwards. For the very best speed on Linux, start
            Python with ``OMP_NUM_THREADS`` set (e.g. to half the cores) instead.

    Returns:
        The :class:`TrainingResult` with the per-epoch history.

    Raises:
        TrainingFailedError: when the training code raised (after the window was closed).
        ProcessError: when the process is incomplete (checked before anything starts).
    """
    if not isinstance(process, TrainingProcess):
        raise TypeError(f"run() expects a TrainingProcess, got {type(process).__name__}")
    _check_options(torch_threads, log_every)
    process.resolve_steps()  # fail fast, before any window opens
    if start_paused and not gui:
        raise ValueError("start_paused needs the GUI: there is no terminal to resume from")

    control = TrainingControl(process.epochs, start_paused=start_paused)
    channel = SyncChannel()
    tracker = NetworkTracker(process.model, max_neurons=max_neurons)

    if gui and isolate_gui:
        from .gui.remote import RemoteFrontend  # imported lazily: only needed for a window

        remote = RemoteFrontend(
            channel=channel,
            control=control,
            registry=create_registry(process.commands),
            close_on_finish=close_on_finish,
            title=title,
        )
        manager = TrainingManager(
            process, channel=channel, control=control, tracker=tracker, linger=not close_on_finish
        )
        with _torch_threads(torch_threads):
            result = remote.run(manager)
    elif gui:
        from .gui.app import QtFrontend

        frontend = QtFrontend(
            channel=channel,
            control=control,
            registry=create_registry(process.commands),
            close_on_finish=close_on_finish,
            title=title,
        )
        manager = TrainingManager(
            process,
            channel=channel,
            control=control,
            tracker=tracker,
            linger=not close_on_finish,
            pump=frontend.pump,
        )
        with _torch_threads(torch_threads):
            result = frontend.run(manager)
    else:
        ConsoleFrontend(channel, log_every=log_every)
        manager = TrainingManager(process, channel=channel, control=control, tracker=tracker)
        result = manager.run()

    if result.state is TrainingState.FAILED and result.error is not None:
        error = result.error
        message = f"training failed: {type(error).__name__}: {error}"
        raise TrainingFailedError(message, result) from error
    return result

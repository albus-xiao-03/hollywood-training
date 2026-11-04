"""Hollywood Training: watch PyTorch networks train in a Hollywood-style native GUI.

Three steps, all in ordinary PyTorch style::

    import hollywood_training as ht

    @ht.track                                   # 1. track the network
    class Net(nn.Module): ...

    process = ht.TrainingProcess(model, optimizer, train_loader,
                                 loss_fn=loss_fn, val_data=val_loader, epochs=100)

    @process.train_step                         # 2. wrap your training code
    def train_step(batch): ...

    ht.run(process)                             # 3. hand it to the manager
"""

from importlib import import_module

from ._version import __version__
from .commands.registry import Argument, Command, CommandError
from .sync.events import EpochRecord, TrainingState

_LAZY = {
    "CommandContext": ".commands.context",
    "NetworkTracker": ".tracking.tracker",
    "ProcessError": ".training.process",
    "TrainingFailedError": ".runner",
    "TrainingProcess": ".training.process",
    "TrainingResult": ".training.history",
    "inspect_network": ".tracking.tracker",
    "is_tracked": ".tracking.decorator",
    "run": ".runner",
    "track": ".tracking.decorator",
}

__all__ = [
    "Argument",
    "Command",
    "CommandContext",
    "CommandError",
    "EpochRecord",
    "NetworkTracker",
    "ProcessError",
    "TrainingFailedError",
    "TrainingProcess",
    "TrainingResult",
    "TrainingState",
    "__version__",
    "inspect_network",
    "is_tracked",
    "run",
    "track",
]


def __getattr__(name: str):
    # Exports that need torch are imported on first use (PEP 562), so the torch-free
    # data modules of this package can be used on their own - e.g. by the GUI process.
    module = _LAZY.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(module, __name__), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(__all__)

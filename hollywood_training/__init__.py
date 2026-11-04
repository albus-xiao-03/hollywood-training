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

from ._version import __version__
from .commands import Argument, Command, CommandContext, CommandError
from .runner import TrainingFailedError, run
from .sync import EpochRecord, TrainingState
from .tracking import NetworkTracker, inspect_network, is_tracked, track
from .training import ProcessError, TrainingProcess, TrainingResult

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

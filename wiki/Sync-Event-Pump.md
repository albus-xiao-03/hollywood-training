# Sync Event Pump

`hollywood_training/sync/pump.py` · part of [Synchronization](Synchronization)

An event pump is how the training loop gives a frontend time to run. The
[Training Manager](Training-Manager) calls `poll()` at every safe point between batches and
`wait(timeout)` whenever it has nothing to do for a moment. This is what keeps the
in-process window responsive while it shares the main thread with training. The default
`WaitingPump` just sleeps until a control request arrives.

## Responsibilities

* Define the `EventPump` protocol that the manager calls.
* Provide `WaitingPump` for frontends that need no event loop on the training thread.

## API

```python
class EventPump(Protocol):
    def poll(self) -> None:
        """Let the frontend handle pending events; return quickly."""

    def wait(self, timeout: float) -> None:
        """Let the frontend run for up to ``timeout`` seconds (may return earlier)."""
```

A `typing.Protocol` (not `runtime_checkable`). Any object with these two methods
qualifies; no inheritance needed.

```python
class WaitingPump:
    def __init__(self, control: TrainingControl) -> None: ...
    def poll(self) -> None: ...
    def wait(self, timeout: float) -> None: ...
```

`poll()` does nothing. `wait(timeout)` calls `control.wait(timeout)`. It returns after
`timeout` seconds, or as soon as any request reaches the [TrainingControl](Sync-Control)
(pause, resume, step, stop, a setting, a queued action, or `close()`).

## How it works

All pump calls happen on the training thread. The manager calls the pump at these points:

| Call | Where in the manager | Purpose |
|---|---|---|
| `poll()` | `_checkpoint`, before every epoch and after every batch | handle pending input |
| `wait(0.1)` | while paused, until resumed or stopped | idle without spinning |
| `wait(0.1)` | while lingering in `FINISHED` (`linger=True`), until more epochs or a stop are requested | idle without spinning |
| `wait(min(remaining, 0.1))` | during the per-epoch delay (`delay <ms>`) | sleep in short slices |

After every `wait` the manager applies pending requests and re-checks the condition it
waits for, so a pump may return early at any time.

| Frontend | Pump | `poll()` | `wait(timeout)` |
|---|---|---|---|
| [Console](Console-Frontend) | `WaitingPump` (default) | no-op | `control.wait(timeout)` |
| [Window process](GUI-Remote-Frontend) (default GUI) | `WaitingPump` (default) | no-op | `control.wait(timeout)`; commands run on the training process's command thread and wake it through the control |
| [In-process window](GUI-In-Process-Frontend) | `QtPump` (`gui/app.py`) | `QApplication.processEvents()`, at most every 4 ms | runs a nested `QEventLoop` for `timeout`; `interrupt()` ends it early after a command and when the window closes |

Sharing the thread keeps PyTorch on the thread that created the data and the model, and
the GUI never competes with training for the interpreter lock.

## Used by / uses

* Uses [`TrainingControl`](Sync-Control) (`WaitingPump`).
* [Training Manager](Training-Manager): `TrainingManager(..., pump=None)` uses
  `WaitingPump(control)` when no pump is passed.
* [Runner](Runner): passes the frontend's `QtPump` only for `run(..., isolate_gui=False)`.

## Example

A pump that counts polls and waits like `WaitingPump`, useful in tests:

```python
from hollywood_training.sync import SyncChannel, TrainingControl, WaitingPump
from hollywood_training.training import TrainingManager

class CountingPump:                      # satisfies EventPump structurally
    def __init__(self, control: TrainingControl) -> None:
        self.polls = 0
        self._waiting = WaitingPump(control)

    def poll(self) -> None:
        self.polls += 1

    def wait(self, timeout: float) -> None:
        self._waiting.wait(timeout)

control = TrainingControl(process.epochs)          # process: a ht.TrainingProcess
channel = SyncChannel()
channel.connect(lambda e: e.synchronous and channel.acknowledge(e.sequence))
pump = CountingPump(control)
TrainingManager(process, channel=channel, control=control, pump=pump).run()
print(pump.polls)                # after every batch (both phases) and before every epoch
```

## Notes

* `poll()` runs after every batch, so it must be cheap. `QtPump` rate-limits it.
* Only the in-process window shares the training thread. In the default isolated GUI the
  window has its own process, and the manager uses `WaitingPump`.

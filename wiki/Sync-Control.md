# Sync Control

`hollywood_training/sync/control.py` · part of [Synchronization](Synchronization)

`TrainingControl` carries runtime requests from a frontend to the training thread.
Commands only *request* changes through it, and the training thread applies them at safe
points between batches. Anything that must touch the model or optimizer is queued as an
*action*, a callable the training thread runs with a `TrainingTarget`, so it never races
with an optimisation step. All methods are thread-safe.

## Responsibilities

* Hold the requested pause / stop / step state, the total number of epochs and the
  per-epoch delay.
* Queue actions for the training thread and hand back the ones that never ran.
* Count changes (`revision`) and wake a training thread that waits for requests.

## API

```python
@dataclass(frozen=True, slots=True)
class TrainingTarget:
    model: Any  # torch.nn.Module
    optimizer: Any  # torch.optim.Optimizer

ControlAction = Callable[[TrainingTarget], "str | None"]
```

`TrainingTarget` is what a queued action may touch; it is handed to the action on the
training thread. An action returns a message for the user, or `None`.

```python
class TrainingControl:
    def __init__(self, total_epochs: int, *, start_paused: bool = False) -> None: ...

    # requests (any thread)
    def pause(self) -> None: ...
    def resume(self) -> None: ...
    def step(self, epochs: int = 1) -> None: ...
    def stop(self) -> None: ...
    def set_total_epochs(self, total: int) -> None: ...
    def set_delay(self, seconds: float) -> None: ...
    def submit(self, action: ControlAction, description: str = "") -> bool: ...

    # state (read-only properties)
    paused: bool
    stop_requested: bool
    closed: bool
    total_epochs: int
    delay: float
    completed_epochs: int
    revision: int
    stepping_until: int | None

    # training-thread side
    def take_actions(self) -> list[tuple[ControlAction, str]]: ...
    def epoch_completed(self, epoch: int) -> None: ...
    def wait(self, timeout: float) -> None: ...
    def close(self) -> list[tuple[ControlAction, str]]: ...
```

The constructor raises `ValueError("total_epochs must be at least 1")` when
`int(total_epochs) < 1`.

| Request | Effect | Raises |
|---|---|---|
| `pause()` | sets *paused*, cancels a running `step`; ignored after `stop()` | |
| `resume()` | clears *paused*, cancels a running `step` | |
| `step(epochs)` | clears *paused*; pauses again once `completed_epochs + epochs` epochs are complete | `ValueError` if `epochs < 1` |
| `stop()` | sets *stop requested*, clears *paused* | |
| `set_total_epochs(total)` | sets the total | `ValueError` if `total < max(1, completed_epochs)` |
| `set_delay(seconds)` | extra wait after every epoch (0 = as fast as synchronisation allows) | `ValueError("delay must be a finite, non-negative number of seconds")` if negative, NaN or infinite |
| `submit(action, description)` | queues `action`; returns `False` once closed | |

| State | Meaning |
|---|---|
| `closed` | the training thread no longer runs queued actions |
| `delay` | seconds |
| `revision` | increases with every request; lets the training thread notice changes cheaply |
| `stepping_until` | epoch after which an active `step` pauses, else `None` |

| Training-thread side | Effect |
|---|---|
| `take_actions()` | removes and returns every queued `(action, description)` |
| `epoch_completed(epoch)` | records progress; turns a finished `step` into a pause |
| `wait(timeout)` | sleeps until any request arrives or `timeout` seconds pass |
| `close()` | marks the control closed; returns the actions that never ran |

## How it works

* One `threading.Condition` guards every field. Each request that is not ignored or
  refused, and `close()`, increments `revision` and wakes all threads in `wait()`.
  `take_actions()` and `epoch_completed()` do not change `revision`.
* The [Training Manager](Training-Manager) applies requests at every safe point (before
  each epoch and after each batch) and after every wait in its loops while paused,
  lingering after the last epoch, or sleeping the per-epoch delay. Applying means:
  1. run each queued action with `TrainingTarget(process.model, process.optimizer)`. A
     returned message becomes a `SUCCESS` `LogMessage`; an exception becomes an `ERROR`
     log `"<description> failed: <Type>: <message>"` (`action` when the description is
     empty), and training continues;
  2. publish a `StateChanged` with the current settings if `revision` differs from the
     last one the manager saw.

  At a safe point the manager then ends the session if a stop was requested, or, while
  *paused*, delivers `TrainingPaused` and waits on the [event pump](Sync-Event-Pump).
* After each `EpochCompleted` hand-off the manager calls `epoch_completed(epoch)`, then
  sleeps `delay` seconds if it is positive (ended early by a pause or stop).
* When the training thread exits, the manager calls `close()` and logs every returned
  action that has a description as a `WARNING`: `"not applied (session ended): …"`. From
  then on `submit` returns `False` and [`CommandContext.submit`](Commands-Context) runs
  actions directly.

| Command | Request |
|---|---|
| `pause`, `resume`, `step [n]` | `pause()`, `resume()`, `step(n)` |
| `stop`, `quit` | `stop()` |
| `epochs <n>`, `delay <ms>` | `set_total_epochs(n)`, `set_delay(ms / 1000)` |
| `lr`, `save`, custom commands | `submit(action, description)` through `CommandContext.submit` |

Once `closed` is set, the built-in `pause`, `resume`, `step`, `epochs` and `delay` refuse
with `the training session has ended`, and `stop` answers
`the training session has already ended` ([Built-in Commands](Commands-Builtin)).

## Used by / uses

* No dependencies besides the standard library.
* Created by `run()` as `TrainingControl(process.epochs, start_paused=start_paused)`
  ([Runner](Runner)), or by `TrainingManager` when none is passed.
* Applied by the [Training Manager](Training-Manager); waited on by `WaitingPump`.
* Requests come from [commands](Commands) through [Commands Context](Commands-Context), and
  from the frontends: `stop()` when the window is closed or the link is lost
  ([GUI Remote Frontend](GUI-Remote-Frontend), [GUI In-Process Frontend](GUI-In-Process-Frontend)),
  and on Ctrl+C in the in-process frontend.

## Example

```python
from hollywood_training.sync import TrainingControl

control = TrainingControl(10, start_paused=True)

def set_lr(target):                         # runs on the training thread
    for group in target.optimizer.param_groups:
        group["lr"] = 0.01
    return "learning rate changed"          # becomes a SUCCESS log message

control.submit(set_lr, "set learning rate")
control.step(3)                             # run three epochs, then pause again
assert not control.paused and control.stepping_until == 3
# pass `control` to TrainingManager(process, channel=..., control=control)
```

## Notes

* Properties read single attributes without taking the lock.
* `stop()` wins over a pause: it clears *paused*, and `pause()` afterwards is ignored.
* `step(n)` counts from the completed epochs, so an epoch in progress is the first of
  the `n`.
* `TrainingTarget` types its fields as `Any`; the module imports no torch.

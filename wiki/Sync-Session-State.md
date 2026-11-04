# Sync Session State

`hollywood_training/sync/session.py` · part of [Synchronization](Synchronization)

`SessionState` is a frontend's picture of the session, built from the event stream. A
frontend applies every event it receives and then reads `SessionState` to draw; commands
read it too (`status`, `history`, ...). Version counters tell views when cached drawings
must be rebuilt. It is not locked: one thread applies the events.

## Responsibilities

* Fold each event into plain attributes (`apply`).
* Keep the epoch history and the latest architecture and weights.
* Provide derived values: best epoch, progress fractions, a smoothly advancing timer, ETA.

## API

```python
class SessionState:
    def __init__(self) -> None: ...
    def apply(self, event: Event) -> None: ...
    @property
    def network_name(self) -> str: ...
    def elapsed_now(self) -> float: ...
    @property
    def last_record(self) -> EpochRecord | None: ...
    def best_record(self) -> EpochRecord | None: ...
    def epoch_progress(self) -> float: ...
    def batch_progress(self) -> float | None: ...
    def eta(self) -> float | None: ...
```

Public attributes and what updates them:

| Attribute | Type | Initial | Updated by |
|---|---|---|---|
| `info` | `SessionInfo \| None` | `None` | `SessionStarted` |
| `architecture` | `ArchitectureInfo \| None` | `None` | `SessionStarted`; `EpochCompleted.architecture` when not `None` |
| `weights` | `WeightSnapshot \| None` | `None` | `SessionStarted`, `EpochCompleted`, `TrainingPaused`; `SessionFinished` when it has weights |
| `history` | `list[EpochRecord]` | `[]` | `EpochCompleted` appends its record |
| `state` | `TrainingState` | `PREPARING` | `SessionStarted`, `TrainingPaused` (sets `PAUSED`), `StateChanged`, `SessionFinished` |
| `epoch` | `int` | `0` | completed epochs: `EpochCompleted`, `StateChanged` |
| `total_epochs` | `int` | `0` | `SessionStarted`, `EpochCompleted`, `StateChanged` |
| `learning_rate` | `float` | `nan` | `SessionStarted`, `EpochCompleted` (rate for the next epoch), `StateChanged` |
| `batch_size` | `int \| None` | `None` | `SessionStarted`; `EpochCompleted` when not `None`; `BatchProgress` only while still `None` |
| `delay` | `float` | `0.0` | `StateChanged` (seconds) |
| `phase`, `batch` | `Phase \| None`, `int` | `None`, `0` | `BatchProgress`; reset by `EpochCompleted` |
| `total_batches` | `int \| None` | `None` | `SessionStarted` (training batches); `BatchProgress` (current phase) |
| `batch_loss` | `float \| None` | `None` | `BatchProgress.loss` |
| `finished` | `SessionFinished \| None` | `None` | `SessionFinished` |
| `architecture_version` | `int` | `0` | +1 on `SessionStarted` and on a refined architecture |
| `weights_version` | `int` | `0` | +1 whenever `weights` is replaced |
| `history_version` | `int` | `0` | +1 on `SessionStarted` and on every `EpochCompleted` |

Queries:

| Query | Returns |
|---|---|
| `network_name` | `info.network_class`, or `""` before `SessionStarted` |
| `elapsed_now()` | active training seconds: the last event's `elapsed`, plus the time since that event while `state` is `RUNNING` |
| `last_record` | the latest `EpochRecord`, or `None` |
| `best_record()` | `best_epoch(history)`: the record with the lowest finite `monitored_loss` (`val_loss`, else `train_loss`); NaN and infinite losses are skipped; `None` when no record qualifies |
| `epoch_progress()` | `min(1.0, epoch / total_epochs)`; `0.0` when `total_epochs <= 0` |
| `batch_progress()` | `min(1.0, batch / total_batches)`; `None` when `total_batches` is `None` or `0` |
| `eta()` | `elapsed_now() / epoch * max(0, total_epochs - epoch)`; `None` before the first epoch or when not `RUNNING` |

## How it works

* `SessionStarted` resets every attribute except the version counters, then takes the
  info, architecture, totals, learning rate, batch size, training batch count, state and
  initial weights from the event, and sets the timer to 0.
* `LogMessage` is ignored: the frontend shows the text, the state does not store it.
  Objects that are not events are ignored as well.
* Version counters only ever increase, even across a reset for a new session, so a view
  that cached a drawing for version N never mistakes it for current.
* Timer: every other event stores its `elapsed` together with a `time.monotonic()`
  anchor. `elapsed_now()` adds the time since the anchor only while `RUNNING`, so the
  process timer advances smoothly between events and stands still while paused.

## Used by / uses

* Uses the events and records of [Sync Events](Sync-Events).
* Each frontend holds one: [Console Frontend](Console-Frontend),
  [GUI In-Process Frontend](GUI-In-Process-Frontend), [GUI Window Process](GUI-Window-Process),
  and [GUI Remote Frontend](GUI-Remote-Frontend), whose copy in the training process mirrors
  what the window shows so that commands can read it.
* Views: [GUI Header](GUI-Header) (`network_name`, `state`), [GUI Network View](GUI-Network-View)
  (`architecture_version`, `weights_version`), [GUI Loss Chart](GUI-Loss-Chart) (`history`,
  `history_version`, `best_record()`), [GUI Status Panel](GUI-Status-Panel)
  (`elapsed_now()`, `epoch_progress()`, `batch_progress()`, `eta()`).
* [Commands](Commands) read it through `context.session`. The built-in `epochs` and `delay`
  commands also write `total_epochs` and `delay` directly, so that `status` shows the new
  value before the next `StateChanged` arrives.

## Example

```python
from hollywood_training.sync import SessionState, SyncChannel
from hollywood_training.training import TrainingManager

channel, session = SyncChannel(), SessionState()

def sink(envelope):
    session.apply(envelope.event)
    if envelope.synchronous:
        channel.acknowledge(envelope.sequence)

channel.connect(sink)
TrainingManager(process, channel=channel).run()   # process: a ht.TrainingProcess, 4 epochs

print(session.network_name, session.state.value, f"{session.epoch}/{session.total_epochs}")
print(session.best_record().epoch, session.finished.reason)   # e.g. 4 'completed 4 epochs'
print(session.epoch_progress(), session.eta())                # 1.0 None (no longer running)
```

## Notes

* Not locked: one thread applies events. The window-process frontend's mirror in the
  training process is written by the training thread (in the sink) and read by commands on
  the training process's command thread, which also makes the two plain attribute writes
  of `epochs` and `delay`. This relies on the interpreter lock; a command may see values
  that are a moment stale.
* `best_record()` scans the whole history on each call.

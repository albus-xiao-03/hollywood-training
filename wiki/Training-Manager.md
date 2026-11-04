# Training Manager

`hollywood_training/training/manager.py` · part of [Training](Training)

`TrainingManager` runs a [`TrainingProcess`](Training-Process) epoch by epoch on the
thread that calls it, in lock-step with one frontend. It reports through a
[`SyncChannel`](Sync-Channel), blocking on synchronous events until the frontend has
drawn them. It applies [`TrainingControl`](Sync-Control) requests at safe points between
batches, and never touches a GUI itself.

## Responsibilities

* Run the epoch loop: train, validate, step the scheduler, call the epoch hooks.
* Snapshot the weights and hand each epoch to the frontend before the next one starts.
* Pump the frontend between batches and apply pause, step, stop, delay and queued actions.
* Optionally linger after the last epoch, so that `epochs <n>` can continue the run.
* Measure active time, and end every session with `SessionFinished` and a [`TrainingResult`](Training-Results).

## API

```python
class TrainingError(RuntimeError): ...
```

Raised when `train_data` or `val_data` yields no batches in an epoch
(`train_data produced no batches in epoch 1`). Like any exception, it fails the session.

```python
class TrainingManager:
    def __init__(
        self,
        process: TrainingProcess,
        *,
        channel: SyncChannel,
        control: TrainingControl | None = None,
        tracker: NetworkTracker | None = None,
        linger: bool = False,
        progress_interval: float = 1 / 30,
        pump: EventPump | None = None,
    ) -> None
```

| Parameter | Meaning |
|---|---|
| `process` | the wrapped training process |
| `channel` | the hand-off to the frontend; without a connected sink the channel raises `RuntimeError` |
| `control` | runtime controls; default `TrainingControl(process.epochs)` |
| `tracker` | default `NetworkTracker(process.model)` |
| `linger` | after the last epoch, wait in `FINISHED` for `epochs <n>` until stopped |
| `progress_interval` | minimum seconds between `BatchProgress` events |
| `pump` | the [event pump](Sync-Event-Pump); default `WaitingPump(control)` |

Properties: `process`, `control`, `tracker`, `state -> TrainingState` (`PREPARING`
until the start), `history -> tuple[EpochRecord, ...]`.

```python
def run(self) -> TrainingResult
def partial_result(self) -> TrainingResult
```

`run()` runs the session to its end and returns a result with state `FINISHED`,
`STOPPED` or `FAILED`. Exceptions are caught and stored as `result.error`; a
`KeyboardInterrupt` is re-raised after `SessionFinished` was delivered.
`partial_result()` returns `TrainingResult(STOPPED, history, elapsed)` for the epochs
completed so far. The GUI frontends use it when `run()` is interrupted.

## How it works

```
run()
├─ process.resolve_steps()
├─ start: SessionInfo + snapshot(0) → deliver(SessionStarted)   → RUNNING (PAUSED if start_paused)
├─ loop: all requested epochs done? → linger, or end
│        checkpoint ─ stop? → end
│        run epoch n → finish epoch n
└─ deliver(SessionFinished) → TrainingResult
```

The completion check comes first. The epoch-boundary checkpoint, and any pause there,
therefore happens only before an epoch that will actually run.

**One epoch:**

1. Read the learning rate; it becomes `EpochRecord.learning_rate`.
2. **Train:** `model.train(True)` under `torch.enable_grad()`; `train_step(batch)` per batch.
3. **Validate** (if configured): `model.train(False)` (that is, `eval()`) under
   `torch.no_grad()`; `validation_step(batch)` per batch.
4. Build the `EpochRecord`. Losses are batch means, each metric is averaged over the
   batches that returned it, and `duration` is the active time of both phases.
5. Step the scheduler. `ReduceLROnPlateau` gets `step(record.monitored_loss)`: the
   validation loss, or the training loss without validation
   ([Sync Events](Sync-Events)). Any other scheduler gets `step()`.
6. Call the `on_epoch_end` hooks with the record.
7. Append the record to `history` and take a baseline weight snapshot. If the forward
   trace refined the architecture, attach the new `ArchitectureInfo`.
8. **Hand-off:** `deliver(EpochCompleted)` blocks until the frontend acknowledges it,
   which it does after drawing.
9. Call `control.epoch_completed(n)`, which turns a finished `step` into a pause, then
   wait out the optional delay.

Per batch, the manager runs the step and passes the output to `read_step_output`. It
then warns about a non-finite loss (at most once per phase and epoch), infers the batch
size while it is unknown, and sends a throttled `BatchProgress` before the checkpoint.
The session's first training batch runs under a forward trace
([Tracking Network Tracker](Tracking-Network-Tracker)).

**Checkpoints** run before every epoch and after every batch. A checkpoint calls
`pump.poll()`, then runs the queued actions with a `TrainingTarget(model, optimizer)`. An
action's message is logged as `SUCCESS`. An exception is logged as `ERROR`
(`<description> failed: Type: message`) and the session continues. If
`control.revision` changed, the checkpoint publishes a `StateChanged` without a message.
It then honours a stop or pause request.

| Request | Effect |
|---|---|
| `pause` | The clock pauses. A non-baseline snapshot goes out in a synchronous `TrainingPaused`, followed by `StateChanged(PAUSED, "paused (epoch 3, train batch 7) - 'resume' or 'step' continues")`, or `paused (before epoch 4) ...` at a boundary. The manager waits in `pump.wait(0.1)` and keeps running queued actions. On resume the clock restarts and `StateChanged(RUNNING, "resumed")` is sent |
| `step n` | resumes; after `n` more epochs the control pauses again at the next epoch boundary. If those epochs end the run, the session finishes (or lingers) without pausing |
| `stop` | The phase returns, the epoch in progress produces no record, and the session ends `STOPPED`. A stop that arrives after all requested epochs have completed, for example while lingering, ends it `FINISHED` |
| `delay` | after every hand-off the manager waits in slices of at most 0.1 s through `pump.wait`, applying requests; the wait ends early on stop or pause |
| `epochs n` | the total is compared at every epoch boundary |

**Linger** (`linger=True`): once all requested epochs are done, the clock pauses. A
pause still pending (e.g. from `step`) is released with `control.resume()`, and
`StateChanged(FINISHED, "all 200 epochs complete - 'epochs <n>' trains further")` is
published. Requests are applied every 0.1 s. When the total grows, the clock restarts,
`StateChanged(RUNNING, "continuing to epoch 250")` is published, and the run continues.

**Lost frontend:** if a synchronous `deliver` returns `False`, the channel was closed
(for example, the window closed), and the manager calls `control.stop()`.

**Errors**

| Raised during the session | Outcome |
|---|---|
| `Exception` (step, hook, scheduler, `StepOutputError`, `TrainingError`, `ProcessError` from `resolve_steps`) | `FAILED`. The traceback is logged as `ERROR`, `SessionFinished.error` is `"Type: message"`, and `result.error` holds the exception |
| `KeyboardInterrupt` | `STOPPED`. `interrupted` is logged as `WARNING`, `SessionFinished` is delivered, then the interrupt is re-raised |
| other `BaseException` | propagates without `SessionFinished` |

In every case the clock stops and `control.close()` is called. Queued actions that have
a description are logged as `WARNING` `not applied (session ended): <description>`. The
final snapshot uses `baseline=False`, and `weights` is `None` if it fails.

**Events emitted**

| Event | Sync | When |
|---|---|---|
| `SessionStarted(info, weights, state)` | yes | once; `weights` is the epoch-0 snapshot |
| `LogMessage` | no | `training <class> for <n> epochs on <device>` (`1 epoch` in the singular) after the start; warnings, errors, action messages |
| `BatchProgress` | no | always for a phase's first and last batch (when the count is known), otherwise at most once per `progress_interval` |
| `StateChanged` | no | pause, resume, linger and continue (with a message); any control change (without one) |
| `TrainingPaused` | yes | when a pause takes effect; `epoch` is the epoch in progress, or the completed epochs at a boundary |
| `EpochCompleted` | yes | after every epoch; `learning_rate` is read after the scheduler step; `architecture` is set only when refined |
| `SessionFinished` | yes | once; `reason` such as `completed 3 epochs`, `stopped after 2 epochs`, `failed after 1 completed epoch` |

## Used by / uses

* Uses: [Sync Channel](Sync-Channel), [Sync Control](Sync-Control),
  [Sync Events](Sync-Events), [Sync Event Pump](Sync-Event-Pump),
  [Tracking Network Tracker](Tracking-Network-Tracker),
  [Training Descriptions](Training-Descriptions), [Training Steps](Training-Steps),
  [Training Active Clock](Training-Active-Clock) and [Training Results](Training-Results).
* Used by: [Runner](Runner), which creates it in every mode, and the
  [GUI Remote Frontend](GUI-Remote-Frontend) and
  [GUI In-Process Frontend](GUI-In-Process-Frontend), which call `run()`.

## Example

A minimal frontend: a sink that prints each epoch and acknowledges synchronous events
at once.

```python
from hollywood_training.sync import EpochCompleted, SyncChannel
from hollywood_training.training import TrainingManager

channel = SyncChannel()

def sink(envelope):                         # called on the training thread
    if isinstance(envelope.event, EpochCompleted):
        print(envelope.event.record.epoch, envelope.event.record.train_loss)
    if envelope.synchronous:
        channel.acknowledge(envelope.sequence)

channel.connect(sink)
result = TrainingManager(process, channel=channel).run()
```

## Notes

* Call `run()` on the thread that should train; your steps run there. Other threads
  reach the manager only through the thread-safe `TrainingControl` and the channel.
* `elapsed` is active time. It excludes pauses and lingering, but includes the
  per-epoch hand-off and the optional delay.
* Epoch losses and metrics are unweighted batch means, so a short last batch counts
  like a full one.
* A mid-epoch `stop` leaves that epoch's weight updates in the model. The final snapshot
  includes them.

# Training Results

`hollywood_training/training/history.py` · part of [Training](Training)

`TrainingResult` is what a managed run returns: the final state, the per-epoch history,
the active training time and, when available, the error and the final weights. It is an
immutable record that the [Training Manager](Training-Manager) builds once the session
is over, and that [`run()`](Runner) returns. It lets a script work with the outcome
after the window has closed, for example to print the best epoch.

## Responsibilities

* Carry the outcome of one session: its state, `EpochRecord` history, elapsed time,
  error and final weights.
* Offer convenient views: the epoch count, the loss lists and the best epoch.

## API

```python
@dataclass(frozen=True, slots=True, eq=False)
class TrainingResult:
    state: TrainingState
    history: tuple[EpochRecord, ...]
    elapsed: float  # active training time in seconds (pauses excluded)
    error: BaseException | None = None
    final_weights: WeightSnapshot | None = None
```

| Field | Meaning |
|---|---|
| `state` | `FINISHED`, `STOPPED` or `FAILED` ([Sync Events](Sync-Events)) |
| `history` | one `EpochRecord` per completed epoch, in order |
| `elapsed` | active training time in seconds, from the manager's [`ActiveClock`](Training-Active-Clock) |
| `error` | the exception that failed the session; `None` otherwise |
| `final_weights` | the weight snapshot taken at the end ([Tracking Weight Data](Tracking-Weight-Data)); `None` if it could not be taken |

```python
@property
def epochs_completed(self) -> int
@property
def train_losses(self) -> list[float]
@property
def val_losses(self) -> list[float | None]
def best_epoch(self) -> EpochRecord | None
```

* `epochs_completed` is `len(history)`.
* `train_losses` holds `record.train_loss` per epoch.
* `val_losses` holds `record.val_loss` per epoch, with `None` for epochs without
  validation.
* `best_epoch()` delegates to `best_epoch(history)` ([Sync Events](Sync-Events)). It
  returns the record with the lowest finite `monitored_loss`, which is the validation
  loss, or the training loss for a record without validation. NaN and infinite losses
  are ignored, and ties go to the earliest epoch. It returns `None` when no record has a
  finite loss, which includes an empty history.

## How it works

The manager creates the result at the end of `TrainingManager.run()`:

| Ending | `state` | `error` | `final_weights` |
|---|---|---|---|
| every requested epoch completed (also a stop while lingering) | `FINISHED` | `None` | snapshot |
| stopped early, or the frontend closed | `STOPPED` | `None` | snapshot |
| an exception in the session | `FAILED` | the exception | snapshot |

The final snapshot is taken with `baseline=False`, and the same object is sent in
`SessionFinished.weights`. `TrainingManager.partial_result()` builds
`TrainingResult(STOPPED, history, elapsed)` with no error and no weights. The GUI
frontends use it when `KeyboardInterrupt` interrupts `manager.run()`, and they re-raise
the interrupt afterwards.

`run()` never returns a `FAILED` result. It raises
[`TrainingFailedError`](Runner) instead, and the result is available as
`exc.result`.

## Used by / uses

* Uses: `EpochRecord`, `TrainingState` and `best_epoch` ([Sync Events](Sync-Events)), and
  `WeightSnapshot` ([Tracking Weight Data](Tracking-Weight-Data)).
* Used by: the [Training Manager](Training-Manager), which creates it; [Runner](Runner),
  which returns it or wraps it in `TrainingFailedError`; and the
  [GUI Remote Frontend](GUI-Remote-Frontend) and
  [GUI In-Process Frontend](GUI-In-Process-Frontend), which pass it through.

## Example

```python
import hollywood_training as ht

result = ht.run(process)
print(f"{result.state.value}: {result.epochs_completed} epochs in {result.elapsed:.1f} s")

best = result.best_epoch()
if best is not None:
    accuracy = best.val_metrics.get("accuracy")
    print(f"best epoch {best.epoch}: validation loss {best.val_loss}, accuracy {accuracy}")

try:
    ht.run(failing_process)
except ht.TrainingFailedError as exc:
    print(exc.result.state, exc.result.epochs_completed, repr(exc.result.error))
```

## Notes

* `eq=False`: two results are equal only if they are the same object.
* `elapsed` excludes pauses and the time spent lingering after the last epoch. It
  includes the per-epoch hand-off to the frontend and any `delay`.

# Training Active Clock

`hollywood_training/training/timing.py` · part of [Training](Training)

`ActiveClock` is a pausable stopwatch based on `time.perf_counter`. It measures the wall
time spent actively training, excluding pauses. The
[Training Manager](Training-Manager) owns one per session. Its reading is the `elapsed`
value in every event and in the [`TrainingResult`](Training-Results), and the source of
each `EpochRecord.duration`. The process timer of the
[status panel](GUI-Status-Panel) shows this time, so it stands still while training is
paused.

## Responsibilities

* Accumulate running time across any number of start and pause cycles.
* Report the elapsed time at any moment, whether running or paused.

## API

```python
class ActiveClock:
    def __init__(self) -> None

    @property
    def running(self) -> bool

    def start(self) -> None
    def pause(self) -> None
    def elapsed(self) -> float
```

| Member | Behaviour |
|---|---|
| `ActiveClock()` | a new clock is paused and reads `0.0` |
| `running` | `True` between `start()` and `pause()` |
| `start()` | starts counting; does nothing if already running |
| `pause()` | adds the running interval to the total and stops counting; does nothing if already paused |
| `elapsed()` | seconds accumulated, plus the current interval while running |

There is no reset; a new session gets a new manager and therefore a new clock.

## How it works

The clock keeps an accumulated total and the `perf_counter()` value of the last
`start()`, which is `None` while paused. The manager drives it like this:

| Moment | Clock |
|---|---|
| `SessionStarted` acknowledged and the state is `RUNNING` | `start()` |
| a pause takes effect (before `TrainingPaused` is sent) | `pause()` |
| resume, or a `step` request | `start()` |
| all requested epochs done and the manager lingers | `pause()` |
| more epochs requested while lingering | `start()` |
| session end, success or not | `pause()` |

`EpochRecord.duration` is the difference between two `elapsed()` readings: one at the
start of the epoch and one after its validation phase. A pause inside an epoch is
therefore not counted. Frontends receive only these readings. Between events,
[Sync Session State](Sync-Session-State) advances its own copy (`elapsed_now()`) while
the state is `RUNNING`.

## Used by / uses

* Uses: `time.perf_counter` only. The module imports neither torch nor Qt.
* Used by: the [Training Manager](Training-Manager). It is re-exported as
  `hollywood_training.training.ActiveClock`.

## Example

```python
import time
from hollywood_training.training import ActiveClock

clock = ActiveClock()
clock.start()
time.sleep(0.2)
clock.pause()             # time from here on is not counted
time.sleep(1.0)
clock.start()
time.sleep(0.1)
print(f"{clock.elapsed():.1f} s, running={clock.running}")   # 0.3 s, running=True
```

## Notes

* The clock is not thread-safe: it uses plain attributes and no lock. The manager uses
  it only on the training thread.

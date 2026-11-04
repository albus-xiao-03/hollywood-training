# Sync Events

`hollywood_training/sync/events.py` · part of [Synchronization](Synchronization)

The events the training thread sends to a frontend, and the records they carry. Every
class is an immutable slot dataclass with plain data only (no modules, no tensors), so
an event can cross threads safely and be pickled to the window process. Four events are
synchronous: the training thread waits until the frontend has applied and drawn them.

## Responsibilities

* Lifecycle, phase and log-level enums.
* The records `EpochRecord` (one completed epoch) and `SessionInfo` (static description).
* The seven events, the `Event` union and the synchronous classification.

## API

### Enums

```python
class TrainingState(str, Enum):
    PREPARING = "preparing"
    RUNNING = "running"
    PAUSED = "paused"
    FINISHED = "finished"  # all requested epochs completed (more may still be requested)
    STOPPED = "stopped"  # ended early on request
    FAILED = "failed"  # user code raised an exception

    @property
    def active(self) -> bool: ...  # True for PREPARING, RUNNING and PAUSED

class Phase(str, Enum):
    TRAIN = "train"
    VALIDATION = "validation"

class LogLevel(str, Enum):
    INFO = "info"
    SUCCESS = "success"
    WARNING = "warning"
    ERROR = "error"
```

The enums subclass `str`, so `TrainingState.RUNNING == "running"`. The GUI terminal
colours a log line by its `LogLevel.value`.

### Records

```python
@dataclass(frozen=True, slots=True)
class EpochRecord:
    epoch: int
    train_loss: float
    val_loss: float | None
    train_metrics: Mapping[str, float] = field(default_factory=dict)
    val_metrics: Mapping[str, float] = field(default_factory=dict)
    learning_rate: float = 0.0  # learning rate used during the epoch
    duration: float = 0.0  # seconds spent training + validating
    train_batches: int = 0
    val_batches: int = 0

    @property
    def monitored_loss(self) -> float: ...

def best_epoch(records: Iterable[EpochRecord]) -> EpochRecord | None: ...
```

Results of one completed epoch. `epoch` starts at 1. The losses are means over the
epoch's batches (`val_loss` is `None` without validation), and the metric mappings hold
the per-batch means of the extra values a step returned. `learning_rate` is the first
parameter group's rate at the start of the epoch; `duration` is active time.

`monitored_loss` is the loss that decides "best": `val_loss` if there is one, else
`train_loss`. `best_epoch(records)` returns the record with the lowest *finite*
`monitored_loss`; NaN and infinite losses are skipped, and the result is `None` when no
record has a finite one (or there are no records).

```python
@dataclass(frozen=True, slots=True)
class SessionInfo:
    network_class: str
    display_name: str | None
    architecture: ArchitectureInfo
    total_epochs: int
    batch_size: int | None
    train_batches: int | None
    val_batches: int | None
    has_validation: bool
    loss_function: str
    optimizer: str
    optimizer_details: str
    normalization: str
    scheduler: str | None
    device: str
    learning_rate: float
    neuron_limit: int
```

Static description of a session, built by `build_session_info`
([Training Descriptions](Training-Descriptions)). Examples of the text fields:

| Field | Value |
|---|---|
| `display_name` | `TrainingProcess(name=...)`, else the `@track` name, else `None` |
| `batch_size`, `train_batches`, `val_batches` | explicit `batch_size` or `train_data.batch_size`; `len()` of each data source when it has one |
| `loss_function` | `"CrossEntropyLoss (mean)"`; `"defined in step"` without `loss_fn` |
| `optimizer`, `optimizer_details` | `"Adam"`, `"betas=0.9/0.999"` |
| `normalization` | `"data: z-score · layers: BatchNorm1d ×2"` |
| `device` | `"cpu (8 threads)"`, `"cuda:0 (<GPU name>)"` |
| `neuron_limit` | neurons drawn per layer column (`run(max_neurons=...)`) |

### Events

```python
@dataclass(frozen=True, slots=True, eq=False)
class SessionStarted:
    info: SessionInfo
    weights: WeightSnapshot
    state: TrainingState

@dataclass(frozen=True, slots=True)
class BatchProgress:
    epoch: int
    phase: Phase
    batch: int
    total_batches: int | None
    loss: float | None
    batch_size: int | None
    elapsed: float

@dataclass(frozen=True, slots=True, eq=False)
class EpochCompleted:
    record: EpochRecord
    weights: WeightSnapshot
    total_epochs: int
    learning_rate: float  # learning rate for the next epoch (after any scheduler step)
    batch_size: int | None
    elapsed: float
    architecture: ArchitectureInfo | None = None  # set when the architecture was refined

@dataclass(frozen=True, slots=True, eq=False)
class TrainingPaused:
    epoch: int  # epoch in progress (completed epochs + 1) or completed epochs at a boundary
    phase: Phase | None
    batch: int
    weights: WeightSnapshot
    elapsed: float

@dataclass(frozen=True, slots=True)
class StateChanged:
    state: TrainingState
    epoch: int  # completed epochs
    total_epochs: int
    learning_rate: float
    delay: float
    elapsed: float
    message: str = ""

@dataclass(frozen=True, slots=True)
class LogMessage:
    text: str
    level: LogLevel = LogLevel.INFO

@dataclass(frozen=True, slots=True, eq=False)
class SessionFinished:
    state: TrainingState
    epochs_completed: int
    elapsed: float
    weights: WeightSnapshot | None
    reason: str
    error: str | None = None
```

The four events with a `WeightSnapshot` (numpy arrays) use `eq=False` and compare by
identity. `elapsed` is always active training time in seconds, pauses excluded
([Training Active Clock](Training-Active-Clock)). How the [manager](Training-Manager) fills them:

| Event | Sent | Values |
|---|---|---|
| `SessionStarted` | delivered once | weights after 0 epochs; `state` is `PAUSED` with `start_paused`, else `RUNNING` |
| `BatchProgress` | notified after a batch | `batch` counts from 1; always sent for the first batch and, when the count is known, the last, otherwise at most every `progress_interval` (default 1/30 s) |
| `EpochCompleted` | delivered after every epoch | `architecture` is set when tracing the first training batch changed the architecture |
| `TrainingPaused` | delivered when a pause takes effect | at an epoch boundary: completed epochs, `phase=None`, `batch=0` |
| `StateChanged` | notified | on transitions with a `message` (e.g. `"resumed"`), and with an empty message after setting changes; `delay` in seconds |
| `LogMessage` | notified | the start line (`INFO`); warnings (non-finite loss, interrupt, actions never applied); errors (tracebacks, failed actions); messages of queued actions (`SUCCESS`) |
| `SessionFinished` | delivered once | `state` is `FINISHED`, `STOPPED` or `FAILED`; `reason` e.g. `"completed 3 epochs"`, `"stopped after 2 epochs"`; `error` is `"<Type>: <message>"` when user code raised; `weights` is `None` if the final snapshot failed |

### Union and classification

```python
Event = (
    SessionStarted | BatchProgress | EpochCompleted | TrainingPaused
    | StateChanged | LogMessage | SessionFinished
)

SYNCHRONOUS_EVENTS: tuple[type, ...] = (SessionStarted, EpochCompleted, TrainingPaused, SessionFinished)

def is_synchronous(event: object) -> bool: ...  # isinstance(event, SYNCHRONOUS_EVENTS)
```

## How it works

The module declares data, plus the `best_epoch` helper. The
[Training Manager](Training-Manager) creates the events and sends the four synchronous
ones with `deliver`, the rest with `notify`.
[`SyncChannel`](Sync-Channel) does not consult `is_synchronous`; the envelope's
`synchronous` flag records which method was used.

## Used by / uses

* Uses `ArchitectureInfo` ([Tracking Architecture Model](Tracking-Architecture-Model)) and
  `WeightSnapshot` ([Tracking Weight Data](Tracking-Weight-Data)), both torch-free.
* Produced by the [Training Manager](Training-Manager). `on_epoch_end` hooks receive the
  `EpochRecord` ([Training Process](Training-Process)); `TrainingResult` holds the records
  and the final `TrainingState` ([Training Results](Training-Results)).
* Consumed by [Sync Session State](Sync-Session-State), the [Console Frontend](Console-Frontend),
  the [GUI Main Window](GUI-Main-Window) and its widgets, and the [built-in commands](Commands-Builtin).
* One definition of "best": `SessionState.best_record()` and `TrainingResult.best_epoch()`
  delegate to `best_epoch`, and `monitored_loss` is what the manager passes to a
  `ReduceLROnPlateau` scheduler and what `status`, the console, the status panel and the
  loss chart show as the best loss.
* `EpochRecord` and `TrainingState` are also exported as `ht.EpochRecord` and `ht.TrainingState`.

## Example

```python
import hollywood_training as ht
from hollywood_training.sync import EpochCompleted, Envelope, LogLevel, LogMessage, best_epoch

records: list[ht.EpochRecord] = []

@process.on_epoch_end                      # process: a ht.TrainingProcess
def report(record: ht.EpochRecord) -> None:
    records.append(record)
    best = best_epoch(records)             # None while no loss is finite
    print(record.epoch, record.monitored_loss, "best:", best.epoch if best else None)

def sink(envelope: Envelope) -> None:      # connected to `channel`, see Sync Channel
    event = envelope.event
    if isinstance(event, EpochCompleted):
        print(f"epoch {event.record.epoch}/{event.total_epochs}, next lr {event.learning_rate:g}")
    elif isinstance(event, LogMessage) and event.level is LogLevel.ERROR:
        print(event.text)
    if envelope.synchronous:
        channel.acknowledge(envelope.sequence)
```

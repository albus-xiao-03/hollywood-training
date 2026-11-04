# Training Process

`hollywood_training/training/process.py` · part of [Training](Training)

`TrainingProcess` wraps a training process written in plain PyTorch into one object:
the model, optimizer, data and step functions. [`run()`](Runner) hands that object to the
[Training Manager](Training-Manager). It exists so that the native workflow stays
unchanged (SPEC 2 and 3). Steps are ordinary functions, registered through constructor
arguments or through decorators that return them unchanged.

## Responsibilities

* Hold the components of one process and the values shown in the GUI.
* Validate them at construction, so mistakes surface before training starts.
* Register the training step, the validation step, epoch hooks and custom terminal commands.
* Resolve which step functions run, falling back to the default supervised steps.

## API

```python
EpochHook = Callable[[EpochRecord], None]

class ProcessError(ValueError): ...

@dataclass(frozen=True, slots=True)
class ProcessSteps:
    train: StepFunction
    validation: StepFunction | None
    epoch_end: tuple[EpochHook, ...]
```

`ProcessError` means a process cannot run as configured; `resolve_steps()` raises it.
`ProcessSteps` holds the functions the manager calls.

```python
class TrainingProcess:
    def __init__(
        self,
        model: nn.Module,
        optimizer: Any,
        train_data: Iterable[Any],
        *,
        epochs: int,
        loss_fn: Callable[..., Any] | None = None,
        val_data: Iterable[Any] | None = None,
        scheduler: Any | None = None,
        train_step: StepFunction | None = None,
        validation_step: StepFunction | None = None,
        on_epoch_end: EpochHook | Sequence[EpochHook] | None = None,
        batch_size: int | None = None,
        normalization: str | None = None,
        name: str | None = None,
    ) -> None
```

| Parameter | Checked at construction | Meaning |
|---|---|---|
| `model` | `nn.Module`, else `TypeError` | the network being trained |
| `optimizer` | has `param_groups`, else `TypeError` | its optimizer |
| `train_data` | iterable, not a one-shot iterator (generator, `iter(...)`), else `TypeError` | training batches, iterated again every epoch (usually a `DataLoader`) |
| `epochs` | `int` ≥ 1 and not `bool`, else `ValueError` | initial total; `epochs <n>` changes it at runtime |
| `loss_fn` | callable, else `TypeError` | required by the default steps; shown in the GUI |
| `val_data` | same as `train_data` | optional validation batches |
| `scheduler` | not checked | stepped after every epoch; `ReduceLROnPlateau` receives the validation loss, or the training loss |
| `train_step` | callable, else `TypeError` | same as `@process.train_step` |
| `validation_step` | callable, else `TypeError` | same as `@process.validation_step` |
| `on_epoch_end` | each hook callable, else `TypeError` | one hook or a sequence of hooks |
| `batch_size` | `int` ≥ 1 and not `bool`, else `ValueError` | shown in the GUI; read from `train_data.batch_size` when omitted |
| `normalization` | not checked | description of the data normalisation, e.g. `"z-score"` |
| `name` | not checked | display name; when `None`, the `@track` name is used, if any |

Read-only properties (no setters): `model`, `optimizer`, `train_data`, `val_data`,
`loss_fn`, `scheduler`, `epochs`, `batch_size`, `normalization`, `name`, and
`commands -> tuple[Command, ...]` (the commands registered with `command()`).

```python
def train_step(self, function: StepFunction) -> StepFunction
def validation_step(self, function: StepFunction) -> StepFunction
def on_epoch_end(self, function: EpochHook) -> EpochHook
```

These are decorators. Each raises `TypeError` if `function` is not callable, registers
it, and returns it unchanged. A second `train_step` or `validation_step` replaces the
first. `on_epoch_end` hooks accumulate and run in registration order, constructor hooks
first.

```python
def command(
    self,
    name: str,
    *,
    summary: str,
    arguments: Sequence[Argument] = (),
    aliases: Sequence[str] = (),
    details: str = "",
) -> Callable[[Handler], Handler]
```

A decorator factory for a custom terminal command `handler(context, *arguments)`. The
handler is wrapped in a `Command` with `category="custom"` and returned unchanged. The
`Command` is built when the decorator is applied, so an invalid name (not one lowercase
word) or a required argument after an optional one raises `ValueError` then. See
[Commands Registry](Commands-Registry).

```python
def resolve_steps(self) -> ProcessSteps
```

Returns the steps to run, or raises `ProcessError` (table below). `repr(process)` gives,
for example, `TrainingProcess(model=SmallNet, epochs=3, validation=yes)`.

## How it works

| Situation | `resolve_steps()` result |
|---|---|
| `train_step` registered | that function |
| no `train_step`, `loss_fn` given | `supervised_train_step(model, optimizer, loss_fn)` |
| no `train_step`, no `loss_fn` | `ProcessError` |
| `val_data` and a `validation_step` | that function |
| `val_data`, no `validation_step`, `loss_fn` given | `supervised_validation_step(model, loss_fn)` |
| `val_data`, no `validation_step`, no `loss_fn` | `ProcessError` |
| a `validation_step` but no `val_data` | `ProcessError` |
| neither | `validation=None`: no validation phase |

`run()` calls `resolve_steps()` before any window opens, so errors surface early.
`TrainingManager.run()` calls it again when the session starts, and each call builds
new default steps. The process only stores and calls the functions, and never inspects
them. In both GUI modes `run()` adds `commands` to the built-in registry
(`create_registry(process.commands)`), where a name or alias that is already taken
raises `ValueError`. With `gui=False` there is no terminal, so the commands go unused.

## Used by / uses

* Uses: [Training Steps](Training-Steps) (`StepFunction`, the default steps),
  [Commands Registry](Commands-Registry) (`Argument`, `Command`, `Handler`) and
  [Sync Events](Sync-Events) (`EpochRecord`).
* Used by: [Runner](Runner), the [Training Manager](Training-Manager), and
  `build_session_info` in [Training Descriptions](Training-Descriptions).

## Example

```python
import torch
import hollywood_training as ht

process = ht.TrainingProcess(model, optimizer, train_loader, val_data=val_loader,
                             loss_fn=torch.nn.CrossEntropyLoss(), epochs=240)

@process.train_step
def train_step(batch):
    inputs, targets = batch
    optimizer.zero_grad()
    logits = model(inputs)
    loss = process.loss_fn(logits, targets)
    loss.backward()
    optimizer.step()
    return {"loss": loss, "accuracy": (logits.argmax(dim=1) == targets).float().mean()}

@process.on_epoch_end
def checkpoint(record):                    # runs on the training thread
    if record.epoch % 50 == 0:
        torch.save(model.state_dict(), f"epoch{record.epoch}.pt")

@process.command("boost", summary="multiply the learning rate",
                 arguments=[ht.Argument("factor", float)])
def boost(context, factor):
    def apply(target):                     # runs on the training thread, between batches
        for group in target.optimizer.param_groups:
            group["lr"] *= factor
        return f"learning rate x{factor:g}"
    return context.submit(apply, f"boost learning rate x{factor:g}")

ht.run(process)    # no validation_step registered: the default supervised one is used
```

In the constructor form the functions are passed directly, e.g.
`ht.TrainingProcess(..., loss_fn=loss_fn, val_data=val_loader, train_step=train_step)`.

## Notes

* Steps and hooks run on the thread that called `run()`. A hook receives the frozen
  `EpochRecord` after the scheduler step, and its return value is ignored. An exception
  in a step or a hook ends the session as `FAILED`.
* `epochs` keeps its initial value; the runtime total lives in
  [`TrainingControl`](Sync-Control).
* Command handlers run on the frontend's thread. They should touch the model only through
  `context.submit(...)`, which runs the action on the training thread at its next safe
  point ([Commands Context](Commands-Context)).

# Training Steps

`hollywood_training/training/steps.py` · part of [Training](Training)

A *step* is a plain function `step(batch)` that holds ordinary PyTorch code for one
batch. This module converts what a step returns into plain floats (`read_step_output`).
It also provides the default supervised steps, which a
[Training Process](Training-Process) uses when it has no custom `train_step` or
`validation_step`, and the small batch helpers that those steps and the manager need.

## Responsibilities

* Define the step contract: return the batch loss (a scalar tensor or a number), or a
  mapping with `"loss"` plus extra metrics.
* Validate step return values and convert them into a `StepResult`.
* Provide the textbook supervised train and validation steps.
* Infer the batch size from a batch, and move nested tensors to a device.

## API

```python
StepFunction = Callable[[Any], Any]

class StepOutputError(TypeError): ...

@dataclass(frozen=True, slots=True)
class StepResult:
    loss: float
    metrics: dict[str, float] = field(default_factory=dict)
```

`StepFunction` is the type of a step. `StepOutputError` means a step returned something
that is not a loss. `StepResult` holds the loss and metrics of one batch as plain floats.

```python
def read_step_output(value: object, *, step_name: str) -> StepResult
```

Normalises a step's return value. `step_name` appears in error messages; the manager
passes `"train_step"` or `"validation_step"`.

| `value` | Result |
|---|---|
| tensor with exactly one element (any shape) | `loss = float(value.detach().item())` |
| `numbers.Real` other than `bool` (`int`, `float`, `np.float32`, ...) | `loss = float(value)` |
| `Mapping` with a `"loss"` key | loss as above; every other entry becomes `metrics[str(key)]`, converted the same way |
| `Mapping` without `"loss"` | `StepOutputError`: `train_step returned a mapping without a 'loss' entry` |
| `None` | `StepOutputError`: `train_step must return the batch loss, but returned None` |
| a tensor with more elements, `bool`, `str`, a numpy array, anything else | `StepOutputError`, e.g. `train_step result must be a scalar, got a tensor of shape (3,)` |

Non-finite values are accepted here; the manager logs a warning for them.

```python
def infer_batch_size(batch: object) -> int | None
```

Returns the leading dimension of the first tensor found. For a tensor that is
`shape[0]` (`None` for a 0-d tensor). For a non-empty list or tuple it looks at the
first item, and for a non-empty mapping at the first value, recursively. Anything else
gives `None`.

```python
def move_to_device(value: Any, device: torch.device) -> Any
```

Moves tensors to `device` with `.to(device)`, including tensors inside mappings
(returned as a plain `dict`), namedtuples (same type), lists and tuples (same type).
Other values come back unchanged.

```python
def split_supervised_batch(batch: Any) -> tuple[Any, Any]
```

Returns `(inputs, targets)` from a list or tuple of exactly two items, which is what a
`DataLoader` over a `TensorDataset` yields. Anything else raises `ValueError`: `the
default steps expect batches of (inputs, targets); register a custom
train_step/validation_step for other batch layouts`.

```python
def supervised_train_step(
    model: nn.Module, optimizer: torch.optim.Optimizer, loss_fn: Callable[..., torch.Tensor]
) -> StepFunction
```

Returns `default_train_step(batch)`. It splits the batch, moves inputs and targets to
`model_device(model)`, then calls `optimizer.zero_grad()`, computes
`loss = loss_fn(model(inputs), targets)`, calls `loss.backward()` and `optimizer.step()`,
and returns `loss`.

```python
def supervised_validation_step(
    model: nn.Module, loss_fn: Callable[..., torch.Tensor]
) -> StepFunction
```

Returns `default_validation_step(batch)`. It splits the batch, moves it to the model's
device and returns `loss_fn(model(inputs), targets)`. It calls neither `eval()` nor
`no_grad()`, because the manager already applies both.

`move_to_device` and `split_supervised_batch` are not re-exported by
`hollywood_training.training`; import them from `hollywood_training.training.steps`.

## How it works

The [Training Manager](Training-Manager) calls the step for every batch, then
`read_step_output(output, step_name=...)`. It adds the loss to the phase's loss sum, and
each metric to a sum and a count per name, so a metric is averaged over the batches that
returned it. While no batch size is known (neither `batch_size=` nor
`train_data.batch_size` gave one), it calls `infer_batch_size(batch)`. The manager never
calls `backward()` or `optimizer.step()` itself, so a custom `train_step` must perform
the whole optimisation step. The default steps look up `model_device(model)`
([Training Descriptions](Training-Descriptions)) on every batch, so they keep working
after the model is moved with `.to(...)`.

## Used by / uses

* Uses: `model_device` from [Training Descriptions](Training-Descriptions).
* Used by: [Training Process](Training-Process), where `resolve_steps()` builds the
  default steps, and the [Training Manager](Training-Manager) (`read_step_output`,
  `infer_batch_size`).

## Example

```python
import torch
from hollywood_training.training import read_step_output, supervised_train_step

read_step_output(torch.tensor(0.5), step_name="train_step")
# StepResult(loss=0.5, metrics={})
read_step_output({"loss": torch.tensor(1.0), "acc": 0.75}, step_name="train_step")
# StepResult(loss=1.0, metrics={'acc': 0.75})

step = supervised_train_step(model, optimizer, torch.nn.CrossEntropyLoss())
loss = step((inputs, targets))      # one optimisation step; returns the loss tensor
```

A custom step that also returns a metric:

```python
@process.train_step
def train_step(batch):
    inputs, targets = batch
    optimizer.zero_grad()
    logits = model(inputs)
    loss = loss_fn(logits, targets)
    loss.backward()
    optimizer.step()
    return {"loss": loss, "accuracy": (logits.argmax(dim=1) == targets).float().mean()}
```

## Notes

* Tensor values are read with `.item()`. For a CUDA tensor this waits for the device,
  once per returned value per batch.
* Metric names are free-form. [Formatting](Formatting) shows names that contain `acc`,
  `precision`, `recall` or `f1` as percentages when the value is between 0 and 1.
* A metric returned by only some batches is averaged over those batches only. See
  [Training Manager](Training-Manager).

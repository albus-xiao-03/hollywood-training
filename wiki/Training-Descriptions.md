# Training Descriptions

`hollywood_training/training/describe.py` · part of [Training](Training)

This module builds human-readable descriptions of the training components. It reads
only public attributes (`type(...).__name__`, `optimizer.defaults`,
`optimizer.param_groups`, `DataLoader.batch_size`) or values given explicitly to
`TrainingProcess`. The results fill the `SessionInfo` that the
[status panel](GUI-Status-Panel) and the [console header](Console-Frontend) show. That
is how the GUI names the loss, optimizer, device and normalisation without inspecting
user code (SPEC 8).

## Responsibilities

* Describe the loss function, optimizer, scheduler, device and normalisation as short strings.
* Read the current learning rate, the batch size and the number of batches.
* Find the device of a model.
* Assemble the `SessionInfo` sent with `SessionStarted`.

## API

```python
def model_device(model: nn.Module) -> torch.device
```
Device of the first initialised parameter, or buffer if no parameter is initialised.
Placeholders of lazy modules are skipped. Returns CPU when nothing is found.

```python
def describe_device(device: torch.device) -> str
```
* CUDA, when available: `cuda:<index> (<device name>)`. A device without an index uses
  the current CUDA device.
* CPU: `cpu (<n> threads)`, where `n` is `torch.get_num_threads()` (`thread` when `n`
  is 1).
* Anything else: `str(device)`, e.g. `mps`.

```python
def describe_loss(loss_fn: Any) -> str
```
* `None` gives `defined in step`.
* An `nn.Module` gives its class name plus a string `reduction` attribute when it has
  one, e.g. `MSELoss (sum)`.
* Any other object gives its `__name__` (`l1_loss`), or its type name if there is none.

```python
def describe_optimizer(optimizer: Any) -> tuple[str, str]
```
Returns `(type name, details)`, e.g. `("Adam", "betas=0.9/0.999")`. The details come
from `optimizer.defaults`, in this order: `momentum`, `betas`, `alpha`, `rho`,
`weight_decay` (shown as `wd`), `dampening`, `nesterov`, `amsgrad`.

* `None`, `False` and zero values are left out.
* `True` shows the bare label.
* Tuples are joined with `/`.
* Numbers use the `:g` format.
* With more than one parameter group, `<n> param groups` is appended.

The learning rate is not part of the details.

```python
def describe_scheduler(scheduler: Any) -> str | None
```
`None` for no scheduler, otherwise the type name, e.g. `StepLR`.

```python
def describe_normalization(architecture: ArchitectureInfo, data_normalization: str | None) -> str
```
Returns `data: <declared> · layers: <found>`. The data part appears only when a
non-empty description was given. The layer part is
`architecture.normalization_summary()`, e.g. `BatchNorm1d ×2, LayerNorm`, or `none`
(see [Tracking Architecture Model](Tracking-Architecture-Model)).

```python
def current_learning_rate(optimizer: Any) -> float
```
Returns the `"lr"` of the first parameter group as a `float`, or NaN when there are no
`param_groups` or no `"lr"`.

```python
def count_batches(data: Iterable[Any] | None) -> int | None
def resolve_batch_size(explicit: int | None, data: Iterable[Any]) -> int | None
```
`count_batches` returns `len(data)` when the source supports it (a `DataLoader`
does), else `None`; it also returns `None` for `data=None`. `resolve_batch_size`
returns the explicit value if given, else `data.batch_size` if that is an `int`, else
`None`. A `DataLoader` built with a `batch_sampler` has `batch_size=None`.

```python
def build_session_info(
    process: TrainingProcess,
    architecture: ArchitectureInfo,
    *,
    total_epochs: int,
    neuron_limit: int,
) -> SessionInfo
```
Builds the static description of a session (see the table below).

`hollywood_training.training` re-exports `model_device`, `describe_loss`,
`describe_optimizer`, `describe_normalization` and `current_learning_rate`. The other
functions are available from `hollywood_training.training.describe`.

## How it works

`build_session_info` fills every `SessionInfo` field ([Sync Events](Sync-Events)):

| Field | Source |
|---|---|
| `network_class` | `architecture.class_name` |
| `display_name` | `process.name`, or else `architecture.display_name` (the `@track` name) |
| `architecture` | the `ArchitectureInfo` passed in |
| `total_epochs`, `neuron_limit` | the keyword arguments; the manager passes `control.total_epochs` and `tracker.max_neurons` |
| `batch_size` | `resolve_batch_size(process.batch_size, process.train_data)` |
| `train_batches`, `val_batches` | `count_batches(process.train_data)`, `count_batches(process.val_data)` |
| `has_validation` | `process.val_data is not None` |
| `loss_function` | `describe_loss(process.loss_fn)` |
| `optimizer`, `optimizer_details` | `describe_optimizer(process.optimizer)` |
| `normalization` | `describe_normalization(architecture, process.normalization)` |
| `scheduler` | `describe_scheduler(process.scheduler)` |
| `device` | `describe_device(model_device(process.model))` |
| `learning_rate` | `current_learning_rate(process.optimizer)` |

## Used by / uses

* Uses: `ArchitectureInfo` ([Tracking Architecture Model](Tracking-Architecture-Model)),
  `is_uninitialized` ([Tracking Layer Classification](Tracking-Layer-Classification)),
  and `SessionInfo` ([Sync Events](Sync-Events)), which is imported inside
  `build_session_info`.
* Used by: the [Training Manager](Training-Manager) (`build_session_info`,
  `count_batches`, `current_learning_rate`, `resolve_batch_size`) and the default steps
  in [Training Steps](Training-Steps) (`model_device`).

## Example

```python
import torch
from torch import nn
from hollywood_training.training import (
    current_learning_rate, describe_loss, describe_optimizer,
)

model = nn.Sequential(nn.Linear(2, 4), nn.BatchNorm1d(4), nn.ReLU(), nn.Linear(4, 1))
sgd = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, nesterov=True)
describe_optimizer(sgd)                                     # ('SGD', 'momentum=0.9, nesterov')
describe_optimizer(torch.optim.Adam(model.parameters(), weight_decay=1e-4))
                                                            # ('Adam', 'betas=0.9/0.999, wd=0.0001')
describe_loss(nn.MSELoss(reduction="sum"))                  # 'MSELoss (sum)'
describe_loss(nn.functional.l1_loss)                        # 'l1_loss'
current_learning_rate(sgd)                                  # 0.1
```

## Notes

* `SessionInfo` is built once, when the session starts. The CPU thread count in
  `device` is read at that moment, so it reflects the cap that [`run()`](Runner)
  applies in GUI modes, e.g. `cpu (5 threads)`.
* Only the first parameter group's learning rate is shown and recorded.

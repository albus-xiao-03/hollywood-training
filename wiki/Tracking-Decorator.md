# Tracking Decorator

`hollywood_training/tracking/decorator.py` · part of [Tracking](Tracking)

`track` declares a network for tracking. It works on an `nn.Module` class, as a decorator
with or without options, and on an instance, as a plain function, for models you did not
define yourself (e.g. `nn.Sequential`). The declaration is kept in registries held by this
module, and the target is returned unchanged. `__init__`, `forward` and pickling
therefore behave exactly as in plain PyTorch.

## Responsibilities

* Validate and store display options (`TrackOptions`) for a class or an instance.
* Return the decorated class or instance itself: no wrapper, no subclass, no added
  attributes or hooks.
* Look the options up for a model: instance first, then its class hierarchy. Subclasses
  inherit the declaration of a tracked base class.

## API

```python
@dataclass(frozen=True, slots=True)
class TrackOptions:
    name: str | None = None
    input_labels: tuple[str, ...] | None = None
    output_labels: tuple[str, ...] | None = None
```

Display options declared with `track`. `name` is shown next to the network's class name.
`input_labels` name the input features (first column of neurons), and `output_labels` the
outputs (last column).

```python
def track(
    target: T | None = None,
    /,
    *,
    name: str | None = None,
    input_labels: Sequence[str] | None = None,
    output_labels: Sequence[str] | None = None,
) -> T | Callable[[T], T]
```

* `target` (positional-only): an `nn.Module` subclass or instance. Omit it to pass
  options.
* `name`, `input_labels`, `output_labels` (keyword-only): become the `TrackOptions`. Each
  label is converted with `str(value)` and stored as a tuple.
* **Returns** the target itself, or, without a target, a decorator that registers its
  argument and returns it.
* **Raises** `TypeError` when a label argument is a single `str`, which is checked at
  once. It also raises `TypeError` for a class that is not an `nn.Module` subclass
  (`@track expects a torch.nn.Module subclass, got ...`), and for anything that is neither
  a class nor an instance (`track() expects a torch.nn.Module subclass or instance, got
  ...`). With the options form, these two errors come from the returned decorator.

| Call | Effect |
|---|---|
| `@track` on a class | registers the class with default options |
| `@track(name=..., ...)` or `@track()` | returns a decorator that registers its argument (class or instance) with these options |
| `track(instance)` | registers this instance only. Other instances of its class are unaffected |
| `track(instance, name=..., ...)` | registers this instance only, with these options |

The overloads that type checkers see (`T = TypeVar("T")`):

```python
@overload
def track(
    target: T,
    /,
    *,
    name: str | None = ...,
    input_labels: Sequence[str] | None = ...,
    output_labels: Sequence[str] | None = ...,
) -> T: ...
@overload
def track(
    target: None = None,
    /,
    *,
    name: str | None = ...,
    input_labels: Sequence[str] | None = ...,
    output_labels: Sequence[str] | None = ...,
) -> Callable[[T], T]: ...
```

```python
def track_options(model: nn.Module) -> TrackOptions | None
def is_tracked(model: nn.Module) -> bool
```

`track_options` returns the instance's own entry, or else the entry of the first class in
`type(model).__mro__` that has one, or `None`. `is_tracked` is
`track_options(model) is not None`.

## How it works

* There are two module-level `weakref.WeakKeyDictionary` registries, one keyed by class
  and one keyed by module instance. An entry disappears when its class or instance is
  garbage-collected, so tracking never keeps a model alive.
* `track` builds the `TrackOptions` first. Then it either registers the target or returns
  `lambda obj: _register(obj, options)`.
* `_register` checks `isinstance(target, type)` first (class path, which requires an
  `nn.Module` subclass), then `isinstance(target, nn.Module)` (instance path).
* Calling `track` again on the same target replaces its options. Decorating a subclass
  overrides what it inherits, because the MRO walk stops at the nearest tracked class.

## Used by / uses

* Uses `torch.nn.Module` (type checks only) and `weakref`.
* [Tracking Network Tracker](Tracking-Network-Tracker): `NetworkTracker` defaults its
  options to `track_options(model)`, then `track_options(unwrap(model))`, then
  `TrackOptions()`. Untracked models therefore work too, with no name or labels.
* [Tracking Inspector](Tracking-Inspector): `build_architecture` copies `name` into
  `display_name` and the labels into `input_labels` / `output_labels`.
* [Training Descriptions](Training-Descriptions): `SessionInfo.display_name` is
  `TrainingProcess(name=...)` when given, otherwise the `@track` name.
* The top-level package exposes `ht.track` and `ht.is_tracked` (lazy exports).

## Example

```python
import hollywood_training as ht
from torch import nn
from hollywood_training.tracking import track_options

@ht.track(name="Spiral Classifier", input_labels=("x", "y"), output_labels=("arm A", "arm B", "arm C"))
class SpiralNet(nn.Module):
    def __init__(self, hidden: int = 16) -> None:
        super().__init__()
        self.fc1 = nn.Linear(2, hidden)
        self.act = nn.Tanh()
        self.fc2 = nn.Linear(hidden, 3)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))

class WideSpiralNet(SpiralNet):                 # inherits the declaration
    pass

track_options(WideSpiralNet(64)).name          # 'Spiral Classifier'

model = ht.track(nn.Sequential(nn.Linear(1, 24), nn.SiLU(), nn.Linear(24, 1)))
ht.is_tracked(model)                           # True
ht.is_tracked(nn.Sequential())                 # False: other instances are unaffected
```

## Notes

* `track_options` and `is_tracked` expect an **instance** and use it exactly as passed.
  `is_tracked(SpiralNet)`, the class, is `False`, because the lookup walks
  `type(SpiralNet).__mro__`, which is `(type, object)`.
* Wrappers are not looked through: `is_tracked(nn.DataParallel(net))` and
  `is_tracked(torch.compile(net))` are `False`.
  `NetworkTracker` unwraps by itself (see
  [Tracking Network Tracker](Tracking-Network-Tracker)).

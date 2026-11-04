# Formatting

`hollywood_training/formatting.py` · part of [Architecture](Architecture)

Text formatting shared by the terminal commands, the console frontend and the GUI. Every
frontend shows durations, losses, learning rates and metrics with the same rules, and
shows missing values as the same placeholder. The module imports only `math` and
`typing` (`EpochRecord` is imported for type checking only), so the window process can
use it without torch.

## Responsibilities

* Format durations, losses, learning rates, metrics, percentages, data sizes and counts.
* Show missing values (`None` or NaN) as `PLACEHOLDER`.
* Produce the one-line epoch summary used by the console and the GUI terminal.

## API

```python
PLACEHOLDER = "—"

def duration(seconds: float | None, *, millis: bool = True) -> str
def loss(value: float | None) -> str
def learning_rate(value: float | None) -> str
def metric(name: str, value: float) -> str
def percent(fraction: float | None, digits: int = 1) -> str
def data_size(num_bytes: float | None) -> str
def count(value: int | None) -> str
def epoch_summary(record: EpochRecord, total_epochs: int) -> str
```

| Function | Rule | Examples |
|---|---|---|
| `duration` | `HH:MM:SS.mmm`, or `HH:MM:SS` with `millis=False`. Hours keep counting past 99; milliseconds are truncated. Missing or negative values give `—` | `duration(3723.5)` → `01:02:03.500`; `duration(3723.5, millis=False)` → `01:02:03`; `duration(90000)` → `25:00:00.000`; `duration(-1)` → `—` |
| `loss` | `.4f` for 0 and for 1e-3 ≤ \|v\| < 1e4, otherwise `.3e`. Infinities give `inf` / `-inf` | `loss(0.123456)` → `0.1235`; `loss(1.5e-5)` → `1.500e-05`; `loss(12345.6)` → `1.235e+04` |
| `learning_rate` | `.4g` | `learning_rate(0.000123456)` → `0.0001235`; `learning_rate(1e-5)` → `1e-05`; NaN → `—` |
| `metric` | a percentage with one decimal when the name contains `acc`, `precision`, `recall` or `f1` (case-insensitive) and 0 ≤ value ≤ 1; otherwise `loss(value)` | `metric("accuracy", 0.931)` → `93.1%`; `metric("accuracy", 1.2)` → `1.2000`; `metric("mae", 0.5)` → `0.5000` |
| `percent` | `fraction * 100` with `digits` decimals, then `%` | `percent(0.5)` → `50.0%`; `percent(0.12345, 2)` → `12.35%` |
| `data_size` | binary units `B`, `KiB`, `MiB`, `GiB`, `TiB`. Bytes have no decimals, larger units one; above that the value stays in `TiB` | `data_size(512)` → `512 B`; `data_size(1536)` → `1.5 KiB`; `data_size(3 * 1024**3)` → `3.0 GiB` |
| `count` | thousands separators; only `None` gives `—` | `count(1234567)` → `1,234,567` |

`epoch_summary(record, total_epochs)` joins these parts with ` | `:

1. `epoch <n>/<total>`, with `n` right-aligned to the width of `total`
2. `train <loss>`
3. `val <loss>`, only when `record.val_loss` is not `None`
4. one `<name> <metric>` per training metric
5. one `val <name> <metric>` per validation metric
6. `lr <learning_rate>`
7. the epoch duration as `<seconds:.3f>s`

```
epoch  12/200 | train 0.4123 | val 0.4555 | accuracy 91.0% | val accuracy 88.5% | lr 0.001 | 0.235s
```

## How it works

A private helper `_missing(value)` is true for `None` and for a `float` NaN. `duration`,
`loss`, `learning_rate`, `percent` and `data_size` return `PLACEHOLDER` for missing
values. `metric` passes a NaN on to `loss`, because NaN fails the 0-to-1 range check,
and `loss` returns the placeholder. `count` checks only for `None`.

## Used by / uses

| Module | Functions used |
|---|---|
| [Console Frontend](Console-Frontend) | `count`, `duration`, `epoch_summary`, `learning_rate`, `loss` |
| [Commands Builtin](Commands-Builtin) | `PLACEHOLDER`, `count`, `duration`, `learning_rate`, `loss`, `metric`, `percent` |
| [GUI Main Window](GUI-Main-Window) | `count`, `duration`, `epoch_summary` |
| [GUI Status Panel](GUI-Status-Panel) | `PLACEHOLDER`, `count`, `duration`, `learning_rate`, `loss`, `metric`, `percent` |
| [GUI Header](GUI-Header), [GUI Network View](GUI-Network-View) | `count` |
| [GUI Loss Chart](GUI-Loss-Chart) | `loss` |
| [GUI Usage Panel](GUI-Usage-Panel) | `data_size` |

Uses: `math`, plus `EpochRecord` ([Sync Events](Sync-Events)) for type hints only.

## Example

```python
from hollywood_training import formatting as fmt

fmt.duration(3723.5)            # '01:02:03.500'
fmt.loss(0.123456)              # '0.1235'
fmt.metric("accuracy", 0.931)   # '93.1%'
fmt.data_size(3 * 1024**3)      # '3.0 GiB'
fmt.learning_rate(float("nan")) # '—'
```

## Notes

* `PLACEHOLDER` is an em dash (U+2014), not a hyphen.
* `duration` truncates instead of rounding, and binary floating point can drop a
  millisecond: `duration(1.001)` gives `00:00:01.000`.
* `data_size` rounds at unit boundaries: `data_size(1023.6)` gives `1024 B`.

# Usage Panel

`hollywood_training/gui/widgets/usage_panel.py` · part of [GUI](GUI)

`UsagePanel` shows the current CPU and memory usage of the machine and of the training
process. When the model trains on a GPU that can be read (CUDA or Apple MPS), it also
shows GPU utilisation and memory (SPEC GUI 4). Readings are `UsageSample`s. They are
either pushed in from elsewhere (`set_sample`, used by the window process) or pulled
twice a second from a sampler function (`set_sampler`, used in-process). The bars ease
toward each new reading, and a sparkline shows the last minute of CPU usage.

## Responsibilities

* Display the latest `UsageSample` as labelled bars with values and details.
* Show GPU rows only when the reading contains a GPU part.
* Smooth the displayed levels, and keep one minute of CPU history.

## API

```python
SAMPLE_INTERVAL_MS = 500
HISTORY = 120  # one minute of CPU readings
```

```python
class UsagePanel(HudPanel):
    def __init__(self, clock: AnimationClock, parent: QWidget | None = None) -> None

    @property
    def sample(self) -> UsageSample | None
    def set_sampler(self, sampler: Callable[[], UsageSample] | None) -> None
    def set_sample(self, sample: UsageSample) -> None
    def stop(self) -> None
    def sample_now(self) -> None
```

| Member | Behaviour |
|---|---|
| `sample` | the latest reading, or `None` before the first one |
| `set_sampler(sampler)` | takes a reading from `sampler` at once, then every `SAMPLE_INTERVAL_MS` ms. `None` stops pulling |
| `set_sample(sample)` | shows a reading taken elsewhere, and appends its CPU percentage to the history |
| `stop()` | stops the pull timer. `MainWindow.closeEvent` calls it |
| `sample_now()` | calls the sampler once. An exception from the sampler is ignored |

The title is "System usage", the code `SYS-04` and the minimum size 320 × 140.

## How it works

| Row | Shown when | Bar | Value | Detail line |
|---|---|---|---|---|
| `CPU` | always | `cpu_percent / 100` | `cpu_percent`, such as `32.1%` | `proc <process_cpu_percent>% · <cpu_count> cores` |
| `MEM` | always | `memory_fraction` | `<used> / <total>` in B, KiB, MiB, GiB or TiB | `proc <process_memory>` |
| `GPU` | `gpu.utilization` is not `None` | `utilization / 100` | `utilization` in % | the GPU's name |
| `VRAM` | `gpu.memory_used` is not `None` | used / total, striped when the total is unknown | `<used> / <total>`, with `?` for an unknown total | the GPU's name |

* On every clock tick the bars move toward their targets with
  `approach(..., half_life=0.12)` ([Animation](GUI-Animation)). Each row's bar highlight
  is slightly offset in time.
* The sparkline plots the last `HISTORY` CPU readings, one minute at two readings per
  second. The newest reading is at the right edge, the area under the line is filled,
  and the label is `CPU 60s`.
* Before the first reading the panel shows `SENSORS OFFLINE`.
* The `proc` values describe the process that took the sample. In both frontends this is
  the training process, so `proc 275%` means it is using 2.75 cores.

**Who feeds it:**

| Frontend | How |
|---|---|
| [Remote Frontend](GUI-Remote-Frontend) | samples `SystemMonitor` every 0.5 s on a thread of the training process and sends `("usage", sample)`. The [Window Process](GUI-Window-Process) calls `set_sample` |
| [In-Process Frontend](GUI-In-Process-Frontend) | calls `set_sampler(SystemMonitor(info.device).sample)` on `SessionStarted` |

## Used by / uses

* Used by the [Main Window](GUI-Main-Window) as `window.usage`.
* Uses [Usage Data](Monitor-Usage-Data) (`UsageSample`, `GpuUsage`),
  [HUD Panel](GUI-HUD-Panel) (`paint_bar`), [Theme](GUI-Theme),
  [Animation](GUI-Animation) and [Formatting](Formatting) (`data_size`). The readings
  themselves come from the [System Monitor](Monitor-System) and the
  [GPU Probes](Monitor-GPU).

## Example

```python
import time

from PySide6.QtWidgets import QApplication

from hollywood_training.gui import AnimationClock
from hollywood_training.gui.widgets import UsagePanel
from hollywood_training.monitor import UsageSample

app = QApplication([])
clock = AnimationClock()
panel = UsagePanel(clock)
panel.set_sample(UsageSample(
    timestamp=time.monotonic(), cpu_percent=32.1, process_cpu_percent=275.0, cpu_count=10,
    memory_used=1_200_000_000, memory_total=12_600_000_000, process_memory=590_000_000,
))                                     # no gpu: only the CPU and MEM rows
panel.show()
clock.start()                          # bars ease toward the reading
```

## Notes

* Usage is not synchronised with epochs. The panel is not part of `render_now()`, and
  readings are applied whenever they arrive.
* In the in-process frontend the pull timer fires only while the training loop pumps Qt,
  so readings may be delayed by a long batch.

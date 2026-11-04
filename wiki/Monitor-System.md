# System Monitor

`hollywood_training/monitor/system.py` · part of [Monitoring](Monitoring)

`SystemMonitor` samples CPU, memory and, when applicable, GPU usage of the machine and of
the current process. It uses psutil, so it works the same on Linux, macOS and Windows.
Sampling is cheap and non-blocking; the GUI frontends call `sample()` twice a second.

## Responsibilities

* Measure machine-wide and per-process CPU and memory.
* Attach a GPU reading when the training device is a GPU that can be read.

## API

```python
class SystemMonitor:
    def __init__(self, device: str | torch.device | None = None) -> None: ...
    @property
    def has_gpu(self) -> bool: ...
    def sample(self) -> UsageSample: ...
```

* `device`: the training device, as a `torch.device`, a device string such as
  `"cuda:0"`, or a `SessionInfo.device` description such as `"cuda:0 (<GPU name>)"`.
  It is passed to [`create_gpu_probe`](Monitor-GPU); `None` means no GPU probe.
* `has_gpu`: `True` when a GPU probe was created.
* `sample()`: returns one [`UsageSample`](Monitor-Usage-Data).

## How it works

The constructor creates `psutil.Process(os.getpid())`, stores
`psutil.cpu_count(logical=True) or 1` and creates the GPU probe. It also calls
`psutil.cpu_percent(None)` and `process.cpu_percent(None)` once. Both measure since their
previous call, so this primes them, and the first `sample()` reports usage since
construction.

`sample()` fills the fields as follows:

| Field | Value |
|---|---|
| `timestamp` | `time.monotonic()` |
| `cpu_percent` | `psutil.cpu_percent(None)` |
| `process_cpu_percent` | `process.cpu_percent(None)`; `0.0` on `psutil.Error` |
| `cpu_count` | the count stored at construction |
| `memory_used` | `total - available` of one `psutil.virtual_memory()` reading |
| `memory_total` | `total` of the same reading |
| `process_memory` | `process.memory_info().rss`; `0` on `psutil.Error` |
| `gpu` | `probe.sample()` (may be `None`), or `None` without a probe |

## Used by / uses

* Uses psutil, torch (for the `device` type), [GPU Probes](Monitor-GPU) and
  [Usage Data](Monitor-Usage-Data).
* [GUI Remote Frontend](GUI-Remote-Frontend): on the first `SessionStarted` it creates a
  monitor for `info.device` and starts a daemon usage thread that sends
  `("usage", monitor.sample())` every 0.5 s, until shutdown or a failed send.
* [GUI In-Process Frontend](GUI-In-Process-Frontend): on `SessionStarted` it creates a
  monitor and passes `monitor.sample` to `UsagePanel.set_sampler`; the
  [usage panel](GUI-Usage-Panel) calls it every 500 ms and ignores exceptions.
* Exported lazily as `hollywood_training.monitor.SystemMonitor`.

## Example

```python
import time
from hollywood_training.monitor import SystemMonitor

monitor = SystemMonitor("cpu (8 threads)")    # the text before the first space is parsed
for _ in range(3):
    time.sleep(0.5)                           # CPU usage is measured between calls
    sample = monitor.sample()
    print(f"{sample.cpu_percent:5.1f}%  process {sample.process_cpu_percent:.0f}%  "
          f"memory {sample.memory_fraction:.0%}")
```

## Notes

* CPU percentages cover the time since the previous `sample()`; sample at a steady
  interval for meaningful values.
* Only the per-process readings are guarded: a `psutil.Error` there yields zeros, while
  errors from the machine-wide calls propagate.
* The measured process is the one that created the monitor: the training process in both
  GUI modes.

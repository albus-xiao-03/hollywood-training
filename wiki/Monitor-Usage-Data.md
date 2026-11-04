# Usage Data

`hollywood_training/monitor/usage.py` · part of [Monitoring](Monitoring)

Usage readings as plain data: what frontends receive and draw. `UsageSample` is one
reading of the machine and of the sampling process; `GpuUsage` is one reading of a GPU.
Both are frozen slot dataclasses, and the module imports neither torch nor psutil, so
the window process can unpickle and draw them.

## Responsibilities

* Define the reading types shared by the sampler, the link and the usage panel.
* Derive the machine memory fraction.

## API

```python
@dataclass(frozen=True, slots=True)
class GpuUsage:
    backend: str  # "cuda" or "mps"
    name: str
    utilization: float | None  # percent, when the driver exposes it
    memory_used: int | None  # bytes
    memory_total: int | None  # bytes
```

| Field | Meaning |
|---|---|
| `backend` | `"cuda"` ([`CudaProbe`](Monitor-GPU)) or `"mps"` ([`MpsProbe`](Monitor-GPU)) |
| `name` | device name, e.g. from `torch.cuda.get_device_name`; `"Apple GPU (MPS)"` for MPS |
| `utilization` | percent; `None` when not exposed (always for MPS, for CUDA without NVML) |
| `memory_used` | bytes in use on the device |
| `memory_total` | bytes available on the device; `None` when unknown |

```python
@dataclass(frozen=True, slots=True)
class UsageSample:
    timestamp: float
    cpu_percent: float  # whole machine, 0-100
    process_cpu_percent: float  # this process, 100 = one full core
    cpu_count: int
    memory_used: int  # bytes, whole machine
    memory_total: int
    process_memory: int  # resident set size of this process, bytes
    gpu: GpuUsage | None = None

    @property
    def memory_fraction(self) -> float: ...
```

| Field | Meaning |
|---|---|
| `timestamp` | `time.monotonic()` in the sampling process |
| `cpu_percent` | whole machine, 0-100 |
| `process_cpu_percent` | the sampling process; 100 = one full core, so it can exceed 100 |
| `cpu_count` | logical CPUs |
| `memory_used`, `memory_total` | whole machine, bytes |
| `process_memory` | resident set size of the sampling process, bytes |
| `gpu` | a `GpuUsage` when the training device has a probe, else `None` |

`memory_fraction` is `memory_used / memory_total`, or `0.0` when `memory_total` is 0.

"The sampling process" is the training process in both GUI modes
([Monitoring](Monitoring)).

## How it works

The module only declares data. [`SystemMonitor.sample()`](Monitor-System) fills a
`UsageSample`, and a [GPU probe](Monitor-GPU) supplies its `gpu`.

## Used by / uses

* Standard library only (`dataclasses`).
* Produced by [System Monitor](Monitor-System) and [GPU Probes](Monitor-GPU).
* Sent as `("usage", UsageSample)` over the [GUI Link Protocol](GUI-Link-Protocol).
* Drawn by the [GUI Usage Panel](GUI-Usage-Panel): bars for `cpu_percent` and
  `memory_fraction`, details with `process_cpu_percent`, `cpu_count` and `process_memory`,
  a GPU row when `utilization` is not `None`, a VRAM row when `memory_used` is not `None`
  (with a bar only when `memory_total` is known), and a CPU sparkline of the last 120
  readings.

## Example

Building a sample by hand, for instance to test a view:

```python
import time
from hollywood_training.monitor import GpuUsage, UsageSample

gpu = GpuUsage("cuda", "test GPU", utilization=87.0,
               memory_used=6 * 1024**3, memory_total=24 * 1024**3)
sample = UsageSample(time.monotonic(), cpu_percent=35.0, process_cpu_percent=180.0,
                     cpu_count=16, memory_used=8 * 1024**3, memory_total=32 * 1024**3,
                     process_memory=2 * 1024**3, gpu=gpu)
sample.memory_fraction        # 0.25
```

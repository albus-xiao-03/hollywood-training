# GPU Probes

`hollywood_training/monitor/gpu.py` · part of [Monitoring](Monitoring)

GPU usage probes for the device a model trains on. A probe reads one GPU's memory, and
its utilisation where the driver exposes it, and returns a
[`GpuUsage`](Monitor-Usage-Data). `create_gpu_probe` picks the probe for a device, or
none for a CPU, and the GUI then hides its GPU section.

## Responsibilities

* Read CUDA memory and utilisation (NVIDIA, and AMD ROCm builds of PyTorch).
* Read Apple MPS memory.
* Choose the probe that matches the training device.

## API

```python
class GpuProbe(Protocol):
    def sample(self) -> GpuUsage | None: ...
```

The interface [`SystemMonitor`](Monitor-System) relies on.

```python
class CudaProbe:
    def __init__(self, index: int) -> None: ...
    def sample(self) -> GpuUsage | None: ...
```

Reads CUDA device `index`. The constructor stores `torch.cuda.get_device_name(index)`.
`sample()` calls `torch.cuda.mem_get_info(index)` for `(free, total)` and returns `None`
if that raises `RuntimeError`. Utilisation comes from `float(torch.cuda.utilization(index))`;
if that call raises any exception (NVML missing or unsupported), utilisation is `None`
and the probe stops asking. Returns `GpuUsage("cuda", name, utilization, total - free, total)`.

```python
class MpsProbe:
    def sample(self) -> GpuUsage | None: ...
```

Reads Apple's Metal Performance Shaders backend. Returns `None` when `torch.mps` is
missing or `torch.mps.driver_allocated_memory()` raises `AttributeError` or
`RuntimeError`. The total is `torch.mps.recommended_max_memory()` when that function
exists and does not raise `RuntimeError`, else `None`. Returns
`GpuUsage("mps", "Apple GPU (MPS)", None, used, total)`.

```python
def create_gpu_probe(device: str | torch.device | None) -> GpuProbe | None: ...
```

A probe for `device` when it is a GPU, else `None`.

## How it works

`create_gpu_probe` decides:

| Step | Result |
|---|---|
| `device is None` | `None` |
| parse `torch.device(str(device).split(" ", 1)[0])`; `RuntimeError` or `TypeError` | `None` |
| type `cuda` and `torch.cuda.is_available()` | `CudaProbe(device.index)`, or `CudaProbe(torch.cuda.current_device())` without an index |
| type `mps`, `torch.backends.mps` present and `is_available()` | `MpsProbe()` |
| anything else | `None` |

Only the text before the first space is parsed, so the `SessionInfo.device` descriptions
the frontends pass (`"cuda:0 (<GPU name>)"`, `"cpu (8 threads)"`) work as well as
`"cuda:1"` or a `torch.device`.

| Backend | `utilization` | `memory_used` | `memory_total` |
|---|---|---|---|
| CUDA | percent with NVML, else `None` | `total - free` from `mem_get_info` | `total` from `mem_get_info` |
| MPS | always `None` | `driver_allocated_memory()` | `recommended_max_memory()` or `None` |

NVML utilisation needs `nvidia-ml-py`, which the `gpu` extra installs:
`pip install ".[gpu]"`. Memory readings work without it.

## Used by / uses

* Uses torch and [Usage Data](Monitor-Usage-Data) (`GpuUsage`).
* [System Monitor](Monitor-System): `SystemMonitor.__init__` calls `create_gpu_probe`,
  and `sample()` calls the probe's `sample()`.
* Exported lazily from `hollywood_training.monitor`: `GpuProbe`, `CudaProbe`, `MpsProbe`,
  `create_gpu_probe`.

## Example

```python
import torch
from hollywood_training.monitor import create_gpu_probe

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
probe = create_gpu_probe(device)
if probe is None:
    print("no GPU readings for", device)
else:
    reading = probe.sample()                  # GpuUsage, or None if the read failed
    if reading is not None:
        print(reading.name, reading.memory_used, reading.memory_total, reading.utilization)
```

## Notes

* A `CudaProbe` gives up on utilisation after the first failure and reports `None` from
  then on; memory is read on every call.
* `CudaProbe.__init__` calls `torch.cuda.get_device_name` without a guard.
* `create_gpu_probe` returns `None` for `cuda` devices when CUDA is not available, and for
  every other device type.

# Monitoring

`hollywood_training/monitor/` · layer of the [Architecture](Architecture)

The `monitor` package measures CPU and memory usage of the machine and of the training
process, plus GPU usage when the model trains on a GPU. Readings are plain data
(`UsageSample`, `GpuUsage`) that the [usage panel](GUI-Usage-Panel) draws. The data
module needs neither torch nor psutil. The sampler and the GPU probes need both and are
loaded lazily.

## What is measured

| Reading | Source | `UsageSample` field |
|---|---|---|
| machine CPU, 0-100 % | `psutil.cpu_percent(None)` | `cpu_percent` |
| process CPU, 100 = one full core | `psutil.Process.cpu_percent(None)` | `process_cpu_percent` |
| logical CPUs | `psutil.cpu_count(logical=True)` | `cpu_count` |
| machine memory | `psutil.virtual_memory()`: `total - available`, `total` | `memory_used`, `memory_total` |
| process memory | resident set size, `Process.memory_info().rss` | `process_memory` |
| GPU memory and utilisation | a [GPU probe](Monitor-GPU) for the training device | `gpu` |

## Where sampling happens

The sampler always runs in the training process, where the training device is known.

| Frontend | `SystemMonitor` created | Sampled | Reaches the panel |
|---|---|---|---|
| window process (default) | on the first `SessionStarted`, for `info.device` | every 0.5 s on the training process's usage thread | `("usage", sample)` over the [link](GUI-Link-Protocol), then `UsagePanel.set_sample` |
| in-process window | on `SessionStarted`, for `info.device` | every 500 ms by the panel's timer (`UsagePanel.set_sampler(monitor.sample)`) | directly |
| console | not used | | |

## Modules

| Module | Page | Contents |
|---|---|---|
| `usage.py` | [Usage Data](Monitor-Usage-Data) | `UsageSample`, `GpuUsage` (no torch, no psutil) |
| `system.py` | [System Monitor](Monitor-System) | `SystemMonitor` (psutil, plus a GPU probe) |
| `gpu.py` | [GPU Probes](Monitor-GPU) | `GpuProbe`, `CudaProbe`, `MpsProbe`, `create_gpu_probe` |

## Package exports

`monitor/__init__.py` imports `usage` eagerly and exports `GpuUsage` and `UsageSample`.
`CudaProbe`, `GpuProbe`, `MpsProbe`, `create_gpu_probe` (from `gpu`) and `SystemMonitor`
(from `system`) are listed in an explicit `_LAZY` map and imported on first access by a
PEP 562 `__getattr__`. Both modules import torch, and `system` also imports psutil. So the
window process can unpickle `UsageSample`s without loading either. `__all__` and `dir()`
list all seven names.

## GPU support

| Training device | Probe | Memory | Utilisation |
|---|---|---|---|
| `cuda` (NVIDIA; also AMD ROCm builds of PyTorch) | `CudaProbe` | `torch.cuda.mem_get_info`: used = total − free | `torch.cuda.utilization`, when NVML is available (the `gpu` extra installs `nvidia-ml-py`: `pip install ".[gpu]"`) |
| `mps` (Apple silicon) | `MpsProbe` | `torch.mps.driver_allocated_memory()`; total from `recommended_max_memory()` when available | not exposed |
| anything else (CPU, …) | none | | |

Without a probe `UsageSample.gpu` is `None` and the panel shows no GPU rows.

## Example

```python
from hollywood_training import formatting as fmt
from hollywood_training.monitor import SystemMonitor

monitor = SystemMonitor("cpu")      # a torch.device, a device string or SessionInfo.device
sample = monitor.sample()
print(f"cpu {sample.cpu_percent:.1f}% · process {sample.process_cpu_percent:.0f}% · "
      f"memory {fmt.data_size(sample.memory_used)} / {fmt.data_size(sample.memory_total)}")
print(monitor.has_gpu, sample.gpu)  # False None
```

"""GPU usage probes for the device a model trains on.

* **CUDA** (NVIDIA, and AMD ROCm builds of PyTorch): memory from
  ``torch.cuda.mem_get_info``; utilisation from ``torch.cuda.utilization`` when NVML is
  available (``pip install hollywood-training[gpu]``), otherwise omitted.
* **MPS** (Apple silicon): memory allocated by PyTorch; utilisation is not exposed.
* Anything else (CPU): no probe - the GUI then hides the GPU section.
"""

from __future__ import annotations

from typing import Protocol

import torch

from .usage import GpuUsage


class GpuProbe(Protocol):
    def sample(self) -> GpuUsage | None: ...


class CudaProbe:
    """Reads the CUDA device ``index``."""

    def __init__(self, index: int) -> None:
        self._index = index
        self._name = torch.cuda.get_device_name(index)
        self._utilization_available = True

    def sample(self) -> GpuUsage | None:
        try:
            free, total = torch.cuda.mem_get_info(self._index)
        except RuntimeError:
            return None
        utilization = None
        if self._utilization_available:
            try:
                utilization = float(torch.cuda.utilization(self._index))
            except Exception:  # NVML missing or unsupported: stop asking
                self._utilization_available = False
        return GpuUsage("cuda", self._name, utilization, int(total - free), int(total))


class MpsProbe:
    """Reads Apple's Metal Performance Shaders backend."""

    def sample(self) -> GpuUsage | None:
        mps = getattr(torch, "mps", None)
        if mps is None:
            return None
        try:
            used = int(mps.driver_allocated_memory())
        except (AttributeError, RuntimeError):
            return None
        total = None
        recommended = getattr(mps, "recommended_max_memory", None)
        if callable(recommended):
            try:
                total = int(recommended())
            except RuntimeError:
                total = None
        return GpuUsage("mps", "Apple GPU (MPS)", None, used, total)


def create_gpu_probe(device: str | torch.device | None) -> GpuProbe | None:
    """A probe for ``device`` when it is a GPU, else None."""
    if device is None:
        return None
    try:
        device = torch.device(str(device).split(" ", 1)[0])
    except (RuntimeError, TypeError):
        return None
    if device.type == "cuda" and torch.cuda.is_available():
        index = device.index if device.index is not None else torch.cuda.current_device()
        return CudaProbe(index)
    if device.type == "mps":
        backend = getattr(torch.backends, "mps", None)
        if backend is not None and backend.is_available():
            return MpsProbe()
    return None

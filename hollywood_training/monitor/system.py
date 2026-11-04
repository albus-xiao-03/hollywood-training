"""``SystemMonitor``: CPU, memory and (when applicable) GPU usage of the machine and process.

Uses ``psutil`` so it works the same on Linux, macOS and Windows. Sampling is cheap and
non-blocking; call :meth:`SystemMonitor.sample` periodically (the GUI does so twice a
second).
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass

import psutil
import torch

from .gpu import GpuProbe, GpuUsage, create_gpu_probe


@dataclass(frozen=True, slots=True)
class UsageSample:
    """One reading of the machine and of this process."""

    timestamp: float
    cpu_percent: float  # whole machine, 0-100
    process_cpu_percent: float  # this process, 100 = one full core
    cpu_count: int
    memory_used: int  # bytes, whole machine
    memory_total: int
    process_memory: int  # resident set size of this process, bytes
    gpu: GpuUsage | None = None

    @property
    def memory_fraction(self) -> float:
        return self.memory_used / self.memory_total if self.memory_total else 0.0


class SystemMonitor:
    """Samples usage for a training session running on ``device``."""

    def __init__(self, device: str | torch.device | None = None) -> None:
        self._process = psutil.Process(os.getpid())
        self._cpu_count = psutil.cpu_count(logical=True) or 1
        self._gpu: GpuProbe | None = create_gpu_probe(device)
        # cpu_percent(None) measures since the previous call; prime both counters.
        psutil.cpu_percent(None)
        self._process.cpu_percent(None)

    @property
    def has_gpu(self) -> bool:
        return self._gpu is not None

    def sample(self) -> UsageSample:
        memory = psutil.virtual_memory()
        try:
            process_memory = int(self._process.memory_info().rss)
            process_cpu = float(self._process.cpu_percent(None))
        except psutil.Error:
            process_memory, process_cpu = 0, 0.0
        return UsageSample(
            timestamp=time.monotonic(),
            cpu_percent=float(psutil.cpu_percent(None)),
            process_cpu_percent=process_cpu,
            cpu_count=self._cpu_count,
            memory_used=int(memory.total - memory.available),
            memory_total=int(memory.total),
            process_memory=process_memory,
            gpu=self._gpu.sample() if self._gpu is not None else None,
        )

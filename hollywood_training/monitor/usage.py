"""Usage readings as plain data (no torch, no psutil): what frontends receive and draw."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class GpuUsage:
    """One GPU reading."""

    backend: str  # "cuda" or "mps"
    name: str
    utilization: float | None  # percent, when the driver exposes it
    memory_used: int | None  # bytes
    memory_total: int | None  # bytes


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

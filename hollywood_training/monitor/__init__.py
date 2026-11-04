"""Usage monitoring: CPU, memory and GPU. ``UsageSample`` / ``GpuUsage`` need no torch."""

from importlib import import_module

from .usage import GpuUsage, UsageSample

_LAZY = {
    "CudaProbe": ".gpu",
    "GpuProbe": ".gpu",
    "MpsProbe": ".gpu",
    "create_gpu_probe": ".gpu",
    "SystemMonitor": ".system",
}

__all__ = [
    "CudaProbe",
    "GpuProbe",
    "GpuUsage",
    "MpsProbe",
    "SystemMonitor",
    "UsageSample",
    "create_gpu_probe",
]


def __getattr__(name: str):
    # Exports that need torch are imported on first use (PEP 562), so the torch-free
    # data modules of this package can be used on their own - e.g. by the GUI process.
    module = _LAZY.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(module, __name__), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(__all__)

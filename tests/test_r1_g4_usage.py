"""R1 GUI layout 4 - "The GUI **MUST** provide the information of current usage status,
including but not limited to: `memory usage`, `cpu usage`, `gpu usage` (Only when
appliable)."

* the panel: waits for the first reading; shows CPU (machine, process, cores) and memory
  (used / total, process) from the reading; GPU rows exactly when the reading has a GPU -
  utilisation and memory, or memory alone when the driver gives no utilisation;
* the readings: ``SystemMonitor`` samples plausible values of this machine; there is no GPU
  probe for a CPU device; a CUDA device (faked here: this machine has none) is read through
  ``torch.cuda`` and the reading reaches the sample;
* live: an in-process run's window samples by itself; the default window process receives
  the readings taken in the training process.
"""

from __future__ import annotations

import os
import time

import psutil
import pytest
import torch
from htspec.models import make_setup
from htspec.probe_log import LINGER_CLOSE

import hollywood_training as ht
from hollywood_training import formatting as fmt
from hollywood_training.monitor import system as system_module
from hollywood_training.monitor.gpu import CudaProbe, create_gpu_probe
from hollywood_training.monitor.system import SystemMonitor
from hollywood_training.monitor.usage import GpuUsage, UsageSample

pytestmark = pytest.mark.req("R1.G4")

GIB = 1024**3


def sample(gpu: GpuUsage | None = None) -> UsageSample:
    return UsageSample(
        timestamp=time.monotonic(), cpu_percent=37.5, process_cpu_percent=112.0, cpu_count=8,
        memory_used=6 * GIB, memory_total=16 * GIB, process_memory=512 * 1024**2, gpu=gpu,
    )  # fmt: skip


def drawn(capture, panel) -> list[str]:
    return [text.strip() for text in capture.render(panel)]


# ------------------------------------------------------------------- the panel
def test_before_the_first_reading_the_panel_waits(rig, paint_capture):
    assert "SENSORS OFFLINE" in drawn(paint_capture, rig.window.usage)


def test_cpu_and_memory_are_shown_from_the_reading(rig, paint_capture):
    rig.window.usage.set_sample(sample())
    texts = drawn(paint_capture, rig.window.usage)
    for text in ("CPU", "37.5%", "proc 112% · 8 cores", "MEM", "6.0 GiB / 16.0 GiB",
                 "proc 512.0 MiB"):  # fmt: skip
        assert text in texts, text
    assert not {"GPU", "VRAM"} & set(texts)  # no GPU in the reading: no GPU rows


def test_gpu_rows_appear_with_a_gpu_reading(rig, paint_capture):
    reading = GpuUsage("cuda", "Fake RTX", 64.0, 3 * GIB, 8 * GIB)
    rig.window.usage.set_sample(sample(reading))
    texts = drawn(paint_capture, rig.window.usage)
    for text in ("GPU", "64.0%", "VRAM", "3.0 GiB / 8.0 GiB", "Fake RTX"):
        assert text in texts, text
    bars = paint_capture.bars_of("UsagePanel")
    assert len(bars) == 4  # CPU, MEM, GPU and VRAM levels


def test_without_a_utilisation_only_the_gpu_memory_is_shown(rig, paint_capture):
    reading = GpuUsage("mps", "Apple GPU (MPS)", None, 2 * GIB, None)
    rig.window.usage.set_sample(sample(reading))
    texts = drawn(paint_capture, rig.window.usage)
    assert "VRAM" in texts and "GPU" not in texts
    assert "2.0 GiB / ?" in texts


# ---------------------------------------------------------------- the readings
def test_the_system_monitor_reads_this_machine():
    monitor = SystemMonitor("cpu (4 threads)")
    time.sleep(0.2)
    reading = monitor.sample()
    assert 0.0 <= reading.cpu_percent <= 100.0
    assert reading.process_cpu_percent >= 0.0
    assert reading.cpu_count == psutil.cpu_count(logical=True)
    assert reading.memory_total == psutil.virtual_memory().total
    assert 0 < reading.memory_used <= reading.memory_total
    rss = psutil.Process(os.getpid()).memory_info().rss
    assert 0.5 * rss <= reading.process_memory <= 2 * rss
    assert reading.gpu is None and not monitor.has_gpu


@pytest.mark.parametrize("device", ["cpu", "cpu (10 threads)", "not a device", None])
def test_there_is_no_gpu_probe_for_a_cpu(device):
    assert create_gpu_probe(device) is None


@pytest.mark.skipif(torch.cuda.is_available(), reason="checks the fallback without CUDA")
def test_a_cuda_device_without_cuda_gets_no_probe():
    assert create_gpu_probe("cuda:0") is None


def test_a_cuda_device_is_read_through_torch(monkeypatch):
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda index: f"Fake GPU {index}")
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda index: (6 * GIB, 8 * GIB))
    monkeypatch.setattr(torch.cuda, "utilization", lambda index: 42)
    reading = CudaProbe(0).sample()
    assert reading == GpuUsage("cuda", "Fake GPU 0", 42.0, 2 * GIB, 8 * GIB)


def test_a_gpu_without_a_utilisation_counter_still_reports_its_memory(monkeypatch):
    def no_nvml(index):
        raise RuntimeError("pynvml missing")

    monkeypatch.setattr(torch.cuda, "get_device_name", lambda index: "Fake GPU")
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda index: (1 * GIB, 4 * GIB))
    monkeypatch.setattr(torch.cuda, "utilization", no_nvml)
    probe = CudaProbe(0)
    assert probe.sample() == GpuUsage("cuda", "Fake GPU", None, 3 * GIB, 4 * GIB)
    assert probe.sample().utilization is None


def test_a_gpu_reading_reaches_the_sample(monkeypatch):
    class FakeProbe:
        def sample(self):
            return GpuUsage("cuda", "Fake GPU", 12.0, GIB, 2 * GIB)

    monkeypatch.setattr(system_module, "create_gpu_probe", lambda device: FakeProbe())
    monitor = SystemMonitor("cuda:0 (Fake GPU)")
    assert monitor.has_gpu
    assert monitor.sample().gpu == GpuUsage("cuda", "Fake GPU", 12.0, GIB, 2 * GIB)


# ------------------------------------------------------------------------ live
def test_an_in_process_window_samples_the_machine_by_itself(pilot):
    seen = {}

    def look(window) -> None:
        reading = window.usage.sample
        seen["reading"] = reading

    pilot.on_epoch(2, pilot.after(0.7, look))
    pilot.on_finish(pilot.after(0.8, pilot.close))
    pilot.run(make_setup(epochs=3).process)
    reading = seen["reading"]
    assert reading is not None
    assert reading.memory_total == psutil.virtual_memory().total
    assert reading.cpu_count == psutil.cpu_count(logical=True)
    assert reading.gpu is None  # a CPU run: no GPU rows


@pytest.mark.subprocess
def test_the_window_process_receives_the_readings_of_the_training_process(probe):
    log = probe(script={"steps": [{**LINGER_CLOSE, "delay_ms": 1500}]})
    ht.run(make_setup(epochs=3).process)
    readings = [s["usage"] for s in log.states() if s["usage"]["samples"]]
    assert readings, "the window never received a usage reading"
    last = readings[-1]
    assert last["memory_total"] == psutil.virtual_memory().total
    rss = psutil.Process(os.getpid()).memory_info().rss  # sampled in *this* process
    assert 0.5 * rss <= last["process_memory"] <= 2 * rss
    assert 0.0 <= last["cpu"] <= 100.0 and last["gpu"] is None
    assert max(s["usage"]["samples"] for s in log.states()) >= 2  # it keeps sampling


def test_the_panel_formats_large_and_small_amounts():
    assert fmt.data_size(512) == "512 B"
    assert fmt.data_size(1536) == "1.5 KiB"
    assert fmt.data_size(3 * GIB) == "3.0 GiB"

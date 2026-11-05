"""G4: the GUI shows the current memory, CPU and - when applicable - GPU usage."""

from __future__ import annotations

import time

import psutil
import pytest
import torch

import hollywood_training as ht
from hollywood_training.gui import AnimationClock
from hollywood_training.gui.widgets import UsagePanel, hud
from hollywood_training.monitor import GpuUsage, SystemMonitor, UsageSample, create_gpu_probe

from .helpers import active_window, build_process, text_recorder

pytestmark = pytest.mark.spec("G4")

GIB = 2**30


def sample(gpu: GpuUsage | None = None) -> UsageSample:
    return UsageSample(time.monotonic(), 42.5, 87.0, 8, 6 * GIB, 16 * GIB, 512 * 2**20, gpu)


def make_panel(qtbot) -> tuple[UsagePanel, AnimationClock]:
    clock = AnimationClock(60)
    panel = UsagePanel(clock)
    qtbot.addWidget(panel)
    panel.resize(380, 220)
    return panel, clock


# ------------------------------------------------------------------ readings
def test_the_monitor_reads_cpu_and_memory_of_the_machine_and_the_process():
    monitor = SystemMonitor("cpu")
    first = monitor.sample()
    time.sleep(0.05)
    second = monitor.sample()
    for reading in (first, second):
        assert 0.0 <= reading.cpu_percent <= 100.0
        assert reading.process_cpu_percent >= 0.0
        assert reading.cpu_count == (psutil.cpu_count(logical=True) or 1)
        assert 0 < reading.memory_used <= reading.memory_total
        assert 0 < reading.process_memory < reading.memory_total
        assert 0.0 < reading.memory_fraction <= 1.0
        assert reading.gpu is None
    assert second.timestamp > first.timestamp
    assert not monitor.has_gpu


@pytest.mark.parametrize("device", [None, "cpu", "cpu (4 threads)", "not a device", "meta"])
def test_there_is_no_gpu_reading_without_a_gpu_device(device):
    assert create_gpu_probe(device) is None


@pytest.mark.skipif(torch.cuda.is_available(), reason="checks the machine without CUDA")
def test_asking_for_cuda_without_cuda_gives_no_gpu_section():
    assert create_gpu_probe("cuda:0") is None
    assert not SystemMonitor("cuda").has_gpu


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
def test_a_cuda_device_is_monitored():
    monitor = SystemMonitor("cuda:0")
    reading = monitor.sample().gpu
    assert monitor.has_gpu and reading is not None and reading.backend == "cuda"
    assert reading.memory_total > 0 and 0 <= reading.memory_used <= reading.memory_total
    assert reading.utilization is None or 0.0 <= reading.utilization <= 100.0


@pytest.mark.gpu
@pytest.mark.skipif(
    not (getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()),
    reason="needs Apple MPS",
)
def test_an_mps_device_is_monitored():
    reading = SystemMonitor("mps").sample().gpu
    assert reading is not None and reading.backend == "mps" and reading.memory_used >= 0


# ------------------------------------------------------------------ the panel
@pytest.mark.gui
def test_cpu_and_memory_rows_are_painted(qtbot, monkeypatch):
    texts = text_recorder(monkeypatch, hud)
    panel, _clock = make_panel(qtbot)
    panel.set_sample(sample())
    panel.grab()
    assert "CPU" in texts and "MEM" in texts
    assert any("42.5" in text for text in texts)  # machine CPU
    assert any("16.0 GiB" in text for text in texts)  # machine memory
    assert any("512.0 MiB" in text for text in texts)  # this process
    assert "GPU" not in texts and "VRAM" not in texts  # not applicable on CPU


@pytest.mark.gui
@pytest.mark.parametrize(
    ("gpu", "rows"),
    [
        (GpuUsage("cuda", "Test GPU", 73.0, 2 * GIB, 8 * GIB), {"GPU", "VRAM"}),
        (GpuUsage("mps", "Apple GPU (MPS)", None, GIB, None), {"VRAM"}),
        (GpuUsage("cuda", "No NVML", None, 2 * GIB, 8 * GIB), {"VRAM"}),
    ],
    ids=["cuda", "mps", "cuda-without-utilisation"],
)
def test_gpu_rows_appear_when_training_runs_on_a_gpu(qtbot, monkeypatch, gpu, rows):
    texts = text_recorder(monkeypatch, hud)
    panel, _clock = make_panel(qtbot)
    panel.set_sample(sample(gpu))
    panel.grab()
    assert {label for label in ("GPU", "VRAM") if label in texts} == rows
    assert "CPU" in texts and "MEM" in texts
    assert any(gpu.name in text for text in texts)


@pytest.mark.gui
def test_a_failing_sampler_never_disturbs_the_window(qtbot):
    panel, _clock = make_panel(qtbot)
    panel.set_sampler(lambda: 1 / 0)
    qtbot.wait(600)  # the panel keeps polling twice a second
    assert panel.sample is None
    panel.set_sampler(None)
    assert not panel.grab().isNull()


@pytest.mark.gui
def test_the_panel_keeps_polling_a_sampler(qtbot):
    panel, _clock = make_panel(qtbot)
    readings = []

    def sampler() -> UsageSample:
        readings.append(1)
        return sample()

    panel.set_sampler(sampler)
    qtbot.wait(1200)
    panel.stop()
    assert len(readings) >= 2 and panel.sample is not None


# ------------------------------------------------------------------ in a run
@pytest.mark.gui
def test_the_window_shows_live_readings_during_training(qapp):
    bundle = build_process(epochs=3)
    seen: list[UsageSample] = []

    @bundle.process.on_epoch_end
    def look(record):
        window = active_window()
        if window is not None and window.usage.sample is not None:
            seen.append(window.usage.sample)

    ht.run(bundle.process, isolate_gui=False, close_on_finish=True)
    assert seen, "no usage reading reached the window"
    assert seen[-1].memory_total == psutil.virtual_memory().total
    assert seen[-1].cpu_count == (psutil.cpu_count(logical=True) or 1)


@pytest.mark.gui
def test_readings_sent_by_the_training_process_reach_the_window_process(window_process):
    assert window_process.receive() == ("ready",)
    reading = sample(GpuUsage("cuda", "Remote GPU", 10.0, GIB, 4 * GIB))
    window_process.send("usage", reading)
    window_process.qtbot.waitUntil(lambda: window_process.window.usage.sample == reading)

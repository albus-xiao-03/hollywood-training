"""Usage monitoring, text formatting, the console frontend and run(gui=False)."""

from __future__ import annotations

import pytest
import torch

import hollywood_training as ht
from hollywood_training import formatting as fmt
from hollywood_training.monitor import SystemMonitor, create_gpu_probe


def test_system_monitor_sample_is_sane():
    monitor = SystemMonitor("cpu")
    sample = monitor.sample()
    assert 0 <= sample.cpu_percent <= 100 and sample.cpu_count >= 1
    assert 0 < sample.memory_used <= sample.memory_total
    assert sample.process_memory > 0 and 0 < sample.memory_fraction <= 1
    assert sample.gpu is None and not monitor.has_gpu


@pytest.mark.skipif(torch.cuda.is_available(), reason="checks the no-GPU path")
def test_no_gpu_probe_without_a_gpu():
    assert create_gpu_probe("cpu") is None
    assert create_gpu_probe("cuda:0") is None
    assert create_gpu_probe(None) is None
    assert create_gpu_probe("not a device") is None


def test_formatting():
    assert fmt.duration(3723.5) == "01:02:03.500"
    assert fmt.duration(None) == fmt.PLACEHOLDER
    assert fmt.loss(0.123456) == "0.1235" and fmt.loss(1.5e-5) == "1.500e-05"
    assert fmt.metric("accuracy", 0.931) == "93.1%"
    assert fmt.data_size(3 * 1024**3) == "3.0 GiB"
    assert fmt.learning_rate(float("nan")) == fmt.PLACEHOLDER


def test_run_in_console_mode(make_process, capsys):
    result = ht.run(make_process(epochs=3), gui=False)
    output = capsys.readouterr().out
    assert result.state is ht.TrainingState.FINISHED and result.epochs_completed == 3
    assert "HOLLYWOOD TRAINING :: SmallNet" in output
    assert "epoch 3/3" in output and "FINISHED: completed 3 epochs" in output


def test_run_rejects_bad_arguments(make_process):
    with pytest.raises(TypeError):
        ht.run(object(), gui=False)
    with pytest.raises(ValueError, match="start_paused"):
        ht.run(make_process(), gui=False, start_paused=True)


def test_run_raises_training_failed_with_the_cause(make_process, capsys):
    process = make_process(epochs=2)

    @process.train_step
    def train_step(batch):
        raise RuntimeError("bad batch")

    with pytest.raises(ht.TrainingFailedError) as info:
        ht.run(process, gui=False)
    assert isinstance(info.value.__cause__, RuntimeError)
    assert info.value.result.state is ht.TrainingState.FAILED
    assert "error: RuntimeError: bad batch" in capsys.readouterr().out

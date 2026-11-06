"""R2 requirement 5 - "**ALL** other specified features by `R1` **MUST NOT** be affected."

Every R1 test of this harness counts for R2.5 as well (``conftest.py``: a test of an R1
requirement is also reported under R2.5), because the R1 features must still hold on this
version. This module adds the checks aimed at the interplay of the R2 machinery - the
keyboard focus lock and the pointer guard, which filter the events of every widget and of
the native window - with the R1 features:

* the R1 public API, ``run()`` options, step decorators and commands are all still there;
* all six R1 panels are present and visible;
* painting still happens: the synchronous per-epoch repaint and the animation clock's
  repaints reach every panel (observed through what the panels' paint code draws, not
  with another event filter, which would run before the guard's);
* resizing and closing the window, the terminal's R1 keys (history, completion, Return);
* the console frontend (``gui=False``) still needs no GUI at all.
"""

from __future__ import annotations

import inspect
import textwrap

import pytest
from htspec.models import make_setup
from htspec.subproc import run_python
from PySide6.QtTest import QTest

import hollywood_training as ht
from hollywood_training.training.process import TrainingProcess

pytestmark = pytest.mark.req("R2.5")

R1_EXPORTS = {
    "Argument", "Command", "CommandContext", "CommandError", "EpochRecord", "NetworkTracker",
    "ProcessError", "TrainingFailedError", "TrainingProcess", "TrainingResult",
    "TrainingState", "__version__", "inspect_network", "is_tracked", "run", "track",
}  # fmt: skip
R1_RUN_OPTIONS = {
    "gui": True,
    "isolate_gui": True,
    "start_paused": False,
    "close_on_finish": False,
    "max_neurons": 20,
    "title": None,
    "log_every": None,
    "torch_threads": "auto",
}
R1_COMMAND_LINES = [
    "help", "status", "pause", "resume", "step", "epochs 30", "lr 0.02", "delay 0",
    "history", "arch", "weights", "save {tmp}/model.pt", "screenshot {tmp}/shot.png", "clear",
    "stop",
]  # fmt: skip
PANELS = ("header", "network", "terminal", "status", "chart", "usage")


def test_the_r1_public_api_is_still_exported():
    assert set(ht.__all__) >= R1_EXPORTS
    for name in R1_EXPORTS:
        assert getattr(ht, name) is not None, name


def test_run_still_takes_every_r1_option():
    parameters = inspect.signature(ht.run).parameters
    for name, default in R1_RUN_OPTIONS.items():
        assert name in parameters, name
        assert parameters[name].kind is inspect.Parameter.KEYWORD_ONLY, name
        assert parameters[name].default == default, name


def test_the_process_still_offers_the_r1_step_decorators():
    for name in ("train_step", "validation_step", "on_epoch_end", "command", "resolve_steps"):
        assert callable(getattr(TrainingProcess, name)), name


@pytest.mark.parametrize("line", R1_COMMAND_LINES)
def test_every_r1_command_still_runs(rig, tmp_path, line):
    rig.feeder.epochs(2)
    result = rig.type_line(line.format(tmp=tmp_path))
    assert result is not None and result.ok, result


def test_quit_still_stops_and_closes(rig):
    result = rig.type_line("quit")
    assert result is not None and result.ok
    rig.process_events(100)
    assert rig.control.stop_requested and not rig.window.isVisible()


def test_all_six_r1_panels_are_present_and_visible(rig):
    window = rig.window
    area = window.rect()
    for name in PANELS:
        widget = getattr(window, name)
        assert widget.isVisible(), name
        assert widget.width() > 50 and widget.height() > 50, name
        top_left = widget.mapTo(window, widget.rect().topLeft())
        bottom_right = widget.mapTo(window, widget.rect().bottomRight())
        assert area.contains(top_left) and area.contains(bottom_right), name


def test_the_per_epoch_repaint_still_reaches_every_data_panel(rig, paint_capture):
    window = rig.window
    paint_capture.clear()
    rig.feeder.train_epoch()  # MainWindow.apply -> render_now
    assert window.network.last_painted_epoch == 1
    assert window.chart.last_painted_epoch == 1
    assert "01/10" in paint_capture.strings("StatusPanel")  # the epoch counter was drawn
    assert any(text.startswith("TRAIN ") for text in paint_capture.strings("LossChart"))
    assert "TRAINING" in paint_capture.strings("HeaderBar")  # the state badge


def test_the_animation_still_repaints_every_panel(rig, paint_capture):
    """Each panel draws something on every frame once a probe line and a neuron
    highlight are shown; count those drawings over 0.8 s of idling."""
    rig.feeder.epochs(2)
    assert rig.type_line("probe 1").ok
    assert rig.type_line("neuron fc1 0").ok
    paint_capture.clear()
    QTest.qWait(800)
    frames = {
        "HeaderBar": paint_capture.strings("HeaderBar").count("TRAINING"),
        "StatusPanel": paint_capture.strings("StatusPanel").count("PROCESS TIMER"),
        "LossChart": paint_capture.strings("LossChart").count("EPOCH 1"),
        "NetworkView": paint_capture.strings("NetworkView").count("FC1 · neuron 0"),
        "UsagePanel": paint_capture.strings("UsagePanel").count("SENSORS OFFLINE"),
    }
    assert all(count >= 10 for count in frames.values()), frames


def test_the_window_still_resizes_and_closes(rig, qtbot):
    window = rig.window
    width = window.network.width()
    window.resize(1200, 800)
    rig.process_events(50)
    assert window.network.width() < width
    with qtbot.waitSignal(window.closing, timeout=2000):
        window.close()
    assert not window.isVisible()


def test_the_terminal_r1_keys_still_work_under_the_focus_lock(rig):
    rig.type_line("status")
    rig.type_line("history")
    rig.type("<Up>")
    assert rig.input_text == "history"
    rig.type("<Up>")
    assert rig.input_text == "status"
    rig.type("<Down>")
    assert rig.input_text == "history"
    rig.type("<Down>")
    assert rig.input_text == ""
    rig.type("hist<Tab>")
    assert rig.input_text == "history "
    rig.type("<Return>")
    assert rig.results[-1][1].command.name == "history"


def test_a_real_in_process_run_still_shows_every_epoch(pilot):
    pilot.on_finish(pilot.close)
    result = pilot.run(make_setup(epochs=5).process)
    assert result.epochs_completed == 5
    shown = {r.epoch for r in pilot.renders if r.network_epoch == r.chart_epoch == r.epoch}
    assert set(range(1, 6)) <= shown


@pytest.mark.subprocess
def test_the_console_frontend_still_needs_no_gui():
    code = textwrap.dedent(
        """
        import sys
        import torch
        from torch import nn
        from torch.utils.data import DataLoader, TensorDataset
        import hollywood_training as ht

        torch.manual_seed(0)
        model = nn.Sequential(nn.Linear(2, 4), nn.Tanh(), nn.Linear(4, 2))
        data = DataLoader(TensorDataset(torch.randn(32, 2), torch.randint(0, 2, (32,))), 8)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        process = ht.TrainingProcess(model, optimizer, data, loss_fn=nn.CrossEntropyLoss(),
                                     epochs=3)
        result = ht.run(process, gui=False)
        gui = sorted(m for m in sys.modules if m.startswith("PySide6"))
        print("GUI-MODULES", gui)
        print("STATE", result.state.value, result.epochs_completed)
        """
    )
    completed = run_python(["-c", code], timeout=120)
    assert completed.returncode == 0, completed.stderr
    assert "GUI-MODULES []" in completed.stdout
    assert "STATE finished 3" in completed.stdout
    assert "epoch 1/3" in completed.stdout

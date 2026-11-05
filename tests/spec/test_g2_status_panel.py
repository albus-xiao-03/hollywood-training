"""G2: the GUI shows the training status - total and current epoch, batch size, learning
rate, loss function, optimizer, normalization strategy, a progress bar and a process
timer - and keeps it current."""

from __future__ import annotations

import itertools
import re
import time

import pytest
import torch
from PySide6.QtCore import QPoint, QRect
from torch import nn
from torch.utils.data import DataLoader

import hollywood_training as ht
from hollywood_training.gui import MainWindow
from hollywood_training.gui.widgets import hud, status_panel
from hollywood_training.sync import BatchProgress, Phase, SessionState, StateChanged, TrainingState

from .helpers import (
    G2_LABELS,
    Bundle,
    GuiSession,
    RichNet,
    build_process,
    call_recorder,
    session_started,
    text_recorder,
    xor_data,
)

pytestmark = [pytest.mark.spec("G2"), pytest.mark.gui]

TIMER = re.compile(r"\d{2}:\d{2}:\d{2}\.\d{3}")


def rows_of(window) -> dict[str, str]:
    return dict(pair for row in window.status.rows() for pair in row)


def open_window(qtbot, process: ht.TrainingProcess) -> GuiSession:
    tracker, started = session_started(process)
    session = SessionState()
    window = MainWindow(session)
    qtbot.addWidget(window)
    with qtbot.waitExposed(window):
        window.show()
    bundle = Bundle(process, process.model, process.optimizer, process.loss_fn,
                    process.train_data, process.val_data)  # fmt: skip
    payload = GuiSession(window, session, tracker, bundle, started)
    payload.push(started)
    return payload


def glow_texts(calls) -> list[str]:
    return [args[2] for args, _kwargs in calls]


# ---------------------------------------------------------------- the content
def test_the_configuration_of_the_run_is_shown(gui):
    rows = rows_of(gui.window)
    assert rows["BATCH SIZE"] == "16"
    assert rows["LEARNING RATE"] == "0.1"
    assert rows["LOSS FUNCTION"].startswith("CrossEntropyLoss")
    assert rows["OPTIMIZER"].startswith("SGD")
    assert "z-score" in rows["NORMALIZATION"]


def test_details_of_optimizer_normalization_and_scheduler_are_shown(qtbot):
    torch.manual_seed(0)
    model = RichNet()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.003, weight_decay=1e-4)
    process = ht.TrainingProcess(
        model,
        optimizer,
        DataLoader(xor_data(64), batch_size=32),
        loss_fn=nn.CrossEntropyLoss(),
        epochs=4,
        scheduler=torch.optim.lr_scheduler.StepLR(optimizer, step_size=2),
        normalization="min-max",
    )
    rows = rows_of(open_window(qtbot, process).window)
    assert rows["BATCH SIZE"] == "32" and rows["LEARNING RATE"] == "0.003"
    assert "Adam" in rows["OPTIMIZER"] and "0.0001" in rows["OPTIMIZER"]
    assert "min-max" in rows["NORMALIZATION"] and "BatchNorm1d" in rows["NORMALIZATION"]
    assert rows["SCHEDULER"] == "StepLR" and "cpu" in rows["DEVICE"]


def test_a_loss_defined_inside_the_step_and_no_validation_are_stated(qtbot):
    bundle = build_process(epochs=2, validation=False)
    process = ht.TrainingProcess(
        bundle.model, bundle.optimizer, bundle.train_loader, epochs=2,
        train_step=lambda batch: 1.0,
    )  # fmt: skip
    rows = rows_of(open_window(qtbot, process).window)
    assert rows["LOSS FUNCTION"] and rows["LOSS FUNCTION"] != "—"
    assert rows["VAL LOSS"] == "n/a"


def test_every_required_item_is_painted(gui, monkeypatch):
    texts = text_recorder(monkeypatch, hud)
    glows = call_recorder(monkeypatch, status_panel, "draw_glow_mono")
    bars = call_recorder(monkeypatch, status_panel, "paint_bar")
    gui.complete_epoch(1)  # drawn synchronously, as part of the epoch hand-off
    assert "1/5" in glow_texts(glows)  # current / total epoch
    assert any(TIMER.fullmatch(text) for text in glow_texts(glows))  # process timer
    for label in ("EPOCH", "PROCESS TIMER", *G2_LABELS):
        assert label in texts, label
    for value in ("16", "0.1", "SGD"):
        assert value in texts, value
    assert any(text.startswith("RUN") and "20.0%" in text for text in texts)
    assert bars and bars[0][0][2] == pytest.approx(0.2)  # the epoch progress bar


# ---------------------------------------------------------------- staying current
def test_the_status_follows_every_epoch_and_every_setting_change(gui, monkeypatch):
    glows = call_recorder(monkeypatch, status_panel, "draw_glow_mono")
    for epoch in (1, 2, 3):
        gui.complete_epoch(epoch, train_loss=1.0 / epoch)
        assert f"{epoch}/5" in glow_texts(glows)
        assert rows_of(gui.window)["TRAIN LOSS"] == f"{1.0 / epoch:.4f}"
    gui.push(StateChanged(TrainingState.RUNNING, 3, 8, 0.05, 0.0, 1.0, "total epochs set to 8"))
    glows.clear()
    gui.window.status.repaint()
    assert "3/8" in glow_texts(glows)
    assert rows_of(gui.window)["LEARNING RATE"] == "0.05"


def test_the_process_timer_runs_while_training_and_stops_while_paused(gui, qtbot, monkeypatch):
    glows = call_recorder(monkeypatch, status_panel, "draw_glow_mono")

    def timer_now() -> str:
        glows.clear()
        gui.window.status.repaint()
        return next(text for text in glow_texts(glows) if TIMER.fullmatch(text))

    gui.complete_epoch(1)
    first = timer_now()
    time.sleep(0.15)
    assert timer_now() > first  # running: the timer advances between events
    gui.push(StateChanged(TrainingState.PAUSED, 1, 5, 0.1, 0.0, 0.5, "paused"))
    paused = timer_now()
    qtbot.wait(150)
    assert timer_now() == paused == "00:00:00.500"


def test_progress_bars_show_the_epoch_and_the_batch_progress(gui, monkeypatch):
    texts = text_recorder(monkeypatch, hud)
    bars = call_recorder(monkeypatch, status_panel, "paint_bar")
    gui.complete_epoch(1)
    gui.push(BatchProgress(2, Phase.TRAIN, 3, 6, 0.5, 16, 0.3))
    bars.clear()
    gui.window.status.repaint()
    epoch_fraction, batch_fraction = bars[0][0][2], bars[1][0][2]
    assert epoch_fraction == pytest.approx(0.2) and batch_fraction == pytest.approx(0.5)
    assert any("BATCH 3/6" in text for text in texts)


# ----------------------------------------------------------------------- layout
@pytest.mark.spec("G3", "G4", "G5")
@pytest.mark.parametrize("size", ["minimum", (1280, 800), (1680, 1000)], ids=str)
def test_no_panel_covers_another_at_any_allowed_window_size(gui, qtbot, size):
    """Every panel stays fully visible: at the smallest size the window allows, too
    (QA finding M1)."""
    window = gui.window
    width, height = (window.minimumWidth(), window.minimumHeight()) if size == "minimum" else size
    window.resize(width, height)
    deadline = time.monotonic() + 2.0
    while (window.width(), window.height()) != (width, height) and time.monotonic() < deadline:
        qtbot.wait(20)
    if (window.width(), window.height()) != (width, height):
        pytest.skip(f"the platform kept the window at {window.width()}x{window.height()}")
    qtbot.wait(50)
    panels = {
        "header": window.header,
        "network": window.network,
        "terminal": window.terminal,
        "status": window.status,
        "chart": window.chart,
        "usage": window.usage,
    }
    rects = {
        name: QRect(panel.mapTo(window, QPoint(0, 0)), panel.size())
        for name, panel in panels.items()
    }
    outside = [name for name, rect in rects.items() if not window.rect().contains(rect)]
    overlapping = [
        f"{a}/{b}" for a, b in itertools.combinations(rects, 2) if rects[a].intersects(rects[b])
    ]
    assert outside == [] and overlapping == [], {
        name: rect.getRect() for name, rect in rects.items()
    }


def test_every_required_item_is_painted_at_the_smallest_window_size(gui, qtbot, monkeypatch):
    window = gui.window
    window.resize(window.minimumSize())
    qtbot.wait(50)
    texts = text_recorder(monkeypatch, hud)
    glows = call_recorder(monkeypatch, status_panel, "draw_glow_mono")
    gui.complete_epoch(1, train_metrics={"accuracy": 0.5}, val_metrics={"accuracy": 0.4})
    for label in ("EPOCH", "PROCESS TIMER", *G2_LABELS):
        assert label in texts, label
    assert "1/5" in glow_texts(glows)

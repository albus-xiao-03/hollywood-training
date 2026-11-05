"""G3: the GUI draws the training and validation loss DYNAMICALLY with EACH epoch."""

from __future__ import annotations

import itertools
import math

import numpy as np
import pytest
from PySide6.QtCore import Qt

import hollywood_training as ht
from hollywood_training import formatting as fmt
from hollywood_training.gui import AnimationClock
from hollywood_training.gui.widgets import LossChart, decade_ticks, nice_ticks
from hollywood_training.gui.widgets import loss_chart as loss_chart_module
from hollywood_training.sync import SessionState

from .helpers import (
    active_window,
    build_process,
    epoch_completed,
    image_rgb,
    session_started,
    text_recorder,
)

pytestmark = [pytest.mark.spec("G3"), pytest.mark.gui]


def make_chart(qtbot, *, validation: bool = True, epochs: int = 10):
    """A standalone chart whose animation clock never ticks (frames are stable)."""
    bundle = build_process(epochs=epochs, validation=validation)
    tracker, started = session_started(bundle.process)
    session = SessionState()
    session.apply(started)
    clock = AnimationClock(60)
    chart = LossChart(session, clock)
    qtbot.addWidget(chart)
    chart.resize(480, 300)
    return chart, session, tracker, clock


def add_epoch(session, tracker, epoch, train, val, total=10):
    session.apply(epoch_completed(tracker, epoch, total=total, train_loss=train, val_loss=val))


def test_the_chart_is_redrawn_for_every_epoch(gui):
    for epoch in (1, 2, 3, 4):
        gui.complete_epoch(epoch, train_loss=1.0 / epoch, val_loss=1.2 / epoch)
        assert gui.window.chart.last_painted_epoch == epoch  # drawn during the hand-off


def test_both_curves_are_drawn_with_their_latest_values(qtbot, monkeypatch):
    texts = text_recorder(monkeypatch, loss_chart_module)
    chart, session, tracker, _clock = make_chart(qtbot)
    for epoch, (train, val) in enumerate([(0.9, 1.1), (0.6, 0.8), (0.4, 0.7)], start=1):
        add_epoch(session, tracker, epoch, train, val)
        texts.clear()
        chart.grab()
        assert f"TRAIN {fmt.loss(train)}" in texts
        assert f"VAL {fmt.loss(val)}" in texts
    assert any(text.startswith("BEST") and "@ 3" in text for text in texts)


def test_without_validation_only_the_training_curve_is_drawn(qtbot, monkeypatch):
    texts = text_recorder(monkeypatch, loss_chart_module)
    chart, session, tracker, _clock = make_chart(qtbot, validation=False)
    add_epoch(session, tracker, 1, 0.9, None)
    chart.grab()
    assert "TRAIN 0.9000" in texts
    assert not any(text.startswith("VAL ") for text in texts)


def test_each_new_epoch_extends_the_picture(qtbot):
    chart, session, tracker, _clock = make_chart(qtbot)
    add_epoch(session, tracker, 1, 0.9, 1.0)
    images = [image_rgb(chart.grab())]
    for epoch, loss in ((2, 0.7), (3, 0.5)):
        add_epoch(session, tracker, epoch, loss, loss + 0.1)
        images.append(image_rgb(chart.grab()))
    for before, after in itertools.pairwise(images):
        assert np.abs(before.astype(int) - after.astype(int)).sum() > 0


def test_the_drawn_history_is_the_real_history(qapp):
    """In a real run, what the chart draws at the start of each epoch is exactly the
    losses training produced so far."""
    bundle = build_process(epochs=4, native_steps=False)
    model, optimizer, loss_fn = bundle.model, bundle.optimizer, bundle.loss_fn
    drawn: dict[int, tuple[list, list, int | None]] = {}

    @bundle.process.train_step
    def train_step(batch):
        epoch = bundle.completed[0] + 1
        if epoch not in drawn:
            window = active_window()
            history = window.session.history  # what LossChart draws
            drawn[epoch] = (
                [r.train_loss for r in history],
                [r.val_loss for r in history],
                window.chart.last_painted_epoch,
            )
        inputs, targets = batch
        optimizer.zero_grad()
        loss = loss_fn(model(inputs), targets)
        loss.backward()
        optimizer.step()
        return loss

    result = ht.run(bundle.process, isolate_gui=False, close_on_finish=True)
    assert result.epochs_completed == 4
    for epoch in (2, 3, 4):
        train, val, painted = drawn[epoch]
        assert train == result.train_losses[: epoch - 1]
        assert val == result.val_losses[: epoch - 1]
        assert painted == epoch - 1


def test_clicking_switches_between_linear_and_logarithmic_scale(qtbot, monkeypatch):
    texts = text_recorder(monkeypatch, loss_chart_module)
    chart, session, tracker, _clock = make_chart(qtbot)
    add_epoch(session, tracker, 1, 0.9, 1.0)
    add_epoch(session, tracker, 2, 0.09, 0.2)
    with qtbot.waitExposed(chart):
        chart.show()
    assert not chart.log_scale
    qtbot.mouseClick(chart, Qt.MouseButton.LeftButton, pos=chart.rect().center())
    assert chart.log_scale
    texts.clear()
    chart.grab()
    assert "LOG" in texts


def test_non_positive_losses_fall_back_to_a_linear_axis(qtbot, monkeypatch):
    texts = text_recorder(monkeypatch, loss_chart_module)
    chart, session, tracker, _clock = make_chart(qtbot)
    add_epoch(session, tracker, 1, -0.5, 0.0)
    chart.set_log_scale(True)
    chart.grab()
    assert "LIN" in texts


@pytest.mark.parametrize(
    "losses",
    [[math.nan, 0.5], [math.inf, 0.5], [0.5, -math.inf], [math.nan, math.nan], [1e-12, 1e12]],
    ids=["nan", "inf", "-inf", "all-nan", "huge-range"],
)
def test_extreme_loss_values_are_drawn_without_failing(qtbot, losses):
    chart, session, tracker, _clock = make_chart(qtbot)
    for epoch, loss in enumerate(losses, start=1):
        add_epoch(session, tracker, epoch, loss, loss)
    for log in (False, True):
        chart.set_log_scale(log)
        assert not chart.grab().isNull()
    assert chart.last_painted_epoch == len(losses)


def test_long_runs_are_drawn(qtbot):
    chart, session, tracker, _clock = make_chart(qtbot, epochs=600)
    for epoch in range(1, 601):
        add_epoch(session, tracker, epoch, 1.0 / epoch, 1.1 / epoch, total=600)
    assert not chart.grab().isNull() and chart.last_painted_epoch == 600


def test_axis_ticks_are_round_numbers():
    assert nice_ticks(0, 1, 5) == pytest.approx([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ticks = nice_ticks(0.013, 0.87, 5)
    assert ticks[0] >= 0.013 and ticks[-1] <= 0.87 and len(ticks) >= 3
    assert decade_ticks(0.001, 10.0) == pytest.approx([0.001, 0.01, 0.1, 1.0, 10.0])
    assert nice_ticks(1.0, 1.0) == [1.0]  # an empty range does not loop forever

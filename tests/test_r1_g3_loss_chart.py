"""R1 GUI layout 3 - "The GUI **MUST** crave the loss figures **DYNAMICALLY** along with
**EACH** `epoch`, including: `training loss`, `validation loss`."

What the chart really draws (its curves are recorded as painter paths, its labels as text):

* before the first epoch it waits; after each epoch the training curve and the validation
  curve have exactly one more point - drawn during that epoch's update - and the legend
  shows the latest values (and the best epoch);
* the horizontal axis spans the requested epochs, so the curves grow towards the end; a
  changed total moves the axis;
* boundaries: no validation (one curve), NaN / inf losses (gaps, no failure), negative and
  constant losses, a single epoch, a long run (500 epochs) drawn quickly.

(The repaint of every epoch before the next one starts, in real runs, is verified in
``test_r1_05_synchronization.py``.)
"""

from __future__ import annotations

import time

import pytest
from htspec.models import make_setup
from PySide6.QtTest import QTest

from hollywood_training import formatting as fmt
from hollywood_training.gui.animation import AnimationClock
from hollywood_training.gui.theme import PALETTE
from hollywood_training.gui.widgets import LossChart
from hollywood_training.sync.events import EpochRecord
from hollywood_training.sync.session import SessionState

pytestmark = pytest.mark.req("R1.G3")


def curves(capture) -> dict[str, list[int]]:
    """Point counts of the drawn curve lines (not their glow), by series."""
    lines = {"train": [], "val": []}
    for path in capture.paths_of("LossChart"):
        if path.alpha != 255:
            continue  # the soft glow under each line
        if path.color == PALETTE.accent:
            lines["train"].append(path.elements)
        elif path.color == PALETTE.indigo:
            lines["val"].append(path.elements)
    return lines


def test_before_the_first_epoch_the_chart_waits(rig, paint_capture):
    assert "AWAITING FIRST EPOCH" in paint_capture.render(rig.window.chart)
    assert curves(paint_capture) == {"train": [], "val": []}


def test_each_epoch_adds_one_point_to_both_curves(rig, paint_capture):
    for epoch in range(1, 6):
        paint_capture.clear()
        rig.feeder.train_epoch()  # the window repaints during the epoch's update
        assert rig.window.chart.last_painted_epoch == epoch
        assert curves(paint_capture) == {"train": [epoch], "val": [epoch]}, epoch


def test_the_validation_curve_is_drawn_dashed_the_training_curve_solid(rig, paint_capture):
    rig.feeder.epochs(3)
    paint_capture.render(rig.window.chart)
    styles = {(p.color, p.dashed) for p in paint_capture.paths_of("LossChart") if p.alpha == 255}
    assert (PALETTE.accent, False) in styles and (PALETTE.indigo, True) in styles


def test_the_legend_shows_the_latest_values_and_the_best_epoch(rig, paint_capture):
    for val in (0.9, 0.4, 0.6):
        rig.feeder.train_epoch(val_loss=val)
    last = rig.feeder.records[-1]
    drawn = paint_capture.render(rig.window.chart)
    assert f"TRAIN {fmt.loss(last.train_loss)}" in drawn
    assert f"VAL {fmt.loss(0.6)}" in drawn
    assert f"BEST {fmt.loss(0.4)} @ 2" in drawn


def test_the_axis_spans_the_requested_epochs_and_follows_a_new_total(rig, paint_capture):
    rig.feeder.epochs(2)
    assert "10" in paint_capture.render(rig.window.chart)  # the last tick: all 10 epochs
    rig.feeder.train_epoch(total_epochs=25)
    assert "25" in paint_capture.render(rig.window.chart)
    assert "EPOCH" in paint_capture.strings("LossChart")


def test_without_validation_only_the_training_curve_is_drawn(make_rig, paint_capture):
    rig = make_rig(setup=make_setup(validation=False))
    rig.feeder.start()
    rig.feeder.epochs(3)
    drawn = paint_capture.render(rig.window.chart)
    assert curves(paint_capture) == {"train": [3], "val": []}
    assert not any(text.startswith("VAL ") for text in drawn)


def test_non_finite_losses_leave_gaps(rig, paint_capture):
    for train in (1.0, 0.9, float("nan"), 0.8, 0.7, float("inf"), 0.6, 0.5):
        rig.feeder.train_epoch(train_loss=train, val_loss=None)
    paint_capture.render(rig.window.chart)
    lines = [
        p for p in paint_capture.paths_of("LossChart")
        if p.alpha == 255 and p.color == PALETTE.accent
    ]  # fmt: skip
    assert len(lines) == 1
    assert lines[0].elements == 6  # the six finite points
    assert lines[0].moves == 3  # in three separate stretches: a gap at each non-finite loss


def test_negative_losses_are_drawn_on_the_axis(rig, paint_capture):
    for train in (-2.0, -1.5, -1.0):
        rig.feeder.train_epoch(train_loss=train, val_loss=None)
    drawn = paint_capture.render(rig.window.chart)
    assert curves(paint_capture)["train"] == [3]
    assert any(text.startswith("-") and text[1:2].isdigit() for text in drawn)


def test_constant_losses_are_drawn(rig, paint_capture):
    for _ in range(3):
        rig.feeder.train_epoch(train_loss=1.0, val_loss=1.0)
    paint_capture.render(rig.window.chart)
    assert curves(paint_capture) == {"train": [3], "val": [3]}


def test_a_single_epoch_is_drawn(rig, paint_capture):
    rig.feeder.train_epoch()
    paint_capture.render(rig.window.chart)
    assert curves(paint_capture) == {"train": [1], "val": [1]}


def test_a_long_run_is_drawn_quickly(qtbot, paint_capture):
    session = SessionState()
    session.history = [
        EpochRecord(epoch=k, train_loss=1.0 / k, val_loss=1.2 / k) for k in range(1, 501)
    ]
    session.total_epochs = 500
    session.history_version += 1
    chart = LossChart(session, AnimationClock())
    qtbot.addWidget(chart)
    chart.resize(700, 320)
    chart.show()  # a hidden widget does not paint
    QTest.qWaitForWindowExposed(chart, 5000)
    started = time.perf_counter()
    paint_capture.render(chart)
    assert time.perf_counter() - started < 1.0
    assert curves(paint_capture) == {"train": [500], "val": [500]}

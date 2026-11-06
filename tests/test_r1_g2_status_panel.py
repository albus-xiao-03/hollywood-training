"""R1 GUI layout 2 - "The GUI **MUST** demonstrate the information of training status, which
includes but not limited to: `total epoch`, `current epoch`, `batch size`, `learning rate`,
`loss function`, `optimizer`, `normalization strategy`, `progress bar`, `process timer`."

Each of the nine items, with the values the session really has (``StatusPanel.rows()`` and
what the panel draws):

* current / total epoch - zero-padded to the total, following a changed total;
* batch size - given, read from the ``DataLoader`` or inferred from the first batch;
* learning rate - initial, after changes, after a scheduler step in a real run;
* loss function, optimizer (with its settings), normalisation strategy (data + layers);
* progress bars - the run's fraction and the current epoch's batch fraction;
* process timer - ``HH:MM:SS.mmm``, running while training, standing still while paused;
* boundaries: before a session, no validation, metrics and best epoch, odd values.
"""

from __future__ import annotations

import functools
import re
import time

import pytest
import torch
from htspec.headless import HeadlessRun
from htspec.models import DeepNet, Setup, TinyNet, loader, make_setup
from torch import nn

import hollywood_training as ht
from hollywood_training import formatting as fmt
from hollywood_training.gui.animation import AnimationClock
from hollywood_training.gui.widgets import StatusPanel
from hollywood_training.sync.events import BatchProgress, Phase, SessionStarted, TrainingState
from hollywood_training.sync.session import SessionState

pytestmark = pytest.mark.req("R1.G2")

TIMER = re.compile(r"^\d\d:\d\d:\d\d\.\d{3}$")


def values(panel) -> dict[str, str]:
    return {label: value for row in panel.rows() for label, value in row}


def custom_setup(*, model=None, optimizer=None, loss_fn="default", **process_options) -> Setup:
    model = model if model is not None else TinyNet()
    optimizer = optimizer or torch.optim.SGD(model.parameters(), lr=0.05)
    data = loader()
    real_loss = nn.CrossEntropyLoss()
    process = ht.TrainingProcess(
        model, optimizer, data, epochs=10, train_step=lambda batch: 0.0,
        loss_fn=real_loss if loss_fn == "default" else loss_fn, **process_options,
    )  # fmt: skip
    return Setup(model, optimizer, real_loss, data, None, process)


def started(make_rig, setup: Setup):
    rig = make_rig(setup=setup)
    rig.feeder.start()
    return rig


# ------------------------------------------------------------------ the epochs
def test_before_a_session_the_status_stands_by(make_rig, paint_capture):
    assert "STANDING BY" in paint_capture.render(make_rig().window.status)


def test_the_current_and_the_total_epoch_are_shown(rig, paint_capture):
    rig.feeder.epochs(3)
    drawn = paint_capture.render(rig.window.status)
    assert "EPOCH" in drawn and "03/10" in drawn


def test_the_epoch_counter_is_padded_to_the_width_of_the_total(make_rig, paint_capture):
    rig = make_rig(epochs=1000)
    rig.feeder.start()
    rig.feeder.epochs(2)
    assert "0002/1000" in paint_capture.render(rig.window.status)


def test_the_total_follows_a_change(rig, paint_capture):
    rig.feeder.train_epoch(total_epochs=25)
    assert "01/25" in paint_capture.render(rig.window.status)


# -------------------------------------------------------------- the settings
def test_the_rows_show_the_settings_of_the_run(rig, paint_capture):
    rig.feeder.epochs(1)
    shown = values(rig.window.status)
    record = rig.feeder.records[-1]
    assert shown["BATCH SIZE"] == "16"
    assert shown["LEARNING RATE"] == "0.05"
    assert shown["LOSS FUNCTION"] == "CrossEntropyLoss (mean)"
    assert shown["OPTIMIZER"] == "SGD"
    assert shown["NORMALIZATION"] == "layers: none"
    assert shown["TRAIN LOSS"] == fmt.loss(record.train_loss)
    assert shown["VAL LOSS"] == fmt.loss(record.val_loss)
    assert shown["PARAMETERS"] == "51"
    drawn = paint_capture.render(rig.window.status)
    for label in ("BATCH SIZE", "LEARNING RATE", "LOSS FUNCTION", "OPTIMIZER", "NORMALIZATION"):
        assert label in drawn and shown[label] in drawn, label


def test_the_learning_rate_follows_a_change(rig):
    rig.feeder.epochs(1)
    for group in rig.setup.optimizer.param_groups:
        group["lr"] = 0.012
    rig.feeder.train_epoch()
    assert values(rig.window.status)["LEARNING RATE"] == "0.012"


def test_a_scheduled_learning_rate_reaches_the_status_in_a_real_run(pilot):
    torch.manual_seed(0)
    model = TinyNet()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.08)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)
    process = ht.TrainingProcess(model, optimizer, loader(), loss_fn=nn.CrossEntropyLoss(),
                                 epochs=4, scheduler=scheduler)  # fmt: skip
    seen = {}

    def read(window) -> None:
        seen["lr"] = values(window.status)["LEARNING RATE"]

    pilot.on_epoch(2, read, sync=True)  # exactly at epoch 2, before training goes on
    pilot.on_finish(pilot.close)
    pilot.run(process)
    assert seen == {"lr": fmt.learning_rate(0.02)}  # 0.08 halved after epochs 1 and 2


@pytest.mark.parametrize(
    ("loss_fn", "shown"),
    [
        (nn.MSELoss(reduction="sum"), "MSELoss (sum)"),
        (nn.CrossEntropyLoss(), "CrossEntropyLoss (mean)"),
        (torch.nn.functional.mse_loss, "mse_loss"),
        (None, "defined in step"),
    ],
    ids=["module", "module-default", "function", "in-the-step"],
)
def test_the_loss_function_is_named(make_rig, loss_fn, shown):
    rig = started(make_rig, custom_setup(loss_fn=loss_fn))
    assert values(rig.window.status)["LOSS FUNCTION"] == shown


def my_loss(prediction, target):
    return ((prediction - target) ** 2).mean()


def test_a_plain_function_and_a_partial_are_named(make_rig):
    rig = started(make_rig, custom_setup(loss_fn=my_loss))
    assert values(rig.window.status)["LOSS FUNCTION"] == "my_loss"
    rig = started(make_rig, custom_setup(loss_fn=functools.partial(my_loss)))
    assert values(rig.window.status)["LOSS FUNCTION"] == "partial"


def _optimizers(model):
    return {
        "adam": (torch.optim.Adam(model.parameters()), "Adam (betas=0.9/0.999)"),
        "sgd-momentum": (
            torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, nesterov=True),
            "SGD (momentum=0.9, nesterov)",
        ),
        "adamw": (torch.optim.AdamW(model.parameters()), "AdamW (betas=0.9/0.999, wd=0.01)"),
        "two-groups": (
            torch.optim.SGD(
                [{"params": model.fc1.parameters()}, {"params": model.fc2.parameters(), "lr": 0.2}],
                lr=0.1,
            ),
            "SGD (2 param groups)",
        ),
    }


@pytest.mark.parametrize("kind", ["adam", "sgd-momentum", "adamw", "two-groups"])
def test_the_optimizer_and_its_settings_are_named(make_rig, kind):
    model = TinyNet()
    optimizer, shown = _optimizers(model)[kind]
    rig = started(make_rig, custom_setup(model=model, optimizer=optimizer))
    assert values(rig.window.status)["OPTIMIZER"] == shown


def test_the_normalisation_strategy_names_the_data_and_the_layers(make_rig):
    class TwoNorms(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.fc1 = nn.Linear(2, 4)
            self.norm1 = nn.BatchNorm1d(4)
            self.fc2 = nn.Linear(4, 4)
            self.norm2 = nn.BatchNorm1d(4)
            self.fc3 = nn.Linear(4, 3)

    setup = custom_setup(model=TwoNorms(), normalization="z-score (inputs)")
    rig = started(make_rig, setup)
    expected = "data: z-score (inputs) · layers: BatchNorm1d ×2"
    assert values(rig.window.status)["NORMALIZATION"] == expected
    rig = started(make_rig, custom_setup(model=DeepNet()))
    assert values(rig.window.status)["NORMALIZATION"] == "layers: BatchNorm1d"


# --------------------------------------------------------------- batch size
def test_a_given_batch_size_is_shown(make_rig):
    rig = started(make_rig, custom_setup(batch_size=7))
    assert values(rig.window.status)["BATCH SIZE"] == "7"


def test_an_unknown_batch_size_is_inferred_from_the_first_batch(qtbot):
    batches = [(torch.randn(4, 2), torch.randint(0, 3, (4,))) for _ in range(3)]
    model = TinyNet()
    process = ht.TrainingProcess(model, torch.optim.SGD(model.parameters(), 0.1), batches,
                                 loss_fn=nn.CrossEntropyLoss(), epochs=1)  # fmt: skip
    run = HeadlessRun(process)
    run.run()
    session = SessionState()
    panel = StatusPanel(session, AnimationClock())
    qtbot.addWidget(panel)
    for received in run.received:
        session.apply(received.event)
        if isinstance(received.event, SessionStarted):
            assert values(panel)["BATCH SIZE"] == fmt.PLACEHOLDER  # not known yet
        if isinstance(received.event, BatchProgress):
            break
    assert values(panel)["BATCH SIZE"] == "4"


# ------------------------------------------------------------ progress bars
def test_the_run_progress_bar_shows_the_fraction_of_epochs_done(rig, paint_capture):
    rig.feeder.epochs(3)
    drawn = paint_capture.render(rig.window.status)
    bars = paint_capture.bars_of("StatusPanel")
    assert bars[0].fraction == pytest.approx(0.3)
    assert "RUN 30.0%" in drawn


@pytest.mark.parametrize(("phase", "name"), [(Phase.TRAIN, "TRAINING"),
                                             (Phase.VALIDATION, "VALIDATION")])  # fmt: skip
def test_the_batch_progress_bar_shows_the_fraction_of_the_epoch(rig, paint_capture, phase, name):
    rig.feeder.epochs(1)
    rig.feeder.progress(2, total=4, phase=phase)
    drawn = paint_capture.render(rig.window.status)
    assert f"BATCH 2/4 · {name}" in drawn
    assert paint_capture.bars_of("StatusPanel")[1].fraction == pytest.approx(0.5)


# -------------------------------------------------------------- the timer
def _timer(capture, panel) -> str:
    texts = [t for t in capture.render(panel) if TIMER.match(t)]
    assert texts, "no process timer drawn"
    return texts[0]


def test_the_process_timer_runs_while_training_and_stands_while_paused(rig, paint_capture):
    panel = rig.window.status
    rig.feeder.epochs(1)
    first = _timer(paint_capture, panel)
    time.sleep(0.3)
    second = _timer(paint_capture, panel)
    assert second > first  # the timer advances while training runs
    rig.feeder.paused()
    rig.feeder.state(TrainingState.PAUSED, "paused")
    held = _timer(paint_capture, panel)
    time.sleep(0.3)
    assert _timer(paint_capture, panel) == held
    assert "PROCESS TIMER" in paint_capture.strings("StatusPanel")


# ---------------------------------------------------------------- boundaries
def test_without_validation_the_val_loss_says_not_applicable(make_rig):
    rig = make_rig(setup=make_setup(validation=False))
    rig.feeder.start()
    rig.feeder.epochs(1)
    assert values(rig.window.status)["VAL LOSS"] == "n/a"


def test_metrics_and_the_best_epoch_are_shown(rig):
    rig.feeder.train_epoch(metrics={"accuracy": 0.8}, val_metrics={"accuracy": 0.7})
    shown = values(rig.window.status)
    record = rig.feeder.records[-1]
    assert shown["ACCURACY"] == "80.0%" and shown["VAL ACCURACY"] == "70.0%"
    assert shown["BEST LOSS"] == f"{fmt.loss(record.val_loss)} @ 1"


def test_odd_values_do_not_break_the_status(rig, paint_capture):
    rig.setup.optimizer.param_groups[0]["lr"] = float("nan")
    rig.feeder.train_epoch(learn=False, train_loss=float("inf"), val_loss=float("nan"))
    shown = values(rig.window.status)
    assert shown["LEARNING RATE"] == fmt.PLACEHOLDER
    assert shown["TRAIN LOSS"] == "inf"
    paint_capture.render(rig.window.status)  # draws without an error

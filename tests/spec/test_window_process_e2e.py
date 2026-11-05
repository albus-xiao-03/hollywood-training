"""End-to-end: the default ``ht.run(process)`` with the window in its own process.

A probe inside the real window process (``child_hooks/``) records what the window received,
drew and acknowledged, and plays a scripted user typing into the real terminal.
"""

from __future__ import annotations

import time
import warnings

import numpy as np
import pytest

import hollywood_training as ht
from hollywood_training.sync import TrainingState as State

from .helpers import (
    TinyNet,
    build_process,
    install_child_probe,
    record_deliveries,
    window_process_children,
)

pytestmark = [pytest.mark.e2e, pytest.mark.gui]


def index_of(records: list[dict], predicate) -> int:
    return next(i for i, row in enumerate(records) if predicate(row))


@pytest.mark.spec("R5", "G1", "G3")
def test_every_epoch_is_drawn_in_the_window_before_training_continues(monkeypatch, tmp_path):
    probe = install_child_probe(monkeypatch, tmp_path)
    deliveries = record_deliveries(monkeypatch)
    bundle = build_process(epochs=6)
    result = ht.run(bundle.process, close_on_finish=True)
    assert result.state is State.FINISHED and result.epochs_completed == 6
    probe.assert_installed()

    records = probe.records()
    applied = probe.applied("EpochCompleted")
    assert [row["epoch"] for row in applied] == [1, 2, 3, 4, 5, 6]  # every epoch, in order
    for row in applied:
        assert row["network_painted"] == row["epoch"], row  # drawn when acknowledged
        assert row["chart_painted"] == row["epoch"], row
        np.testing.assert_allclose(  # the REAL weights of that epoch were drawn
            row["weights"]["fc1"], bundle.weights[row["epoch"]]["fc1.weight"].numpy(), atol=1e-6
        )

    received = {
        row["sequence"]: row["epoch"]
        for row in probe.records("received")
        if row.get("event") == "EpochCompleted"
    }
    acks = {
        row["sequence"]: row["t"]
        for row in probe.records("sent")
        if row["message"] == "ack" and row["sequence"] in received
    }
    assert sorted(received) == sorted(acks)  # each epoch acknowledged once drawn
    for sequence, epoch in received.items():
        drawn = index_of(records, lambda r, e=epoch: r["kind"] == "applied" and r.get("event")
                         == "EpochCompleted" and r["epoch"] == e)  # fmt: skip
        acked = index_of(records, lambda r, s=sequence: r["kind"] == "sent"
                         and r["message"] == "ack" and r.get("sequence") == s)  # fmt: skip
        assert drawn < acked

    # Across the two processes (time.monotonic is system-wide): the window had finished
    # drawing epoch e before the hand-off returned, and epoch e+1 started only after that.
    drawn_at = {row["epoch"]: row["t"] for row in applied}
    handed_off = {d.epoch: d for d in deliveries if d.event == "EpochCompleted"}
    for epoch in range(1, 6):
        delivery = handed_off[epoch]
        assert delivery.acknowledged
        assert drawn_at[epoch] <= delivery.end + 0.005, f"hand-off {epoch} ended before drawing"
        next_epoch = bundle.first_batch_time(epoch + 1)
        assert next_epoch > delivery.end, f"epoch {epoch + 1} started before the hand-off"
        assert next_epoch > drawn_at[epoch] - 0.005, f"epoch {epoch + 1} started before drawing"


@pytest.mark.spec("R7", "G5")
def test_commands_typed_into_the_window_control_the_training(monkeypatch, tmp_path):
    checkpoint = tmp_path / "checkpoint.pt"
    probe = install_child_probe(
        monkeypatch,
        tmp_path,
        [
            {"on": {"event": "TrainingPaused", "nth": 1},
             "do": [{"type": "status"}, {"type": "step 2"}]},
            {"on": {"event": "TrainingPaused", "nth": 2},
             "do": [{"type": "lr 0.05"}, {"type": "epochs 4"}, {"type": "resume"}]},
            {"on": {"event": "StateChanged", "state": "finished", "nth": 1},
             "do": [{"type": "history"}, {"type": f'save "{checkpoint}"'},
                    {"type": "quit", "after_ms": 600}]},
        ],
    )  # fmt: skip
    bundle = build_process(epochs=3)
    result = ht.run(bundle.process, start_paused=True)
    probe.assert_installed()
    assert result.state is State.FINISHED and result.epochs_completed == 4
    assert [r.learning_rate for r in result.history] == pytest.approx([0.1, 0.1, 0.05, 0.05])
    assert checkpoint.exists()
    typed = [row["line"] for row in probe.records("sent") if row["message"] == "command"]
    assert typed[:6] == ["status", "step 2", "lr 0.05", "epochs 4", "resume", "history"]
    replies = [row["text"] for row in probe.records("terminal") if row["level"] != "command"]
    assert any("state       paused" in text for text in replies)  # status answered
    assert any("learning rate set to 0.05" in text for text in replies)  # lr applied
    assert any(  # the history table, printed in the window
        text.lstrip().startswith("epoch") and "train" in text and "val" in text for text in replies
    )


@pytest.mark.spec("R5", "R7")
def test_closing_the_window_stops_the_training(monkeypatch, tmp_path):
    close_at_epoch_3 = {"on": {"event": "EpochCompleted", "epoch": 3}, "do": [{"close": True}]}
    install_child_probe(monkeypatch, tmp_path, [close_at_epoch_3])
    bundle = build_process(epochs=500, step_delay=0.002)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = ht.run(bundle.process, close_on_finish=True)  # a missed close cannot hang
    assert result.state is State.STOPPED and 3 <= result.epochs_completed < 500
    assert not [w for w in caught if "unexpectedly" in str(w.message)]  # a normal close
    assert window_process_children() == []


@pytest.mark.spec("R5")
def test_a_crashed_window_stops_the_training_with_a_warning(monkeypatch):
    bundle = build_process(epochs=500, step_delay=0.002)

    @bundle.process.on_epoch_end
    def crash_the_window(record):
        if record.epoch == 2:
            for child in window_process_children():
                child.kill()

    started = time.monotonic()
    with pytest.warns(RuntimeWarning, match="unexpectedly"):
        result = ht.run(bundle.process, close_on_finish=True)  # a missed kill cannot hang
    assert time.monotonic() - started < 60  # no hang
    assert result.state is State.STOPPED and result.epochs_completed < 10


@pytest.mark.spec("R7", "G5")
def test_the_screenshot_command_saves_the_window(monkeypatch, tmp_path):
    shot = tmp_path / "shots" / "window.png"
    install_child_probe(
        monkeypatch,
        tmp_path,
        [{"on": {"event": "StateChanged", "state": "finished", "nth": 1},
          "do": [{"type": f'screenshot "{shot}"'}, {"close": True, "after_ms": 1500}]}],
    )  # fmt: skip
    ht.run(build_process(epochs=2).process)
    assert shot.exists() and shot.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


@pytest.mark.spec("G4")
def test_usage_readings_reach_the_window(monkeypatch, tmp_path):
    probe = install_child_probe(monkeypatch, tmp_path)
    bundle = build_process(epochs=4, step_delay=0.05)  # about 1.5 s of training
    ht.run(bundle.process, close_on_finish=True)
    readings = probe.records("usage")
    assert readings, "no CPU / memory reading arrived in the window"
    for reading in readings:
        assert 0.0 <= reading["cpu_percent"] <= 100.0
        assert 0 < reading["memory_used"] <= reading["memory_total"]
        assert reading["gpu"] is False  # a CPU run shows no GPU section


@pytest.mark.spec("G6", "R4", "R9")
def test_the_window_names_the_network_and_runs_without_pytorch(monkeypatch, tmp_path):
    @ht.track(name="Named in the window")
    class WindowNet(TinyNet):
        pass

    probe = install_child_probe(monkeypatch, tmp_path)
    ht.run(build_process(epochs=1, model=WindowNet()).process, close_on_finish=True)
    started = probe.applied("SessionStarted")
    assert started and "WindowNet" in started[0]["title"]
    (closing,) = probe.records("closing")
    assert closing["torch_loaded"] is False  # the window process is light (README)
    assert closing["web_modules"] == []


@pytest.mark.spec("R5", "G1")
def test_starting_paused_shows_the_initial_network_before_any_training(monkeypatch, tmp_path):
    probe = install_child_probe(
        monkeypatch,
        tmp_path,
        [
            {
                "on": {"event": "TrainingPaused", "nth": 1},
                "do": [{"type": "resume", "after_ms": 300}],
            }
        ],
    )
    bundle = build_process(epochs=2)
    result = ht.run(bundle.process, start_paused=True, close_on_finish=True)
    assert result.epochs_completed == 2
    (first,) = probe.applied("SessionStarted")
    assert first["network_painted"] == 0  # the untrained network is on screen
    np.testing.assert_allclose(first["weights"]["fc1"], bundle.initial["fc1.weight"].numpy())
    assert probe.applied("TrainingPaused")[0]["state"] == "paused"


@pytest.mark.spec("R4")
def test_a_window_that_cannot_start_fails_fast_and_clearly(monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "no-such-platform")
    started = time.monotonic()
    with pytest.raises(RuntimeError, match="GUI process"):
        ht.run(build_process(epochs=1).process, close_on_finish=True)
    assert time.monotonic() - started < 60
    assert window_process_children() == []

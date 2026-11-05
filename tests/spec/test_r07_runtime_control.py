"""R7 (integration): the commands control a training session while it runs - from the
command thread of a frontend, and typed into the in-process window's terminal (G5)."""

from __future__ import annotations

import threading
import time

import numpy as np
import pytest
import torch

import hollywood_training as ht
from hollywood_training.sync import EpochCompleted, LogLevel, LogMessage, TrainingPaused
from hollywood_training.sync import TrainingState as State

from .helpers import ManagedRun, build_process, type_command

pytestmark = pytest.mark.spec("R7")


def paused(run: ManagedRun) -> bool:
    return run.session.state is State.PAUSED


def log_lines(run: ManagedRun, level: LogLevel | None = None) -> list[str]:
    return [
        event.text
        for event in run.frontend.of_type(LogMessage)
        if level is None or event.level is level
    ]


def test_pause_takes_effect_at_a_batch_boundary_and_resume_continues():
    bundle = build_process(epochs=10_000, step_delay=0.002)
    with ManagedRun(bundle.process) as run:
        run.wait_for(lambda: len(bundle.calls) > 3, "training to start")
        assert run.command("pause").ok
        run.wait_for(lambda: paused(run), "the pause to take effect")
        calls = len(bundle.calls)
        time.sleep(0.3)
        assert len(bundle.calls) == calls, "batches ran while paused"
        pause = run.frontend.of_type(TrainingPaused)[-1]
        drawn = pause.weights.layer("fc1").matrix  # what the window shows while paused
        np.testing.assert_array_equal(drawn, bundle.model.fc1.weight.detach().numpy())
        assert run.command("resume").ok
        run.wait_for(lambda: len(bundle.calls) > calls + 3, "training to continue")
        assert run.session.state is State.RUNNING


def test_step_runs_exactly_the_requested_epochs_then_pauses_again():
    bundle = build_process(epochs=50)
    with ManagedRun(bundle.process, start_paused=True) as run:
        run.wait_for(lambda: paused(run), "the initial pause")
        assert bundle.calls == []
        assert run.command("step 3").ok
        run.wait_for(lambda: paused(run) and run.session.epoch == 3, "a pause after 3 epochs")
        time.sleep(0.2)
        assert run.session.epoch == 3 and bundle.completed[0] == 3
        assert run.command("step").ok
        run.wait_for(lambda: paused(run) and run.session.epoch == 4, "a pause after 1 more")
        assert run.command("stop").ok
        result = run.join()
    assert result.epochs_completed == 4


def test_stop_ends_the_session_at_the_next_batch_boundary():
    bundle = build_process(epochs=10_000, step_delay=0.001)
    with ManagedRun(bundle.process) as run:
        run.wait_for(lambda: run.session.epoch >= 2, "two epochs")
        assert run.command("stop").ok
        calls_at_stop = len(bundle.calls)
        result = run.join()
    assert result.state is State.STOPPED
    assert 2 <= result.epochs_completed < 10_000
    assert len(bundle.calls) - calls_at_stop <= 1  # at most the batch already under way


def test_a_new_learning_rate_is_never_applied_in_the_middle_of_a_step():
    bundle = build_process(epochs=10_000, step_delay=0.002)
    with ManagedRun(bundle.process) as run:
        run.wait_for(lambda: run.session.epoch >= 1, "one epoch")
        assert run.command("lr 0.0125").ok
        submitted = run.session.epoch
        # the action is applied at the next safe point, so epoch submitted+2 starts with it
        run.wait_for(lambda: run.session.epoch >= submitted + 2, "an epoch begun with the new lr")
        assert run.command("stop").ok
        result = run.join()
    assert all(before == after for before, after in bundle.step_lrs), "lr changed mid-step"
    assert bundle.optimizer.param_groups[0]["lr"] == pytest.approx(0.0125)
    assert result.history[-1].learning_rate == pytest.approx(0.0125)
    assert any("0.0125" in text for text in log_lines(run, LogLevel.SUCCESS))


def test_epochs_can_shorten_a_running_session():
    bundle = build_process(epochs=10_000, step_delay=0.001)
    with ManagedRun(bundle.process) as run:
        run.wait_for(lambda: run.session.epoch >= 1, "one epoch")
        target = run.session.epoch + 2
        assert run.command(f"epochs {target}").ok
        result = run.join()
    assert result.state is State.FINISHED and result.epochs_completed == target


def test_epochs_continues_a_finished_session():
    bundle = build_process(epochs=2)
    with ManagedRun(bundle.process, linger=True) as run:
        run.wait_for(lambda: run.session.state is State.FINISHED, "the first finish")
        assert run.session.epoch == 2
        assert run.command("epochs 4").ok
        run.wait_for(
            lambda: run.session.state is State.FINISHED and run.session.epoch == 4,
            "the extended finish",
        )
        assert run.command("stop").ok
        result = run.join()
    assert result.state is State.FINISHED and result.epochs_completed == 4
    assert [e.record.epoch for e in run.frontend.of_type(EpochCompleted)] == [1, 2, 3, 4]


def test_save_while_training_is_done_by_the_training_thread(monkeypatch, tmp_path):
    threads = []
    original_save = torch.save

    def recording_save(*args, **kwargs):
        threads.append(threading.current_thread())
        return original_save(*args, **kwargs)

    monkeypatch.setattr(torch, "save", recording_save)
    bundle = build_process(epochs=10_000, step_delay=0.001)
    path = tmp_path / "checkpoint.pt"
    with ManagedRun(bundle.process) as run:
        run.wait_for(lambda: run.session.epoch >= 1, "one epoch")
        assert run.command(f'save "{path}"').ok
        run.wait_for(path.exists, "the checkpoint")
        assert run.command("stop").ok
        run.join()
    assert threads == [run.thread]
    state = torch.load(path, weights_only=True)
    assert state.keys() == bundle.model.state_dict().keys()


def test_custom_command_actions_run_on_the_training_thread():
    bundle = build_process(epochs=10_000, step_delay=0.001)
    ran_on = []

    @bundle.process.command("probe", summary="ask the live model something")
    def probe(context):
        def action(target):
            ran_on.append(threading.current_thread())
            return f"model has {sum(p.numel() for p in target.model.parameters())} parameters"

        return context.submit(action, "probe the model")

    with ManagedRun(bundle.process) as run:
        run.wait_for(lambda: run.session.epoch >= 1, "one epoch")
        assert run.command("probe").ok
        run.wait_for(lambda: ran_on, "the action")
        assert run.command("stop").ok
        run.join()
    assert ran_on == [run.thread]
    assert any("parameters" in text for text in log_lines(run, LogLevel.SUCCESS))


def test_a_failing_action_is_reported_and_training_goes_on():
    bundle = build_process(epochs=10_000, step_delay=0.001)

    @bundle.process.command("explode", summary="an action that fails")
    def explode(context):
        def action(target):
            raise ValueError("no such thing")

        return context.submit(action, "explode")

    with ManagedRun(bundle.process) as run:
        run.wait_for(lambda: run.session.epoch >= 1, "one epoch")
        assert run.command("explode").ok
        run.wait_for(lambda: any("no such thing" in t for t in log_lines(run)), "the error")
        epoch = run.session.epoch
        run.wait_for(lambda: run.session.epoch >= epoch + 2, "training to go on")
        assert run.command("stop").ok
        result = run.join()
    errors = log_lines(run, LogLevel.ERROR)
    assert any("explode" in text and "ValueError" in text for text in errors)
    assert result.state is State.STOPPED


def test_quit_stops_the_training():
    bundle = build_process(epochs=10_000, step_delay=0.001)
    with ManagedRun(bundle.process) as run:
        run.wait_for(lambda: run.session.epoch >= 1, "one epoch")
        assert run.command("quit").ok
        result = run.join()
    assert result.state is State.STOPPED


def test_commands_from_several_threads_at_once_are_safe():
    bundle = build_process(epochs=10_000, step_delay=0.0005)
    errors: list[BaseException] = []
    with ManagedRun(bundle.process) as run:
        run.wait_for(lambda: run.session.epoch >= 1, "one epoch")

        def hammer(lines: list[str]) -> None:
            try:
                for _ in range(20):
                    for line in lines:
                        run.command(line)
            except BaseException as exc:  # pragma: no cover - reported below
                errors.append(exc)

        workers = [
            threading.Thread(target=hammer, args=(lines,))
            for lines in (["pause", "resume"], ["lr 0.05", "lr 0.1"], ["status", "history 3"])
        ]
        for worker in workers:
            worker.start()
        for worker in workers:
            worker.join(30)
        assert run.command("resume").ok
        epoch = run.session.epoch
        run.wait_for(lambda: run.session.epoch >= epoch + 1, "training to go on")
        assert run.command("stop").ok
        result = run.join()
    assert errors == [] and result.state is State.STOPPED


# ------------------------------------------------------------ typed in the window
@pytest.mark.gui
@pytest.mark.spec("G5")
def test_commands_typed_into_the_window_control_the_run(drive):
    bundle = build_process(epochs=3)

    def state(window) -> State:
        return window.session.state

    driver = drive(
        [
            ("initial pause", lambda w: state(w) is State.PAUSED,
             lambda w: type_command(w, "status")),
            ("step 2", lambda w: "state       paused" in w.terminal.text(),
             lambda w: type_command(w, "step 2")),
            ("paused after 2", lambda w: state(w) is State.PAUSED and w.session.epoch == 2,
             lambda w: [type_command(w, line) for line in ("lr 0.05", "epochs 4", "resume")]),
            ("finished", lambda w: state(w) is State.FINISHED and w.session.epoch == 4,
             lambda w: type_command(w, "history")),
            ("quit", lambda w: "train" in w.terminal.text().split("history")[-1],
             lambda w: type_command(w, "quit")),
        ],
        timeout=60,
    )  # fmt: skip
    result = ht.run(bundle.process, isolate_gui=False, start_paused=True)
    driver.check()
    assert result.state is State.FINISHED and result.epochs_completed == 4
    assert [r.learning_rate for r in result.history] == pytest.approx([0.1, 0.1, 0.05, 0.05])

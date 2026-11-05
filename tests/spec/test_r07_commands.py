"""R7: a set of well-defined commands controls the managed training at runtime.

Command-level tests: every documented command exists, is described, validates its
arguments and does what the README documents. Integration with a running training
session is covered in ``test_r07_runtime_control.py``.
"""

from __future__ import annotations

import re
from types import SimpleNamespace

import pytest
import torch

import hollywood_training as ht
from hollywood_training.commands import CommandContext, create_registry, split_command_line
from hollywood_training.sync import (
    SessionState,
    StateChanged,
    TrainingControl,
    TrainingState,
    TrainingTarget,
)

from .helpers import TinyNet, build_process, epoch_completed

pytestmark = pytest.mark.spec("R7")

# README "Commands" table, and the aliases listed in the wiki's command reference
DOCUMENTED = {
    "help", "status", "pause", "resume", "step", "stop", "epochs", "lr", "delay",
    "history", "arch", "weights", "save", "screenshot", "clear", "quit",
}  # fmt: skip
ALIASES = {
    "?": "help",
    "continue": "resume",
    "start": "resume",
    "layers": "arch",
    "architecture": "arch",
    "cls": "clear",
    "exit": "quit",
}


def numbers_in(text: str) -> list[float]:
    return [float(value) for value in re.findall(r"[-+]?\d+\.\d+", text)]


# ------------------------------------------------------------------- definitions
def test_the_documented_commands_and_aliases_exist(commands):
    registry = commands.context.registry
    assert set(registry.names()) >= DOCUMENTED
    for alias, name in ALIASES.items():
        assert registry.get(alias) is not None and registry.get(alias).name == name


def test_every_command_is_well_defined(commands):
    registry = commands.context.registry
    for command in registry:
        assert command.name == command.name.lower() and " " not in command.name
        assert command.summary.strip(), f"{command.name} has no summary"
        assert command.usage.split()[0] == command.name
        explained = commands.run(f"help {command.name}")
        assert explained.ok and f"usage: {command.usage}" in explained.output
        for argument in command.arguments:
            assert argument.name in explained.output


def test_help_lists_every_command(commands):
    listing = commands.run("help")
    assert listing.ok
    for command in commands.context.registry:
        assert command.usage in listing.output and command.summary in listing.output
    assert commands.run("?").output == listing.output


# ------------------------------------------------------------------- training
def test_pause_and_resume(commands):
    assert commands.run("pause").ok and commands.control.paused
    again = commands.run("pause")
    assert again.ok and again.output and commands.control.paused  # idempotent, explained
    assert commands.run("resume").ok and not commands.control.paused
    again = commands.run("continue")
    assert again.ok and again.output and not commands.control.paused


def test_step_runs_a_given_number_of_epochs_then_pauses(commands):
    assert commands.run("step").ok and commands.control.stepping_until == 2  # 1 done + 1
    assert commands.run("step 3").ok and commands.control.stepping_until == 4
    assert not commands.control.paused  # runs now, pauses later


@pytest.mark.parametrize("line", ["step 0", "step -1", "step abc", "step 1.5", "step 1 2"])
def test_step_rejects_invalid_counts(commands, line):
    result = commands.run(line)
    assert not result.ok and result.output
    assert commands.control.stepping_until is None


def test_stop_ends_the_run(commands):
    assert commands.run("stop").ok and commands.control.stop_requested


def test_epochs_changes_the_total(commands):
    assert commands.run("epochs 9").ok
    assert commands.control.total_epochs == 9 and commands.session.total_epochs == 9
    commands.control.epoch_completed(3)
    assert not commands.run("epochs 2").ok  # fewer than already completed
    assert commands.control.total_epochs == 9
    assert commands.run("epochs 3").ok and commands.control.total_epochs == 3  # finish now


@pytest.mark.parametrize(
    "line", ["epochs", "epochs 0", "epochs -4", "epochs 2.5", "epochs many", "epochs 5 6"]
)
def test_epochs_rejects_invalid_totals(commands, line):
    result = commands.run(line)
    assert not result.ok and result.output
    assert commands.control.total_epochs == 5


def test_lr_is_applied_by_the_training_thread_between_batches(commands):
    result = commands.run("lr 0.025")
    assert result.ok and result.output
    assert commands.bundle.optimizer.param_groups[0]["lr"] == pytest.approx(0.1)  # not yet
    messages = commands.apply_queued_actions()  # the training thread's next safe point
    assert commands.bundle.optimizer.param_groups[0]["lr"] == pytest.approx(0.025)
    assert any("0.025" in (message or "") for message in messages)


def test_lr_sets_every_parameter_group():
    model = TinyNet()
    optimizer = torch.optim.SGD(
        [{"params": model.fc1.parameters(), "lr": 0.1}, {"params": model.fc2.parameters()}],
        lr=0.01,
    )
    control = TrainingControl(3)
    control.close()  # training over: the change is applied immediately
    context = CommandContext(
        control=control,
        session=SessionState(),
        registry=create_registry(),
        target=TrainingTarget(model, optimizer),
    )
    assert context.registry.execute("lr 0.5", context).ok
    assert [group["lr"] for group in optimizer.param_groups] == [0.5, 0.5]


@pytest.mark.parametrize(
    "line", ["lr", "lr 0", "lr -0.1", "lr nan", "lr inf", "lr fast", "lr 0.1 0.2"]
)
def test_lr_rejects_invalid_values(commands, line):
    result = commands.run(line)
    assert not result.ok and result.output
    assert commands.control.take_actions() == []


@pytest.mark.parametrize(
    ("line", "seconds"), [("delay 250", 0.25), ("delay 0", 0.0), ("delay 1.5", 0.0015)]
)
def test_delay_sets_a_wait_after_every_epoch(commands, line, seconds):
    assert commands.run(line).ok
    assert commands.control.delay == pytest.approx(seconds)


@pytest.mark.parametrize("line", ["delay", "delay -1", "delay nan", "delay inf", "delay slow"])
def test_delay_rejects_invalid_values(commands, line):
    result = commands.run(line)
    assert not result.ok and result.output
    assert commands.control.delay == 0.0


# ----------------------------------------------------------------- inspection
def test_status_reports_the_current_training_state(commands):
    output = commands.run("status").output
    assert "TinyNet" in output and "1/5" in output and "running" in output
    assert "0.9000" in output and "0.8000" in output  # train / validation loss
    assert "accuracy" in output


def test_history_lists_the_completed_epochs(commands):
    commands.session.apply(
        epoch_completed(commands.tracker, 2, total=5, train_loss=0.7, val_loss=0.6)
    )
    everything = commands.run("history").output
    assert "0.9000" in everything and "0.7000" in everything
    last = commands.run("history 1").output
    assert "0.7000" in last and "0.9000" not in last
    assert commands.run("history 1000").ok
    assert not commands.run("history 0").ok and not commands.run("history x").ok


@pytest.mark.parametrize("line", ["arch", "layers", "architecture"])
def test_arch_describes_the_tracked_network(commands, line):
    output = commands.run(line).output
    assert "TinyNet" in output
    assert "fc1" in output and "fc2" in output and "Linear 2→6" in output


def test_weights_reports_the_real_weights(commands):
    overview = commands.run("weights")
    assert overview.ok and "fc1" in overview.output and "fc2" in overview.output
    detail = commands.run("weights fc1")
    assert detail.ok and "fc1" in detail.output
    real = commands.session.weights.layer("fc1").stats
    assert any(abs(value - real.norm) < 1e-4 for value in numbers_in(detail.output))
    unknown = commands.run("weights nope")
    assert not unknown.ok and "fc1" in unknown.output  # tells what exists


def test_inspection_before_any_data_is_harmless():
    context = CommandContext(
        control=TrainingControl(2),
        session=SessionState(),
        registry=create_registry(),
        target=TrainingTarget(TinyNet(), None),
    )
    for line in ("status", "history", "arch", "weights"):
        result = context.registry.execute(line, context)
        assert result.ok and result.output, line


# -------------------------------------------------------------------- session
def test_save_writes_the_models_state_dict(commands, tmp_path):
    target = tmp_path / "nested" / "folder" / "model weights.pt"
    result = commands.run(f'save "{target}"')
    assert result.ok and not target.exists()  # queued for the training thread
    commands.apply_queued_actions()
    state = torch.load(target, weights_only=True)
    expected = commands.bundle.model.state_dict()
    assert state.keys() == expected.keys()
    for key, value in state.items():
        torch.testing.assert_close(value, expected[key])


def test_save_failures_are_reported(commands, tmp_path):
    blocker = tmp_path / "a-file"
    blocker.write_text("not a folder")
    commands.control.close()  # runs immediately, so the failure shows in the result
    result = commands.run(f'save "{blocker / "model.pt"}"')
    assert not result.ok and result.output
    assert not commands.run("save").ok  # the path is required


def test_window_commands_use_the_window(commands, tmp_path):
    shot = tmp_path / "shot.png"
    assert commands.run(f'screenshot "{shot}"').ok
    assert commands.frontend.screenshots == [shot]
    assert commands.run("clear").ok and commands.frontend.cleared == 1
    assert commands.run("cls").ok and commands.frontend.cleared == 2
    assert commands.run("quit").ok
    assert commands.frontend.closed == 1 and commands.control.stop_requested


def test_window_commands_without_a_window_fail_cleanly(commands_without_window):
    env = commands_without_window
    for line in ("screenshot", "clear"):
        result = env.run(line)
        assert not result.ok and result.output
    assert env.run("exit").ok and env.control.stop_requested


# ------------------------------------------------------------- invalid input
@pytest.mark.parametrize(
    "line", ["", "   ", "frobnicate", 'save "unterminated', "pause now", "help nosuch"]
)
def test_malformed_lines_produce_an_error_message_not_an_exception(commands, line):
    result = commands.run(line)
    assert not result.ok and result.output
    assert not commands.control.stop_requested and not commands.control.paused


def test_unknown_commands_suggest_close_matches(commands):
    result = commands.run("histroy")
    assert not result.ok and "history" in result.output


def test_arguments_may_be_quoted():
    assert split_command_line('save "my model.pt"') == ["save", "my model.pt"]


def test_completion_offers_matching_names_and_aliases(commands):
    registry = commands.context.registry
    assert set(registry.complete("s")) >= {"save", "screenshot", "start", "status", "step", "stop"}
    assert registry.complete("hist") == ["history"]
    assert registry.complete("zzz") == []


# ----------------------------------------------------------- session lifecycle
@pytest.mark.parametrize("line", ["pause", "resume", "step", "epochs 9", "delay 10"])
def test_run_changing_commands_are_refused_once_the_session_ended(commands, line):
    commands.control.close()
    result = commands.run(line)
    assert not result.ok and result.output


@pytest.mark.parametrize("line", ["status", "history", "arch", "weights", "help"])
def test_inspection_keeps_working_once_the_session_ended(commands, line):
    commands.control.close()
    assert commands.run(line).ok


def test_lr_and_save_apply_immediately_once_the_session_ended(commands, tmp_path):
    commands.control.close()
    assert commands.run("lr 0.3").ok
    assert commands.bundle.optimizer.param_groups[0]["lr"] == pytest.approx(0.3)
    path = tmp_path / "weights.pt"
    assert commands.run(f'save "{path}"').ok and path.exists()


def test_commands_respect_the_training_state(commands):
    commands.session.apply(StateChanged(TrainingState.FINISHED, 5, 5, 0.1, 0.0, 1.0, "done"))
    assert not commands.run("pause").ok  # nothing to pause
    assert not commands.run("step 2").ok  # 'epochs <n>' extends a finished run instead
    assert commands.run("epochs 8").ok and commands.control.total_epochs == 8
    commands.session.apply(StateChanged(TrainingState.STOPPED, 5, 8, 0.1, 0.0, 1.0, "stop"))
    assert not commands.run("epochs 9").ok  # a stopped run cannot be extended


# ------------------------------------------------------------- custom commands
def _custom_registry(**declaration):
    process = build_process().process

    @process.command("scale", summary="scale a number", **declaration)
    def scale(context, factor, unit):
        return f"{factor * 2:g}{unit}"

    return create_registry(process.commands)


def test_custom_commands_take_typed_and_validated_arguments():
    registry = _custom_registry(
        arguments=[
            ht.Argument("factor", float, "the number"),
            ht.Argument("unit", str, "x or %", optional=True, default="x", choices=("x", "%")),
        ]
    )
    assert registry.execute("scale 1.5", None).output == "3x"
    assert registry.execute("scale 2 %", None).output == "4%"
    for line in ("scale two", "scale 1 km", "scale", "scale 1 x extra"):
        assert not registry.execute(line, None).ok, line
    listing = registry.execute("help", SimpleNamespace(registry=registry)).output
    assert "scale" in listing and "CUSTOM" in listing.upper()


def test_errors_in_command_handlers_become_error_results():
    process = build_process().process

    @process.command("boom", summary="fails")
    def boom(context):
        raise RuntimeError("kaput")

    result = create_registry(process.commands).execute("boom", None)
    assert not result.ok and "RuntimeError" in result.output and "kaput" in result.output


@pytest.mark.parametrize("name", ["Pause2", "two words", "", "UPPER"])
def test_custom_command_names_must_be_single_lowercase_words(name):
    process = build_process().process
    with pytest.raises(ValueError):
        process.command(name, summary="x")(lambda context: None)


def test_custom_commands_cannot_silently_replace_builtins():
    process = build_process().process
    process.command("pause", summary="mine")(lambda context: None)
    with pytest.raises(ValueError):
        create_registry(process.commands)
    process = build_process().process
    process.command("go", summary="mine", aliases=("stop",))(lambda context: None)
    with pytest.raises(ValueError):
        create_registry(process.commands)


def test_required_arguments_cannot_follow_optional_ones():
    process = build_process().process
    arguments = [ht.Argument("a", optional=True), ht.Argument("b")]
    with pytest.raises(ValueError):
        process.command("odd", summary="x", arguments=arguments)(lambda context, a, b: None)


def test_actions_run_on_the_training_side_or_immediately_after_the_end(commands):
    ran = []
    queued = commands.context.submit(lambda target: ran.append(target) or "done", "probe")
    assert queued is None and ran == []  # waits for the training thread
    assert commands.apply_queued_actions() == ["done"]
    assert ran[0].model is commands.bundle.model
    commands.control.close()
    assert commands.context.submit(lambda target: "now", "probe") == "now"

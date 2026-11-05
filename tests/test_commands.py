"""Command parsing, dispatch and the built-in command set."""

from __future__ import annotations

import pytest
import torch

import hollywood_training as ht
from hollywood_training.commands import (
    Argument,
    Command,
    CommandContext,
    CommandRegistry,
    create_registry,
    split_command_line,
)
from hollywood_training.sync import (
    EpochCompleted,
    EpochRecord,
    SessionStarted,
    SessionState,
    TrainingControl,
    TrainingState,
    TrainingTarget,
)
from hollywood_training.tracking import NetworkTracker
from hollywood_training.training.describe import build_session_info


def _noop(ctx, *args):
    return "ran " + " ".join(str(a) for a in args)


# ---------------------------------------------------------------------- registry
def test_registry_names_aliases_and_conflicts():
    registry = CommandRegistry([Command("go", "start", _noop, aliases=("run",))])
    assert registry.get("RUN").name == "go" and "go" in registry
    with pytest.raises(ValueError):
        registry.register(Command("run", "clash", _noop))
    registry.register(Command("run", "replacement", _noop), replace=True)
    assert registry.get("run").summary == "replacement" and registry.get("go") is None
    assert registry.complete("r") == ["run"]


def test_command_declaration_rules():
    with pytest.raises(ValueError):
        Command("Bad", "uppercase", _noop)
    with pytest.raises(ValueError):
        Command("x", "order", _noop, arguments=(Argument("a", optional=True), Argument("b")))


def test_parsing_conversion_and_errors():
    arguments = (Argument("n", int), Argument("m", int, optional=True, default=5))
    registry = CommandRegistry([Command("add", "add", _noop, arguments=arguments)])
    assert registry.execute("add 2", None).output == "ran 2 5"
    assert registry.execute("add 2 3", None).output == "ran 2 3"
    for line, message in [
        ("add", "missing n"),
        ("add 1 2 3", "too many"),
        ("add x", "invalid n"),
        ("ad 1", "did you mean add"),
        ("", "empty"),
    ]:
        result = registry.execute(line, None)
        assert not result.ok and message in result.output


def test_handler_exceptions_become_failed_results():
    registry = CommandRegistry([Command("boom", "fails", lambda ctx: 1 / 0)])
    result = registry.execute("boom", None)
    assert not result.ok and "ZeroDivisionError" in result.output


def test_quoted_arguments():
    assert split_command_line('save "my model.pt"') == ["save", "my model.pt"]


# --------------------------------------------------------------------- built-ins
@pytest.fixture
def context(make_process):
    process = make_process(epochs=5)
    control = TrainingControl(5)
    tracker = NetworkTracker(process.model)
    session = SessionState()
    info = build_session_info(
        process, tracker.architecture.describe(), total_epochs=5, neuron_limit=20
    )
    session.apply(SessionStarted(info, tracker.snapshot(0), TrainingState.RUNNING))
    record = EpochRecord(1, 0.9, 0.8, {"accuracy": 0.5}, {}, 0.1, 0.01, 6, 2)
    session.apply(EpochCompleted(record, tracker.snapshot(1), 5, 0.1, 16, 0.05))
    registry = create_registry(process.commands)
    target = TrainingTarget(process.model, process.optimizer)
    ctx = CommandContext(control=control, session=session, registry=registry, target=target)
    ctx.process = process  # for assertions
    return ctx


def run(ctx, line):
    return ctx.registry.execute(line, ctx)


def test_training_controls(context):
    assert run(context, "pause").ok and context.control.paused
    assert run(context, "pause").output == "already paused"
    assert run(context, "resume").ok and not context.control.paused
    assert run(context, "step 3").ok and context.control.stepping_until == 3
    assert run(context, "delay 250").ok and context.control.delay == 0.25
    assert run(context, "epochs 9").ok and context.control.total_epochs == 9
    assert not run(context, "epochs 0").ok
    assert run(context, "stop").ok and context.control.stop_requested


def test_learning_rate_is_applied_by_the_training_thread(context):
    result = run(context, "lr 0.5")
    assert result.ok and "next batch boundary" in result.output
    assert context.process.optimizer.param_groups[0]["lr"] == 0.1  # not yet
    ((action, _),) = context.control.take_actions()
    target = TrainingTarget(context.process.model, context.process.optimizer)
    assert action(target) == "learning rate set to 0.5"
    assert context.process.optimizer.param_groups[0]["lr"] == 0.5
    assert not run(context, "lr -1").ok


def test_actions_run_immediately_once_training_has_ended(context, tmp_path):
    context.control.close()
    assert run(context, "lr 0.25").output == "learning rate set to 0.25"
    path = tmp_path / "weights.pt"
    assert "saved model weights" in run(context, f'save "{path}"').output
    assert set(torch.load(path)) == set(context.process.model.state_dict())


def test_inspection_commands(context):
    status = run(context, "status").output
    assert "SmallNet" in status and "epoch       1/5" in status and "accuracy" in status
    assert "0.9000" in run(context, "history").output
    assert "fc1" in run(context, "arch").output and "Linear 2→6" in run(context, "layers").output
    weights = run(context, "weights").output
    assert "fc1" in weights and "fc2" in weights
    assert "L2 norm" in run(context, "weights fc1").output
    assert not run(context, "weights nope").ok


def test_help_and_frontend_only_commands(context):
    listing = run(context, "help").output
    assert "TRAINING" in listing and "INSPECTION" in listing and "lr <value>" in listing
    assert "PgUp/PgDn scroll" in listing.splitlines()[-1]  # the terminal keys
    assert "usage: lr <value>" in run(context, "help lr").output
    assert "not available" in run(context, "screenshot").output
    assert run(context, "scale").output == "scale is not available in this frontend"
    assert run(context, "probe").output == "probe is not available in this frontend"
    assert run(context, "neuron").output == "neuron is not available in this frontend"
    assert run(context, "layout").output == "layout is not available in this frontend"
    assert run(context, "copy").output == "copy is not available in this frontend"
    assert run(context, "window").output == "window is not available in this frontend"


def test_custom_commands_from_the_process(make_process):
    process = make_process()

    @process.command("double", summary="double a number", arguments=[Argument("n", float)])
    def double(ctx, n):
        return f"{n * 2:g}"

    registry = create_registry(process.commands)
    assert registry.execute("double 2.5", None).output == "5"
    assert "CUSTOM" in registry.execute("help", type("Ctx", (), {"registry": registry})()).output


def test_public_api_exports_command_types():
    assert ht.Argument is Argument and ht.Command is Command


def test_run_changing_commands_are_refused_after_the_session_ended(context):
    context.control.close()
    for line in ("epochs 9", "step 2", "pause", "resume", "delay 10"):
        result = run(context, line)
        assert not result.ok and "has ended" in result.output, line
    assert run(context, "stop").output == "the training session has already ended"
    assert "fc1" in run(context, "weights").output  # inspection keeps working

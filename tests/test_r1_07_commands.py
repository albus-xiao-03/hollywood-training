"""R1 requirement 7 - "The package **MUST** provide a set of well-defined commands to control
the training process managed by the package during runtime."

*Well-defined*: every command is registered and documented; a command line is parsed
predictably (names without case, shell-like quoting); wrong input never acts - it is refused
with a clear message that names the usage - and changes nothing; boundaries are handled.
*Control the training during runtime*: every built-in command does what it says against a
real ``TrainingManager`` (pause at the next batch boundary, step exactly n epochs, stop,
change the total, the learning rate, the delay, inspect, save, ...), including after the run
ended. Custom commands follow the same rules and touch the model only between batches.
"""

from __future__ import annotations

import re
import threading
import time

import pytest
import torch
from htspec.events import SessionFeeder
from htspec.headless import FakeFrontend, HeadlessRun
from htspec.models import TinyNet, loader, make_setup

import hollywood_training as ht
from hollywood_training import formatting as fmt
from hollywood_training.commands.builtin import create_registry
from hollywood_training.commands.context import CommandContext
from hollywood_training.commands.registry import Command
from hollywood_training.sync.control import TrainingControl, TrainingTarget
from hollywood_training.sync.events import EpochCompleted, TrainingPaused, TrainingState

pytestmark = pytest.mark.req("R1.7")

BUILT_IN = {
    "help": ("?",), "status": (), "pause": (), "resume": ("continue", "start"), "step": (),
    "stop": (), "epochs": (), "lr": (), "delay": (), "history": (),
    "arch": ("layers", "architecture"), "weights": (), "scale": (), "probe": (), "neuron": (),
    "layout": (), "copy": (), "window": (), "save": (), "screenshot": (), "clear": ("cls",),
    "quit": ("exit",),
}  # fmt: skip
CATEGORIES = ("TRAINING", "INSPECTION", "VIEW", "SESSION", "GENERAL")


@pytest.fixture
def context():
    """Commands over a session with three epochs, a recording frontend, no training loop."""
    setup = make_setup(epochs=10)
    feeder = SessionFeeder(setup=setup)
    feeder.start()
    feeder.epochs(3, metrics={"accuracy": 0.5})
    frontend = FakeFrontend()
    control = TrainingControl(10)
    ctx = CommandContext(
        control=control,
        session=feeder.session,
        registry=create_registry(setup.process.commands),
        target=TrainingTarget(setup.model, setup.optimizer),
        frontend=frontend,
    )
    ctx.setup = setup  # type: ignore[attr-defined]
    ctx.recorded = frontend  # type: ignore[attr-defined]
    return ctx


def execute(ctx, line: str):
    return ctx.registry.execute(line, ctx)


def control_state(control: TrainingControl) -> tuple:
    return (control.paused, control.stop_requested, control.total_epochs, control.delay,
            control.stepping_until, control.revision)  # fmt: skip


# ------------------------------------------------------------- the command set
def test_every_built_in_command_and_alias_is_registered(context):
    registry = context.registry
    for name, aliases in BUILT_IN.items():
        command = registry.get(name)
        assert command is not None and command.name == name, name
        assert command.summary, name
        assert set(command.aliases) == set(aliases), name
        for alias in aliases:
            assert registry.get(alias) is command, alias


def test_help_lists_every_command_by_category(context):
    output = execute(context, "help").output
    positions = [output.index(f"{category}\n") for category in CATEGORIES]
    assert positions == sorted(positions)
    for name in BUILT_IN:
        command = context.registry.get(name)
        assert re.search(rf"^  {re.escape(command.usage)}\s+{re.escape(command.summary)}$",
                         output, re.M), name  # fmt: skip
    assert "type 'help <command>' for details" in output


def test_help_explains_one_command(context):
    output = execute(context, "help step").output
    assert output.splitlines()[0] == "usage: step [epochs]"
    assert "run more epochs, then pause" in output
    assert re.search(r"epochs\s+epochs to run before pausing \(default 1\)", output)
    assert "aliases: continue, start" in execute(context, "help resume").output
    assert execute(context, "help ?").output.startswith("usage: help [command]")


def test_help_for_an_unknown_command_is_refused(context):
    result = execute(context, "help xyzzy")
    assert result.ok is False and "unknown command 'xyzzy'" in result.output


@pytest.mark.parametrize(("line", "hint"), [("paus", "did you mean pause"),
                                            ("stauts", "did you mean status"),
                                            ("xyzzy", "type 'help'")])  # fmt: skip
def test_unknown_commands_are_refused_with_a_hint(context, line, hint):
    before = control_state(context.control)
    result = execute(context, line)
    assert result.ok is False and f"unknown command '{line}'" in result.output
    assert hint in result.output
    assert control_state(context.control) == before


@pytest.mark.parametrize("line", ["PAUSE", "Pause", "pAuSe"])
def test_command_names_ignore_case(context, line):
    result = execute(context, line)
    assert result.ok and result.command.name == "pause"
    assert context.control.paused


@pytest.mark.parametrize("line", ["", "   ", "\t"])
def test_empty_lines_are_refused(context, line):
    result = execute(context, line)
    assert result.ok is False and "empty command" in result.output


def test_arguments_are_quoted_like_in_a_shell(context, tmp_path):
    target = tmp_path / "with space" / "deep" / "model.pt"
    result = execute(context, f'save "{target}"')
    assert result.ok, result.output
    for action, _description in context.control.take_actions():  # what the training thread runs
        action(TrainingTarget(context.setup.model, context.setup.optimizer))
    assert target.exists()
    loaded = torch.load(target)
    assert set(loaded) == set(context.setup.model.state_dict())


def test_unbalanced_quotes_are_refused(context):
    result = execute(context, "save 'never closed")
    assert result.ok is False and "cannot parse" in result.output


@pytest.mark.parametrize(
    ("line", "usage"),
    [("lr", "usage: lr <value>"), ("epochs", "usage: epochs <total>"),
     ("save", "usage: save <path>"), ("pause now", "usage: pause"),
     ("step 1 2", "usage: step [epochs]"), ("history 1 2", "usage: history [count]")],
)  # fmt: skip
def test_missing_or_extra_arguments_are_refused_with_the_usage(context, line, usage):
    before = control_state(context.control)
    result = execute(context, line)
    assert result.ok is False and usage in result.output
    assert control_state(context.control) == before


INVALID = [
    "lr abc", "lr 0", "lr -1", "lr inf", "lr nan", "lr -inf", "epochs 0", "epochs -3",
    "epochs 2.5", "epochs abc", "step 0", "step -1", "step x", "delay -1", "delay inf",
    "delay nan", "delay x", "history 0", "history -2", "history x",
]  # fmt: skip


@pytest.mark.parametrize("line", INVALID)
def test_invalid_values_are_refused_and_change_nothing(context, line):
    before = control_state(context.control)
    lr = context.setup.optimizer.param_groups[0]["lr"]
    result = execute(context, line)
    assert result.ok is False and result.output.startswith("invalid ")
    assert control_state(context.control) == before
    assert context.control.take_actions() == []
    assert context.setup.optimizer.param_groups[0]["lr"] == lr


@pytest.mark.parametrize(
    "line", ["lr 1e-300", "lr 1e9", "delay 0", "delay 0.5", "epochs 3", "step 1", "history 1"]
)
def test_boundary_values_are_accepted(context, line):
    assert execute(context, line).ok


# --------------------------------------------------------------- custom commands
def plain_context(setup, registry) -> CommandContext:
    return CommandContext(
        control=TrainingControl(3),
        session=SessionFeeder(setup=setup).session,
        registry=registry,
        target=TrainingTarget(setup.model, setup.optimizer),
    )


def test_custom_commands_are_listed_and_convert_their_arguments():
    setup = make_setup()
    seen = []

    arguments = [ht.Argument("factor", float), ht.Argument("times", int, optional=True, default=1)]

    @setup.process.command("boost", summary="multiply something", arguments=arguments,
                           aliases=("bst",))  # fmt: skip
    def boost(_context, factor, times):
        seen.append((factor, times))
        return f"boost {factor} x{times}"

    registry = create_registry(setup.process.commands)
    ctx = plain_context(setup, registry)
    listing = registry.execute("help", ctx).output
    assert "CUSTOM" in listing and "boost <factor> [times]" in listing
    assert registry.execute("boost 1.5", ctx).output == "boost 1.5 x1"
    assert registry.execute("bst 2 3", ctx).output == "boost 2.0 x3"
    assert registry.execute("boost x", ctx).ok is False
    assert seen == [(1.5, 1), (2.0, 3)]


def test_a_failing_custom_command_is_reported_not_raised():
    setup = make_setup()

    @setup.process.command("explode", summary="raises")
    def explode(_context):
        raise RuntimeError("kaboom")

    registry = create_registry(setup.process.commands)
    ctx = plain_context(setup, registry)
    result = registry.execute("explode", ctx)
    assert result.ok is False and result.output == "RuntimeError: kaboom"


@pytest.mark.parametrize("name", ["Upper", "two words", "", "MiXeD"])
def test_custom_command_names_must_be_one_lowercase_word(name):
    process = make_setup().process
    with pytest.raises(ValueError):
        process.command(name, summary="bad")(lambda ctx: None)


def test_a_required_argument_after_an_optional_one_is_refused():
    with pytest.raises(ValueError):
        Command("odd", "odd", lambda ctx, a, b: None,
                arguments=(ht.Argument("a", optional=True), ht.Argument("b")))  # fmt: skip


@pytest.mark.parametrize("name", ["pause", "status", "?", "exit"])
def test_a_custom_command_may_not_take_a_built_in_name(qapp, name):
    setup = make_setup(epochs=1)
    setup.process.command(name, summary="clash")(lambda ctx: "mine")
    with pytest.raises(ValueError):
        create_registry(setup.process.commands)
    with pytest.raises(ValueError):  # refused before a window opens
        ht.run(setup.process, isolate_gui=False, close_on_finish=True)


def test_custom_commands_touch_the_model_only_between_batches():
    setup = make_setup(epochs=30, native_steps=False)
    state = {"inside": False, "thread": None}
    model, optimizer, loss_fn = setup.model, setup.optimizer, setup.loss_fn

    @setup.process.train_step
    def train(batch):
        state["inside"] = True
        state["thread"] = threading.get_ident()
        time.sleep(0.005)
        optimizer.zero_grad()
        loss = loss_fn(model(batch[0]), batch[1])
        loss.backward()
        optimizer.step()
        state["inside"] = False
        return loss

    observed = []

    @setup.process.command("peek", summary="look at the model between batches")
    def peek(context):
        def action(target):
            observed.append((state["inside"], threading.get_ident(), target.model is model))
            return "peeked"

        return context.submit(action, "peek")

    run = HeadlessRun(setup.process).start()
    run.wait_events(EpochCompleted, 1)
    for _ in range(5):
        assert run.command("peek").ok
        time.sleep(0.02)
    run.wait_for(lambda r: len(observed) == 5)
    run.stop()
    assert observed == [(False, state["thread"], True)] * 5


# ---------------------------------------------------------- runtime semantics
def slow_setup(epochs: int, batch_sleep: float = 0.01, **kwargs):
    counter = {"steps": 0}

    def hook(_batch):
        counter["steps"] += 1
        time.sleep(batch_sleep)

    setup = make_setup(epochs=epochs, train_hook=hook, **kwargs)
    setup.counter = counter  # type: ignore[attr-defined]
    return setup


def test_pause_takes_effect_at_the_next_batch_boundary_and_resume_continues():
    setup = slow_setup(50)
    run = HeadlessRun(setup.process).start()
    run.wait_events(EpochCompleted, 1)
    before = setup.counter["steps"]
    assert run.command("pause").output == "pausing after the current batch"
    run.wait_events(TrainingPaused, 1)
    steps = setup.counter["steps"]
    assert steps <= before + 1  # at most the batch in progress was finished
    timer = run.session.elapsed_now()
    time.sleep(0.4)
    assert setup.counter["steps"] == steps  # no training while paused
    assert run.session.state is TrainingState.PAUSED
    assert run.session.elapsed_now() == pytest.approx(timer)  # the process timer stands still
    assert run.command("pause").output == "already paused"
    assert run.command("resume").output == "resuming"
    run.wait_for(lambda r: setup.counter["steps"] > steps + 2)
    assert run.command("resume").output == "not paused"
    run.stop()


def test_the_process_timer_leaves_out_the_pause():
    setup = slow_setup(50)
    run = HeadlessRun(setup.process).start()
    run.wait_events(EpochCompleted, 1)
    run.command("pause")
    run.wait_events(TrainingPaused, 1)
    started = time.perf_counter()
    time.sleep(0.8)
    run.command("resume")
    result = run.stop()
    wall = time.perf_counter() - started
    assert wall >= 0.8
    epochs_time = sum(record.duration for record in result.history)
    assert result.elapsed < epochs_time + 0.6  # the 0.8 s pause is not counted


def test_step_runs_exactly_the_requested_epochs_then_pauses():
    setup = slow_setup(20, batch_sleep=0.002)
    run = HeadlessRun(setup.process, start_paused=True).start()
    run.wait_events(TrainingPaused, 1)
    assert run.control.completed_epochs == 0
    assert run.command("step 3").output == "running 3 epochs, then pausing"
    run.wait_events(TrainingPaused, 2)
    assert run.control.completed_epochs == 3
    assert run.command("step").output == "running 1 epoch, then pausing"
    run.wait_events(TrainingPaused, 3)
    assert run.control.completed_epochs == 4
    assert len(run.events(EpochCompleted)) == 4
    run.stop()


def test_stop_ends_the_run_with_the_completed_epochs_only():
    setup = slow_setup(100)
    run = HeadlessRun(setup.process).start()
    run.wait_events(EpochCompleted, 2)
    assert run.command("stop").output == "stopping after the current batch"
    result = run.join()
    assert result.state is TrainingState.STOPPED
    assert result.epochs_completed == len(run.events(EpochCompleted)) < 100
    assert run.command("stop").output == "the training session has already ended"


def test_epochs_changes_the_total_of_a_running_session():
    setup = slow_setup(50, batch_sleep=0.003)
    run = HeadlessRun(setup.process).start()
    run.wait_events(EpochCompleted, 1)
    target = run.control.completed_epochs + 2
    assert run.command(f"epochs {target}").output == f"total epochs set to {target}"
    result = run.join()
    assert result.state is TrainingState.FINISHED and result.epochs_completed == target


def test_epochs_below_the_completed_count_is_refused():
    setup = slow_setup(50, batch_sleep=0.003)
    run = HeadlessRun(setup.process).start()
    run.wait_events(EpochCompleted, 3)
    result = run.command("epochs 1")
    assert result.ok is False and "at least" in result.output
    run.stop()


def test_epochs_continues_a_finished_run():
    setup = make_setup(epochs=2)
    run = HeadlessRun(setup.process, linger=True).start()
    run.wait_for(lambda r: r.session.state is TrainingState.FINISHED)
    assert run.command("step").ok is False  # nothing left to step through
    assert run.command("epochs 5").output == "continuing training to epoch 5"
    run.wait_for(lambda r: r.session.state is TrainingState.FINISHED and r.session.epoch == 5)
    result = run.stop()
    assert result.epochs_completed == 5


def test_lr_applies_between_batches_to_every_parameter_group():
    torch.manual_seed(0)
    model = TinyNet()
    optimizer = torch.optim.SGD(
        [{"params": model.fc1.parameters()}, {"params": model.fc2.parameters(), "lr": 0.2}],
        lr=0.05,
    )
    process = ht.TrainingProcess(model, optimizer, loader(64, 8), epochs=40)
    in_step_changes = []

    @process.train_step
    def train(batch):
        lr_before = [group["lr"] for group in optimizer.param_groups]
        time.sleep(0.003)
        optimizer.zero_grad()
        loss = torch.nn.functional.cross_entropy(model(batch[0]), batch[1])
        loss.backward()
        optimizer.step()
        if [group["lr"] for group in optimizer.param_groups] != lr_before:
            in_step_changes.append(lr_before)
        return loss

    run = HeadlessRun(process).start()
    run.wait_events(EpochCompleted, 1)
    result = run.command("lr 0.123")
    assert result.ok and "0.123" in result.output
    run.wait_for(lambda r: all(g["lr"] == 0.123 for g in optimizer.param_groups))
    seen = len(run.events(EpochCompleted))
    run.wait_events(EpochCompleted, seen + 2)
    final = run.stop()
    assert in_step_changes == []  # never in the middle of an optimisation step
    assert final.history[-1].learning_rate == pytest.approx(0.123)


def test_lr_after_the_session_ended_applies_at_once():
    setup = make_setup(epochs=1)
    run = HeadlessRun(setup.process)
    run.run()
    result = run.command("lr 0.5")
    assert result.ok and result.output == "learning rate set to 0.5"
    assert [group["lr"] for group in setup.optimizer.param_groups] == [0.5]


def test_status_reports_the_session(context):
    output = execute(context, "status").output
    session = context.session
    last = session.history[-1]
    best = session.best_record()
    assert "network     TinyNet" in output
    assert "state       running" in output
    assert "epoch       3/10" in output
    assert f"train loss  {fmt.loss(last.train_loss)}" in output
    assert f"val loss    {fmt.loss(last.val_loss)}" in output
    assert f"(epoch {best.epoch})" in output
    assert "lr          0.05" in output
    assert re.search(r"timer\s+\d\d:\d\d:\d\d\.\d{3}", output)
    assert "delay       0 ms" in output
    assert re.search(r"accuracy\s+50\.0%", output)


def test_history_lists_the_last_epochs(context):
    lines = execute(context, "history 2").output.splitlines()
    assert lines[0].split()[:5] == ["epoch", "train", "val", "lr", "time"]
    assert [int(line.split()[0]) for line in lines[1:]] == [2, 3]
    assert len(execute(context, "history").output.splitlines()) == 1 + 3  # all three of 10
    empty = SessionFeeder(setup=make_setup())
    empty.start()
    ctx = CommandContext(
        control=TrainingControl(3),
        session=empty.session,
        registry=create_registry(),
        target=TrainingTarget(None, None),
    )
    assert create_registry().execute("history", ctx).output == "no completed epochs yet"


def test_arch_describes_the_tracked_network(context):
    output = execute(context, "arch").output
    assert output.splitlines()[0].startswith("TinyNet - 51 parameters (51 trainable)")
    assert re.search(r"fc1\s+Linear 2→8\s+Tanh", output)
    assert re.search(r"fc2\s+Linear 8→3", output)
    assert "normalisation: none" in output


def test_weights_reports_the_real_statistics(context):
    overview = execute(context, "weights").output
    assert overview.startswith("weights after epoch 3")
    assert re.search(r"^  fc1\s+mean [+-]\d\.\d{4}", overview, re.M)
    detail = execute(context, "weights fc1").output
    weight = context.setup.model.fc1.weight.detach()
    mean, std = float(weight.mean()), float(weight.std(correction=0))
    assert f"mean / std  {mean:+.5f} / {std:.5f}" in detail
    assert f"min / max   {float(weight.min()):+.5f} / {float(weight.max()):+.5f}" in detail
    assert f"L2 norm     {float(torch.linalg.vector_norm(weight)):.5f}" in detail
    unknown = execute(context, "weights fc9")
    assert unknown.ok is False and "layers: fc1, fc2" in unknown.output


def test_save_writes_a_loadable_state_dict_between_batches(tmp_path):
    setup = slow_setup(50, batch_sleep=0.003)
    target = tmp_path / "nested" / "dir" / "model.pt"
    run = HeadlessRun(setup.process).start()
    run.wait_events(EpochCompleted, 1)
    assert run.command(f"save {target}").ok
    run.wait_for(lambda r: target.exists())
    run.stop()
    fresh = TinyNet()
    fresh.load_state_dict(torch.load(target))  # a complete state_dict of the model


def test_screenshot_clear_and_quit_ask_the_frontend(context, tmp_path):
    shot = tmp_path / "window.png"
    assert execute(context, f"screenshot {shot}").ok
    assert execute(context, "clear").ok
    assert execute(context, "quit").output == "stopping and closing"
    calls = context.recorded.calls
    assert calls[0][0] == "screenshot" and str(calls[0][1]) == str(shot)
    assert calls[1:] == [("clear",), ("close",)]
    assert context.control.stop_requested


def test_after_the_session_ended_run_changing_commands_are_refused(tmp_path):
    setup = make_setup(epochs=1)
    run = HeadlessRun(setup.process)
    run.run()
    for line in ("pause", "resume", "step", "epochs 5", "delay 10"):
        result = run.command(line)
        assert result.ok is False and "the training session has ended" in result.output, line
    for line in ("status", "history", "arch", "weights", "help", "lr 0.2", f"save {tmp_path}/m.pt"):
        assert run.command(line).ok, line
    assert (tmp_path / "m.pt").exists()  # written at once: no training thread any more

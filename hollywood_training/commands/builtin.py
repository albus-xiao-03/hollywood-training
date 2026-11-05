"""The built-in command set for controlling and inspecting a managed training session.

======================  ======================================================
``help [command]``      list commands, or details of one (alias: ``?``)
``status``              state, epoch, losses, learning rate, timer, ETA
``pause``               pause at the next batch boundary
``resume``              continue (aliases: ``continue``, ``start``)
``step [epochs]``       run more epochs (default 1), then pause
``stop``                end training at the next batch boundary
``epochs <total>``      change the total number of epochs (also continues a finished run)
``lr <value>``          set the learning rate of every parameter group
``delay <ms>``          wait this long after every epoch (0 = full speed)
``history [count]``     the last epochs (default 10)
``arch``                the tracked architecture (aliases: ``layers``, ``architecture``)
``weights [layer]``     weight statistics of every layer, or one layer in detail
``save <path>``         save the model's ``state_dict`` with ``torch.save``
``scale [lin|log]``     loss chart axis: linear or logarithmic (switches when omitted)
``probe [epoch|off]``   an epoch's values, marked on the loss chart (default: latest)
``neuron [col] [n]``    highlight a drawn neuron (``neuron off``; no args: columns)
``screenshot [path]``   save a PNG of the window (GUI only)
``clear``               clear the terminal (alias: ``cls``)
``quit``                stop training and close the window (alias: ``exit``)
======================  ======================================================

Commands that change the run are refused once the training session has ended
(``TrainingControl.closed``); inspection, ``lr`` and ``save`` keep working.

The ``view`` commands (``scale``, ``probe``, ``neuron``, ...) only change the window: they
replace the mouse interactions of the first version (SPEC R2) and reach the window as a
:class:`~hollywood_training.commands.view.ViewRequest`.
"""

from __future__ import annotations

import math
import time
from collections.abc import Iterable
from pathlib import Path

import torch

from .. import formatting as fmt
from ..sync.control import TrainingTarget
from ..sync.events import TrainingState
from .context import CommandContext
from .registry import Argument, Command, CommandError, CommandRegistry
from .view import ViewRequest

CATEGORY_ORDER = ("training", "inspection", "view", "session", "custom", "general")
KEYS = "keys: Tab completes · ↑/↓ history · PgUp/PgDn scroll · Ctrl+Home/End top/bottom"


def _positive_int(text: str) -> int:
    value = int(text)
    if value < 1:
        raise ValueError("must be a positive integer")
    return value


def _non_negative_float(text: str) -> float:
    value = float(text)
    if not math.isfinite(value) or value < 0:
        raise ValueError("must be a non-negative number")
    return value


def _positive_float(text: str) -> float:
    value = float(text)
    if not math.isfinite(value) or value <= 0:
        raise ValueError("must be a positive number")
    return value


def _non_negative_int(text: str) -> int:
    value = int(text)
    if value < 0:
        raise ValueError("must be 0 or more")
    return value


def _epoch_or_off(text: str) -> int | str:
    if text == "off":
        return text
    try:
        value = int(text)
    except ValueError:
        value = 0
    if value < 1:
        raise ValueError("must be an epoch number (1 or more) or off")
    return value


def _queued(message: str | None, description: str) -> str:
    return message if message is not None else f"{description} - applied at the next batch boundary"


def _require_session(ctx: CommandContext) -> None:
    """Refuse run-changing commands once the training thread has exited."""
    if ctx.control.closed:
        raise CommandError("the training session has ended")


# ------------------------------------------------------------------ training
def _pause(ctx: CommandContext) -> str:
    _require_session(ctx)
    state = ctx.session.state
    if not state.active:
        raise CommandError(f"training is {state.value}, nothing to pause")
    if ctx.control.paused or state is TrainingState.PAUSED:
        return "already paused"
    ctx.control.pause()
    return "pausing after the current batch"


def _resume(ctx: CommandContext) -> str:
    _require_session(ctx)
    if not ctx.control.paused and ctx.session.state is not TrainingState.PAUSED:
        return "not paused"
    ctx.control.resume()
    return "resuming"


def _step(ctx: CommandContext, epochs: int) -> str:
    _require_session(ctx)
    if not ctx.session.state.active:
        raise CommandError(f"training is {ctx.session.state.value}; use 'epochs <n>' to extend it")
    ctx.control.step(epochs)
    return f"running {epochs} epoch{'s' if epochs != 1 else ''}, then pausing"


def _stop(ctx: CommandContext) -> str:
    state = ctx.session.state
    if ctx.control.closed:
        return "the training session has already ended"
    if not state.active and state is not TrainingState.FINISHED:
        return f"training already {state.value}"
    ctx.control.stop()
    return "stopping after the current batch"


def _epochs(ctx: CommandContext, total: int) -> str:
    _require_session(ctx)
    if ctx.session.state in (TrainingState.STOPPED, TrainingState.FAILED):
        raise CommandError(f"training has {ctx.session.state.value}; it cannot be extended")
    try:
        ctx.control.set_total_epochs(total)
    except ValueError as exc:
        raise CommandError(str(exc)) from None
    ctx.session.total_epochs = total
    if ctx.session.state is TrainingState.FINISHED and total > ctx.session.epoch:
        return f"continuing training to epoch {total}"
    return f"total epochs set to {total}"


def _set_learning_rate(value: float):
    def apply(target: TrainingTarget) -> str:
        for group in target.optimizer.param_groups:
            if isinstance(group["lr"], torch.Tensor):
                group["lr"].fill_(value)
            else:
                group["lr"] = value
        return f"learning rate set to {value:g}"

    return apply


def _lr(ctx: CommandContext, value: float) -> str:
    message = ctx.submit(_set_learning_rate(value), f"set learning rate to {value:g}")
    return _queued(message, f"learning rate {value:g}")


def _delay(ctx: CommandContext, milliseconds: float) -> str:
    _require_session(ctx)
    ctx.control.set_delay(milliseconds / 1000.0)
    ctx.session.delay = milliseconds / 1000.0
    if milliseconds == 0:
        return "delay off - running as fast as synchronisation allows"
    return f"waiting {milliseconds:g} ms after every epoch"


# ---------------------------------------------------------------- inspection
def _status(ctx: CommandContext) -> str:
    s = ctx.session
    last = s.last_record
    best = s.best_record()
    lines = [
        f"network     {s.network_name or fmt.PLACEHOLDER}",
        f"state       {s.state.value}",
        f"epoch       {s.epoch}/{s.total_epochs}",
        f"train loss  {fmt.loss(last.train_loss if last else None)}",
        f"val loss    {fmt.loss(last.val_loss if last else None)}",
        f"best        {fmt.loss(_best_value(best))}" + (f" (epoch {best.epoch})" if best else ""),
        f"lr          {fmt.learning_rate(s.learning_rate)}",
        f"timer       {fmt.duration(s.elapsed_now())}",
        f"eta         {fmt.duration(s.eta(), millis=False)}",
        f"delay       {s.delay * 1000:g} ms",
    ]
    if last is not None:
        for name, value in {**last.train_metrics, **_prefixed(last.val_metrics)}.items():
            lines.append(f"{name:<11} {fmt.metric(name, value)}")
    return "\n".join(lines)


def _best_value(record) -> float | None:
    if record is None:
        return None
    return record.monitored_loss


def _prefixed(metrics: dict[str, float]) -> dict[str, float]:
    return {f"val {name}": value for name, value in metrics.items()}


def _history(ctx: CommandContext, count: int) -> str:
    records = ctx.session.history[-count:]
    if not records:
        return "no completed epochs yet"
    metric_names = list(records[-1].train_metrics)
    header = f"{'epoch':>6}  {'train':>10}  {'val':>10}  {'lr':>10}  {'time':>8}"
    header += "".join(f"  {name:>9}" for name in metric_names)
    lines = [header]
    for r in records:
        row = (
            f"{r.epoch:>6}  {fmt.loss(r.train_loss):>10}  {fmt.loss(r.val_loss):>10}  "
            f"{fmt.learning_rate(r.learning_rate):>10}  {r.duration:>7.3f}s"
        )
        row += "".join(
            f"  {fmt.metric(n, r.train_metrics[n]) if n in r.train_metrics else '':>9}"
            for n in metric_names
        )
        lines.append(row)
    return "\n".join(lines)


def _architecture(ctx: CommandContext) -> str:
    architecture = ctx.session.architecture
    if architecture is None:
        return "no network yet"
    title = architecture.class_name
    if architecture.display_name:
        title += f" ({architecture.display_name})"
    lines = [
        f"{title} - {fmt.count(architecture.total_parameters)} parameters "
        f"({fmt.count(architecture.trainable_parameters)} trainable)"
    ]
    if architecture.input_operations:
        inputs = ", ".join(op.summary for op in architecture.input_operations)
        lines.append(f"  input       {inputs}")
    width = max((len(layer.name) for layer in architecture.layers), default=5)
    for layer in architecture.layers:
        ops = ", ".join(op.summary for op in layer.operations)
        lines.append(f"  {layer.name:<{width}}  {layer.summary:<26} {ops}".rstrip())
    if architecture.unsupported:
        lines.append("  not drawn: " + ", ".join(architecture.unsupported))
    lines.append(f"normalisation: {architecture.normalization_summary()}")
    return "\n".join(lines)


def _weights(ctx: CommandContext, name: str | None) -> str:
    snapshot = ctx.session.weights
    if snapshot is None or not snapshot.layers:
        return "no weights captured yet"
    if name is not None:
        layer = snapshot.layer(name)
        if layer is None:
            raise CommandError(f"no layer {name!r}; layers: {', '.join(snapshot.names())}")
        st = layer.stats
        shown = f"{len(layer.out_index)}x{len(layer.in_index)}"
        lines = [
            f"{layer.name} after epoch {snapshot.epoch}",
            f"  shape       {layer.out_features} out x {layer.in_features} in (drawn {shown})",
            f"  elements    {fmt.count(st.count)}",
            f"  mean / std  {st.mean:+.5f} / {st.std:.5f}",
            f"  min / max   {st.minimum:+.5f} / {st.maximum:+.5f}",
            f"  mean |w|    {st.abs_mean:.5f}",
            f"  L2 norm     {st.norm:.5f}",
        ]
        if layer.update_ratio is not None:
            lines.append(f"  last update {fmt.percent(layer.update_ratio, 3)} of the drawn weights")
        if layer.bias is not None:
            lines.append(f"  bias mean   {float(layer.bias.mean()):+.5f} (drawn neurons)")
        return "\n".join(lines)
    width = max(len(layer.name) for layer in snapshot.layers)
    lines = [f"weights after epoch {snapshot.epoch}"]
    for layer in snapshot.layers:
        st = layer.stats
        update = fmt.percent(layer.update_ratio, 2) if layer.update_ratio is not None else "—"
        lines.append(
            f"  {layer.name:<{width}}  mean {st.mean:+.4f}  std {st.std:.4f}  "
            f"|w| {st.abs_mean:.4f}  Δ {update}"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------- view
def _scale(ctx: CommandContext, mode: str | None) -> str:
    return ctx.show(ViewRequest("scale", (mode,)))


def _probe(ctx: CommandContext, epoch: int | str | None) -> str:
    if epoch == "off":
        return ctx.show(ViewRequest("probe", (None,)))
    history = ctx.session.history
    if not history:
        raise CommandError("no completed epochs yet")
    last = history[-1].epoch
    wanted = last if epoch is None else epoch
    record = next((r for r in history if r.epoch == wanted), None)
    if record is None:
        raise CommandError(f"epoch {wanted} has not completed yet (last completed: {last})")
    ctx.show(ViewRequest("probe", (wanted,)))  # the window draws the probe line
    report = _probe_report(record, ctx.session)
    if ctx.session.state is TrainingState.RUNNING:
        report += (
            "\ntraining is running: the probe line clears at the next epoch ('pause' keeps it)"
        )
    return report


def _neuron(ctx: CommandContext, column: str | None, index: int | None) -> str:
    return ctx.show(ViewRequest("neuron", (column, index)))


def _probe_report(record, session) -> str:
    best = session.best_record()
    title = f"epoch {record.epoch} of {session.total_epochs}"
    if best is not None and best.epoch == record.epoch:
        title += " (best)"
    lines = [
        title,
        f"  train loss  {fmt.loss(record.train_loss)}",
        f"  val loss    {fmt.loss(record.val_loss)}",
        f"  lr          {fmt.learning_rate(record.learning_rate)}",
        f"  time        {record.duration:.3f} s",
    ]
    for name, value in {**record.train_metrics, **_prefixed(record.val_metrics)}.items():
        lines.append(f"  {name:<11} {fmt.metric(name, value)}")
    return "\n".join(lines)


# ------------------------------------------------------------------- session
def _save(ctx: CommandContext, path: str) -> str:
    destination = Path(path).expanduser()

    def save(target: TrainingTarget) -> str:
        destination.parent.mkdir(parents=True, exist_ok=True)
        torch.save(target.model.state_dict(), destination)
        return f"saved model weights to {destination.resolve()}"

    return _queued(ctx.submit(save, f"save weights to {destination}"), "saving")


def _screenshot(ctx: CommandContext, path: str | None) -> str:
    frontend = ctx.require_frontend("screenshot")
    name = path or time.strftime("hollywood-training-%Y%m%d-%H%M%S.png")
    saved = frontend.screenshot(Path(name).expanduser())
    return f"screenshot saved to {saved}" if saved is not None else f"screenshot requested: {name}"


def _clear(ctx: CommandContext) -> str:
    ctx.require_frontend("clear").clear_output()
    return ""


def _quit(ctx: CommandContext) -> str:
    ctx.control.stop()
    if ctx.frontend is not None:
        ctx.frontend.close()
    return "stopping and closing"


def _help(ctx: CommandContext, name: str | None) -> str:
    registry = ctx.registry
    if name is not None:
        command = registry.get(name)
        if command is None:
            raise CommandError(f"unknown command {name!r}")
        lines = [f"usage: {command.usage}", f"  {command.summary}"]
        if command.details:
            lines += [f"  {line}" for line in command.details.splitlines()]
        if command.arguments:
            lines.append("arguments:")
            for argument in command.arguments:
                default = ""
                if argument.optional and argument.default is not None:
                    default = f" (default {argument.default})"
                lines.append(f"  {argument.name:<10} {argument.help}{default}")
        if command.aliases:
            lines.append(f"aliases: {', '.join(command.aliases)}")
        return "\n".join(lines)

    commands = list(registry)
    width = max(len(command.usage) for command in commands)
    categories = [c for c in CATEGORY_ORDER if any(cmd.category == c for cmd in commands)]
    categories += sorted({cmd.category for cmd in commands} - set(categories))
    lines: list[str] = []
    for category in categories:
        lines.append(category.upper())
        lines += [
            f"  {command.usage:<{width}}  {command.summary}"
            for command in commands
            if command.category == category
        ]
    lines.append("type 'help <command>' for details")
    lines.append(KEYS)
    return "\n".join(lines)


def builtin_commands() -> list[Command]:
    """A fresh list of the built-in commands."""
    epochs_arg = Argument("epochs", _positive_int, "epochs to run before pausing", True, 1)
    return [
        Command("pause", "pause at the next batch boundary", _pause, category="training"),
        Command(
            "resume", "continue a paused run", _resume,
            aliases=("continue", "start"), category="training",
        ),
        Command(
            "step", "run more epochs, then pause", _step,
            arguments=(epochs_arg,), category="training",
        ),
        Command("stop", "end training at the next batch boundary", _stop, category="training"),
        Command(
            "epochs", "change the total number of epochs", _epochs,
            arguments=(Argument("total", _positive_int, "new total number of epochs"),),
            category="training",
            details="Raising it after the run finished continues training.",
        ),
        Command(
            "lr", "set the learning rate of every parameter group", _lr,
            arguments=(Argument("value", _positive_float, "new learning rate"),),
            category="training",
            details="A scheduler may change it again at the end of the epoch.",
        ),
        Command(
            "delay", "wait after every epoch (0 = full speed)", _delay,
            arguments=(Argument("ms", _non_negative_float, "milliseconds per epoch"),),
            category="training",
        ),
        Command("status", "show the current training status", _status, category="inspection"),
        Command(
            "history", "show the last epochs", _history,
            arguments=(Argument("count", _positive_int, "number of epochs", True, 10),),
            category="inspection",
        ),
        Command(
            "arch", "show the tracked architecture", _architecture,
            aliases=("layers", "architecture"), category="inspection",
        ),
        Command(
            "weights", "show weight statistics", _weights,
            arguments=(Argument("layer", str, "layer name for details", True),),
            category="inspection",
        ),
        Command(
            "scale", "switch the loss axis between linear and logarithmic", _scale,
            arguments=(
                Argument(
                    "mode", help="lin or log; switches when omitted", optional=True,
                    choices=("lin", "log"),
                ),
            ),
            category="view",
        ),
        Command(
            "probe", "show an epoch's values and mark it on the loss chart", _probe,
            arguments=(
                Argument("epoch", _epoch_or_off, "epoch number or off (default: latest)", True),
            ),
            category="view",
            details="The loss chart draws a vertical line at the epoch, with its loss values.\n"
            "It is removed when training continues (next epoch, resume) or by 'probe off'.",
        ),
        Command(
            "neuron", "highlight a neuron on the network view", _neuron,
            arguments=(
                Argument("column", str, "column title as drawn, or off", True),
                Argument("index", _non_negative_int, "neuron index", True),
            ),
            category="view",
            details="Without arguments: the drawn columns. With a column only: its drawn\n"
            "neurons. 'neuron off' removes the highlight.",
        ),
        Command(
            "save", "save the model's state_dict", _save,
            arguments=(Argument("path", str, "file to write (torch.save)"),),
            category="session",
        ),
        Command(
            "screenshot", "save a PNG of the window", _screenshot,
            arguments=(Argument("path", str, "PNG file to write", True),),
            category="session",
        ),
        Command("clear", "clear the terminal", _clear, aliases=("cls",), category="session"),
        Command(
            "quit", "stop training and close the window", _quit,
            aliases=("exit",), category="session",
        ),
        Command(
            "help", "list commands or explain one", _help,
            arguments=(Argument("command", str, "command to explain", True),),
            aliases=("?",), category="general",
        ),
    ]  # fmt: skip


def create_registry(extra: Iterable[Command] = ()) -> CommandRegistry:
    """Registry with the built-in commands plus ``extra`` (e.g. ``process.commands``)."""
    registry = CommandRegistry(builtin_commands())
    for command in extra:
        registry.register(command)
    return registry

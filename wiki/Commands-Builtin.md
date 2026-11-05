# Built-in Commands

`hollywood_training/commands/builtin.py` · part of [Commands](Commands)

The handlers behind `pause`, `status`, `lr` and the other built-in commands, and the
factory that combines them with a process's custom commands into the session's registry.
Handlers validate their arguments and the session state, then either request a change
from [`TrainingControl`](Sync-Control) or queue an action for the training thread. For
the user-facing reference (usage, output, examples) see [Commands](Commands).

## Responsibilities

* Declare the 16 built-in commands with category, arguments, aliases and help texts.
* Reject invalid values (argument converters) and impossible requests (`CommandError`).
* Refuse the commands that change the run once the training session has ended.
* Route model and optimizer access (`lr`, `save`) through `CommandContext.submit`.
* Build the registry for a session (`create_registry`).

## API

```python
CATEGORY_ORDER = ("training", "inspection", "view", "session", "custom", "general")
KEYS = "keys: Tab completes · ↑/↓ history · PgUp/PgDn scroll · Ctrl+Home/End top/bottom"

def builtin_commands() -> list[Command]: ...
def create_registry(extra: Iterable[Command] = ()) -> CommandRegistry: ...
```

* `CATEGORY_ORDER`: order of the groups in the `help` listing; other categories follow
  alphabetically.
* `KEYS`: the last line of the `help` listing, the keys of the window's
  [terminal](GUI-Terminal).
* `builtin_commands()` returns a fresh list of new `Command` objects on every call.
* `create_registry(extra)` returns `CommandRegistry(builtin_commands())` with every
  command of `extra` (e.g. `process.commands`) registered after it, without `replace`.
  A clashing name or alias raises `ValueError`.

## How it works

### Argument converters

| Converter | Accepts | Message after `invalid <arg> '<text>': ` | Used by |
|---|---|---|---|
| `_positive_int` | `int(text) >= 1` | `must be a positive integer`, or `int()`'s own message | `step`, `epochs`, `history`, `copy` |
| `_non_negative_float` | finite `float(text) >= 0` | `must be a non-negative number` | `delay` |
| `_positive_float` | finite `float(text) > 0` | `must be a positive number` | `lr` |
| `_non_negative_int` | `int(text) >= 0` | `must be 0 or more`, or `int()`'s own message | `neuron` |
| `_percent` | `int(text without a trailing %)` from 1 to 99 | `must be a whole number from 1 to 99`, or `int()`'s own message | `layout` |
| `_epoch_or_off` | `"off"`, or `int(text) >= 1` | `must be an epoch number (1 or more) or off` | `probe` |

### Handlers

"Handler thread" is wherever the frontend runs handlers: the command thread of the
training process for the window process, the GUI (main) thread for the in-process window
([Commands Context](Commands-Context)).

| Command | Handler | Checks | Effect | Runs on |
|---|---|---|---|---|
| `pause` | `_pause` | `_require_session`; `session.state.active`, else `CommandError`; already paused → message | `control.pause()` | handler thread |
| `resume` | `_resume` | `_require_session`; paused in the control or the session, else `not paused` | `control.resume()` | handler thread |
| `step` | `_step` | `_require_session`; `session.state.active`, else `CommandError` | `control.step(epochs)` | handler thread |
| `stop` | `_stop` | `control.closed` → `the training session has already ended`; `STOPPED` / `FAILED` → `training already …` | `control.stop()` | handler thread |
| `epochs` | `_epochs` | `_require_session`; not `STOPPED` / `FAILED`; a `ValueError` from the control becomes `CommandError` | `control.set_total_epochs(total)`; `session.total_epochs = total` | handler thread |
| `lr` | `_lr` | converter only | `submit(_set_learning_rate(value), "set learning rate to <value>")` | **training thread** |
| `delay` | `_delay` | `_require_session`; converter | `control.set_delay(ms / 1000)`; `session.delay = ms / 1000` | handler thread |
| `status` | `_status` | | formats `session` | handler thread |
| `history` | `_history` | | formats `session.history[-count:]` | handler thread |
| `arch` | `_architecture` | | formats `session.architecture` | handler thread |
| `weights` | `_weights` | unknown layer → `CommandError` | formats `session.weights` | handler thread |
| `probe` | `_probe` | no history → `no completed epochs yet`; an epoch not in `session.history` → `epoch <n> has not completed yet (last completed: <last>)` | `ctx.show(ViewRequest("probe", (epoch,)))`, then formats the record (`_probe_report`), plus a note while the session is `RUNNING` that the line clears at the next epoch; `off` → `ctx.show(ViewRequest("probe", (None,)))` | handler thread, then the window |
| `copy` | `_copy` | converter only | `ctx.show(ViewRequest("copy", (lines,)))` | handler thread, then the window |
| `window` | `_window` | `mode` choices `normal`, `max`, `full`, `min` | `ctx.show(ViewRequest("window", (mode,)))` | handler thread, then the window |
| `layout` | `_layout` | converter only; the window checks the panel name | `ctx.show(ViewRequest("layout", (panel, percent)))` | handler thread, then the window |
| `neuron` | `_neuron` | converters only; the window checks the column and the index | `ctx.show(ViewRequest("neuron", (column, index)))` | handler thread, then the window |
| `scale` | `_scale` | `mode` choices `lin`, `log` | `ctx.show(ViewRequest("scale", (mode,)))` | handler thread, then the window |
| `save` | `_save` | | `submit(save, "save weights to <path>")` | **training thread** |
| `screenshot` | `_screenshot` | `require_frontend("screenshot")` | `frontend.screenshot(path)` | handler thread, then the window |
| `clear` | `_clear` | `require_frontend("clear")` | `frontend.clear_output()` | handler thread |
| `quit` | `_quit` | | `control.stop()`; `frontend.close()` when there is a frontend | handler thread |
| `help` | `_help` | unknown name → `CommandError` | formats `context.registry` | handler thread |

### Session check

`_require_session(ctx)` raises `CommandError("the training session has ended")` when
`ctx.control.closed`, i.e. once the training thread has exited. `pause`, `resume`,
`step`, `epochs` and `delay` call it before any other check; `stop` instead returns
`the training session has already ended` as a successful result. The inspection
commands, `lr`, `save`, `screenshot`, `clear`, `quit` and `help` do not check: `lr` and
`save` then run their action immediately through `submit`. Because the check comes first,
the `STOPPED` / `FAILED` branches of `stop` and `epochs` apply only while the control is
still open.

### Training-thread actions

* `_set_learning_rate(value)` returns `apply(target)`. For every entry of
  `target.optimizer.param_groups` it fills a tensor `lr` in place (`fill_`) or assigns the
  float, and returns `"learning rate set to <value>"`. Scheduler state is not touched.
* `_save(ctx, path)` builds `Path(path).expanduser()` in the handler and submits
  `save(target)`, which creates the parent directories, calls
  `torch.save(target.model.state_dict(), destination)` and returns
  `"saved model weights to <resolved path>"`.
* `_queued(message, description)` turns the result of `submit` into the output: the
  action's own message when it ran immediately (training has ended), otherwise
  `"<description> - applied at the next batch boundary"` (`learning rate 0.005`, `saving`).

### Formatting

`status`, `history`, `arch` and `weights` format values with [Formatting](Formatting):
`fmt.loss`, `fmt.learning_rate`, `fmt.metric` (ratio-like metric names as percentages),
`fmt.duration` (timer with milliseconds, ETA without), `fmt.count` and `fmt.percent`; a
missing value prints as `—`. The `best` line of `status` shows `monitored_loss` of
`session.best_record()` ([Sync Events](Sync-Events)). `help` pads every usage to the longest one and appends
`type 'help <command>' for details` and the terminal keys (`KEYS`: Tab, ↑/↓, PgUp/PgDn,
Ctrl+Home/End); `help <command>` prints `(default …)` only for
optional arguments whose default is not `None`.

## Used by / uses

* Uses `Argument`, `Command`, `CommandError`, `CommandRegistry`
  ([Commands Registry](Commands-Registry)), `CommandContext` ([Commands Context](Commands-Context)),
  `TrainingTarget` ([Sync Control](Sync-Control)), `TrainingState` ([Sync Events](Sync-Events)),
  [Formatting](Formatting) and torch.
* [Runner](Runner): `run()` calls `create_registry(process.commands)` for both GUI
  frontends. Loaded lazily through `hollywood_training.commands`.

## Example

```python
from hollywood_training.commands import CommandContext, create_registry
from hollywood_training.sync import SessionState, TrainingControl, TrainingTarget

registry = create_registry(process.commands)        # built-ins + @process.command ones
context = CommandContext(
    control=TrainingControl(process.epochs),
    session=SessionState(),
    registry=registry,
    target=TrainingTarget(process.model, process.optimizer),
)
result = registry.execute("lr 0.01", context)
print(result.ok, result.output)  # True learning rate 0.01 - applied at the next batch boundary
print(registry.execute("history", context).output)  # no completed epochs yet
print(registry.execute("screenshot", context).output)
# screenshot is not available in this frontend   (no frontend was passed)
```

## Notes

* The module imports torch (for `torch.save` and tensor learning rates), which is why
  `hollywood_training.commands` loads it lazily.
* The state checks read `session.state`, the frontend's view. Whether the training
  thread has exited comes from `control.closed` (`_require_session`), because a run that
  finished while waiting for more epochs keeps the state `finished` after it has ended.
* `epochs` and `delay` also write the new value into the session, so `status` shows it
  before the manager's next `StateChanged`.

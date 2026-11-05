# Commands

`hollywood_training/commands/` · layer of the [Architecture](Architecture)

Runtime commands control and inspect a managed session while it runs. You type them into
the window's [terminal](GUI-Terminal), which keeps a history (Up / Down), completes
command names with Tab and scrolls its log with PgUp / PgDn and Ctrl+Home / Ctrl+End. A command never touches training state directly. It requests a
change through [`TrainingControl`](Sync-Control), or it queues an action that the training
thread runs at its next safe point between batches. This page covers the package, the
complete command reference and custom commands.

## Modules

| Module | Page | Contents |
|---|---|---|
| `registry.py` | [Commands Registry](Commands-Registry) | `Argument`, `Command`, `CommandError`, `CommandResult`, `CommandRegistry`, `split_command_line`: declaration, parsing, dispatch |
| `context.py` | [Commands Context](Commands-Context) | `CommandContext`, `FrontendActions`: what a handler can see and do |
| `view.py` | [View Requests](Commands-View) | `ViewRequest`, `ViewError`: commands that only change the window |
| `builtin.py` | [Built-in Commands](Commands-Builtin) | the built-in commands, `builtin_commands()`, `create_registry(extra)` |

`commands/__init__.py` imports `registry`, `context` and `view` eagerly (none needs torch)
and exports `Argument`, `Command`, `CommandContext`, `CommandError`, `CommandRegistry`,
`CommandResult`, `FrontendActions`, `ViewError`, `ViewRequest` and `split_command_line`. Only `builtin_commands` and
`create_registry` (from `builtin`, which imports torch) are listed in an explicit `_LAZY`
map and imported on first access by a PEP 562 `__getattr__`. The window process therefore
never loads torch through this package. The top-level package exports `ht.Argument`,
`ht.Command`, `ht.CommandError` and, lazily, `ht.CommandContext`.

## Where commands run

| Frontend | Typed in | The handler runs | Queued actions run |
|---|---|---|---|
| window process (default) | the window process; the line goes over the [link](GUI-Link-Protocol) | in the **training process**, on its command thread, one line at a time from a queue that the link thread fills ([GUI Remote Frontend](GUI-Remote-Frontend)) | on the training thread, at its next safe point |
| in-process window (`isolate_gui=False`) | the window | on the main thread, while the training loop pumps Qt events ([GUI In-Process Frontend](GUI-In-Process-Frontend)) | on the training thread, at the same safe point, right after the pump returns |
| console (`gui=False`) | no command input | | |

Safe points are: before every epoch, after every batch, and inside the manager's waits
(while paused, while lingering after the last epoch, during the per-epoch delay). Once the
training thread has exited, an action runs immediately on the handler's thread instead.

After the training session has ended (`TrainingControl.closed`), the commands that change
the run (`pause`, `resume`, `step`, `epochs`, `delay`) are refused with
`the training session has ended`. Inspection, `lr`, `save` and the session commands keep
working, so the results stay browsable until the window is closed.

The terminal shows a successful result as normal output and a failed one as an error. A
queued action's message arrives later as a success log line; if the action raises, an
error log line `<description> failed: <Type>: <message>` appears.

## Command line syntax

* `name arg1 arg2 …`; names and aliases are case-insensitive, arguments are positional.
  In usage strings `<arg>` is required and `[arg]` optional.
* Shell-like quoting: `save "my model.pt"`. On Windows backslashes are kept, so
  `save C:\runs\model.pt` works unquoted; elsewhere a backslash escapes the next
  character, as in a shell.
* Mistakes never disturb training. They produce a failed result:

| Input | Output |
|---|---|
| `paus` | `unknown command 'paus' - did you mean pause, status?` |
| `nope` | `unknown command 'nope' - type 'help'` |
| `epochs` | `missing total - usage: epochs <total>` |
| `step 1 2` | `too many arguments - usage: step [epochs]` |
| `lr abc` | `invalid value 'abc': could not convert string to float: 'abc'` |
| `save "a.pt` | `cannot parse command: No closing quotation` |

## Reference

| Usage | Aliases | Effect |
|---|---|---|
| `pause` | | pause at the next batch boundary |
| `resume` | `continue`, `start` | continue a paused run |
| `step [epochs]` | | run more epochs (default 1), then pause |
| `stop` | | end training at the next batch boundary |
| `epochs <total>` | | change the total number of epochs |
| `lr <value>` | | set the learning rate of every parameter group |
| `delay <ms>` | | wait after every epoch (0 = full speed) |
| `status` | | show the current training status |
| `history [count]` | | show the last epochs (default 10) |
| `arch` | `layers`, `architecture` | show the tracked architecture |
| `weights [layer]` | | show weight statistics |
| `scale [mode]` | | switch the loss axis between linear and logarithmic |
| `probe [epoch]` | | show an epoch's values and mark it on the loss chart |
| `neuron [column] [index]` | | highlight a neuron on the network view |
| `layout [panel] [percent]` | | show or change the panel sizes |
| `copy [lines]` | | copy the terminal log to the clipboard |
| `window [mode]` | | show or change the window mode |
| `save <path>` | | save the model's `state_dict` |
| `screenshot [path]` | | save a PNG of the window |
| `clear` | `cls` | clear the terminal |
| `quit` | `exit` | stop training and close the window |
| `help [command]` | `?` | list commands or explain one |

### Training

**`pause`** pauses at the next batch boundary. The training thread then delivers the
current weights (`TrainingPaused`) and waits; the window keeps animating. Output:
`pausing after the current batch`, or `already paused`. Fails with
`training is <state>, nothing to pause` when the session is not active (e.g. `finished`
while the run waits for more epochs), and with `the training session has ended` once the
training thread has exited.

**`resume`** (`continue`, `start`) clears a pause and cancels a pending `step`. Output:
`resuming`, or `not paused`. After `run(process, start_paused=True)` it starts training.
Fails with `the training session has ended` once the training thread has exited.

**`step [epochs]`** runs `epochs` more epochs (positive integer, default 1), then pauses
again; an epoch in progress counts as the first. It works from a paused or a running
session. Output: `running 3 epochs, then pausing`. Fails with
`training is <state>; use 'epochs <n>' to extend it` when the session is not active (a
finished run waiting for more epochs), and with `the training session has ended` once the
training thread has exited.

**`stop`** ends training at the next batch boundary. The window stays open for browsing
(unless `run(..., close_on_finish=True)`), and inspection commands keep working. It also
ends a finished run that waits for more epochs. Output: `stopping after the current batch`,
or `the training session has already ended` once the training thread has exited.

**`epochs <total>`** sets the total number of epochs (positive integer, at least the
completed epochs). While the window waits after the last epoch (state `finished`),
raising the total continues training. Output: `total epochs set to 400`, or
`continuing training to epoch 400`. Fails with `total epochs must be at least 12`, and
with `the training session has ended` once the training thread has exited (for example
after `stop`).

**`lr <value>`** sets the learning rate of every parameter group (positive, finite); tensor
learning rates are filled in place. A scheduler may change it again at the end of the
epoch. Output: `learning rate 0.005 - applied at the next batch boundary`, then the log
line `learning rate set to 0.005`. After training has ended, the second message is the
output.

**`delay <ms>`** waits `ms` milliseconds (non-negative, finite) after every epoch; a
pause or stop ends the wait early. Output: `waiting 150 ms after every epoch`;
`delay 0` gives `delay off - running as fast as synchronisation allows`. Fails with
`the training session has ended` once the training thread has exited.

### Inspection

**`status`** shows the network, state, epoch, last losses, best loss, learning rate, timer
(active time), ETA and delay, then one line per metric of the last epoch (validation
metrics prefixed `val `):

```text
network     SmallNet
state       running
epoch       6/20
train loss  0.6077
val loss    0.5974
best        0.5974 (epoch 6)
lr          0.01
timer       00:00:00.030
eta         00:00:00
delay       0 ms
accuracy    70.9%
```

**`history [count]`** lists the last `count` epochs (positive integer, default 10) with a
column per training metric of the newest epoch. Before the first epoch it prints
`no completed epochs yet`.

```text
 epoch       train         val          lr      time   accuracy
     5      0.6236      0.6131        0.01    0.004s      70.3%
     6      0.6077      0.5974        0.01    0.004s      70.9%
```

**`arch`** (`layers`, `architecture`) shows the tracked architecture: class and display
name, parameter counts, input operations, one line per connection layer with the
operations applied to its output, modules that are not drawn, and normalisation.

```text
SmallNet - 32 parameters (32 trainable)
  fc1  Linear 2→6                 Tanh
  fc2  Linear 6→2
normalisation: none
```

**`weights [layer]`** without an argument prints one line per connection layer of the
latest snapshot: mean, standard deviation, mean `|w|`, and `Δ`, the last update relative
to the previous weights. With a layer name it prints shape (and the drawn part), element
count, mean / std, min / max, mean `|w|`, L2 norm, the last update and the mean bias of the
drawn neurons. Statistics cover the full tensor ([Tracking Weight Data](Tracking-Weight-Data)).
Unknown names fail with `no layer 'x'; layers: fc1, fc2`.

```text
fc1 after epoch 1
  shape       6 out x 2 in (drawn 6x2)
  elements    12
  mean / std  -0.04106 / 0.32427
  min / max   -0.58198 / +0.56066
  mean |w|    0.26051
  L2 norm     1.13226
  last update 0.000% of the drawn weights
  bias mean   -0.11755 (drawn neurons)
```

### View

Window-only commands. They change only the window (they replace the mouse interactions
of the first version, SPEC R2) and need a window: without one they fail with
`<command> is not available in this frontend`. With the window process, the window writes
the answer ([View Requests](Commands-View)).

**`scale [lin|log]`** sets the loss chart's y axis to linear or logarithmic; without an
argument it switches to the other one. Output: `loss axis: logarithmic` or
`loss axis: linear`. A logarithmic axis needs a positive loss; until there is one the
chart stays linear and the output adds `(drawn linear until a loss is positive)`.

**`probe [epoch|off]`** prints the values of a completed epoch (the latest without an
argument) and marks it on the loss chart with the probe line: a vertical line at the
epoch, rings where it meets the training and validation curves, and a box with the
epoch's losses. The line is removed automatically when training continues: at the next
epoch, and when training resumes after a pause or a finished run (`resume`, `step`,
`epochs <n>`); setting changes such as `lr` keep it. `probe off` removes it on request
(`probe line removed`, or `no probe line to remove`). Probing another epoch moves the
line. While training is running, the output ends with
`training is running: the probe line clears at the next epoch ('pause' keeps it)`.

```text
ht ❯ probe 12
epoch 12 of 40
  train loss  0.3178
  val loss    0.4172
  lr          0.001
  time        0.042 s
  accuracy    88.4%
```

The title says `(best)` for the best epoch. Fails with `no completed epochs yet`, with
`epoch 50 has not completed yet (last completed: 25)`, and with
`invalid epoch 'x': must be an epoch number (1 or more) or off`.

**`neuron [column] [index]`** highlights a neuron on the network view with a ring and a
box showing what the neuron tooltip showed before: the column, the neuron index, its
label and its bias. The output is the same text. A column is named by its title as
drawn (`INPUT`, a layer name such as `fc1` or `0`, or `fc2-in` for a reshaped input;
case does not matter). The index is the neuron's index in the layer; only drawn neurons
can be highlighted (wide layers are sampled). `neuron off` removes the highlight.

```text
ht ❯ neuron
drawn columns ('neuron <column> <index>' highlights a neuron):
  INPUT  2 features            neurons 0-1
  FC1    20 of 128 neurons     neurons 0 … 127 (20 drawn)
  FC2    3 neurons             neurons 0-2
ht ❯ neuron fc2 1
FC2 · neuron 1
class 1
bias +0.04213
```

`neuron fc1` lists the drawn neurons of one column. Fails with `no network drawn yet`,
`no column 'x' - columns: …` and `FC1 has no drawn neuron 5 - drawn: 0, 7, 13, …`.

**`layout [panel] [percent]`** shows or changes how much room each panel gets. The panels
are `left` and `right` (the two columns), `network` and `terminal` (the left column), and
`status`, `chart` and `usage` (the right column). `layout chart 50` gives the loss chart
half of the right column; the status and usage panels share the rest in their current
proportions. A trailing `%` is fine. Minimum sizes still apply: when they win, the output
adds `(limited by the panels' minimum sizes)`. `layout <panel>` shows one share, and
`layout reset` restores the initial layout.

```text
ht ❯ layout
columns       left 62% · right 38%
left column   network 70% · terminal 30%
right column  status 44% · chart 36% · usage 20%
'layout <panel> <percent>' resizes, 'layout reset' restores
ht ❯ layout chart 50
right column  status 34% · chart 50% · usage 16%
```

Fails with `no panel 'x' - panels: …`, `invalid percent '0': must be a whole number from
1 to 99` and `'layout reset' takes no percentage`.

**`copy [lines]`** copies the terminal log to the clipboard: all of it, or its last
`lines` lines. The echo of the `copy` command itself is left out. Output:
`copied 12 lines to the clipboard`, or `nothing to copy: the log is empty`. The clipboard
is the window's: on X11 without a clipboard manager, paste before closing the window.

**`window [normal|max|full|min]`** changes the window mode from the keyboard: `max`
maximizes it, `full` fills the screen without a title bar, `min` minimizes it (bring it
back with the system's window switcher, e.g. Alt+Tab), and `normal` restores it. Without
an argument it shows the current mode. Output: `window: maximized`, `window: full screen`,
`window: minimized` or `window: normal (1500 × 920)`. The title bar and the system's own
window shortcuts keep working as well.

### Session

**`save <path>`** runs `torch.save(model.state_dict(), path)` on the training thread at the
next batch boundary. Parent directories are created and `~` is expanded; a relative path
is relative to the training process's working directory. Output:
`saving - applied at the next batch boundary`, then the log line
`saved model weights to /home/me/runs/model.pt`. After training has ended, the second
message is the output.

**`screenshot [path]`** saves a PNG of the window, by default
`hollywood-training-YYYYmmdd-HHMMSS.png` in the working directory. The in-process window
answers `screenshot saved to <path>`. With the window process the output is
`screenshot requested: <path>`, and the window then writes `screenshot saved to <path>` or
`screenshot failed: <reason>`.

**`clear`** (`cls`) clears the terminal output.

**`quit`** (`exit`) requests a stop and closes the window; `run()` then returns. Output:
`stopping and closing`.

### General

**`help [command]`** (`?`) without an argument lists every command, grouped as TRAINING,
INSPECTION, VIEW, SESSION, CUSTOM, GENERAL (other categories follow alphabetically). With a
command name or alias it shows the usage, summary, details, arguments with defaults and
aliases:

```text
ht ❯ help step
usage: step [epochs]
  run more epochs, then pause
arguments:
  epochs     epochs to run before pausing (default 1)
```

## Custom commands

Register a handler on the process with `@process.command(...)`
([Training Process](Training-Process)). The decorator returns the handler unchanged.

```python
def command(                        # TrainingProcess.command
    self,
    name: str,
    *,
    summary: str,
    arguments: Sequence[Argument] = (),
    aliases: Sequence[str] = (),
    details: str = "",
) -> Callable[[Handler], Handler]: ...

@dataclass(frozen=True, slots=True)
class Argument:                     # ht.Argument
    name: str
    converter: Callable[[str], Any] = str
    help: str = ""
    optional: bool = False
    default: Any = None
    choices: tuple[str, ...] | None = None
```

The handler is called as `handler(context, *arguments)` with one converted value per
declared argument (the `default` for an omitted optional one), and returns the text to
print (`None` or `""` prints nothing). `context` is a [`CommandContext`](Commands-Context):
`context.session` to read the session, `context.control` for requests,
`context.submit(action, description)` for anything that touches the model or optimizer,
`context.registry`, `context.frontend`.

```python
import torch
import hollywood_training as ht

@process.command(
    "dropout",
    summary="set the dropout probability",
    arguments=[ht.Argument("p", float, "probability, 0 <= p < 1")],
    details="Changes every nn.Dropout module between two batches.",
)
def dropout(context, p):
    if not 0 <= p < 1:
        raise ht.CommandError("p must be at least 0 and below 1")   # a failed result

    def apply(target):                    # runs on the training thread
        for module in target.model.modules():
            if isinstance(module, torch.nn.Dropout):
                module.p = p
        return f"dropout set to {p:g}"    # shown as a success log line

    return context.submit(apply, f"set dropout to {p:g}")


@process.command(
    "losses",
    summary="show recent losses of one phase",
    arguments=[
        ht.Argument("phase", str, "train or val", choices=("train", "val")),
        ht.Argument("count", int, "number of epochs", optional=True, default=5),
    ],
    aliases=("ls",),
)
def losses(context, phase, count):        # reads the session only: no submit needed
    records = context.session.history[-count:]
    if not records:
        return "no completed epochs yet"
    values = [r.train_loss if phase == "train" else r.val_loss for r in records]
    return "  ".join("—" if v is None else f"{v:.4f}" for v in values)
```

```text
ht ❯ losses test
phase must be one of: train, val
ht ❯ ls val 2
0.5518  0.4949
```

Rules:

* Names and aliases are one lowercase word, and a required argument cannot follow an
  optional one. Violations raise `ValueError` at decoration time.
* Names and aliases must not collide with a built-in or another custom command.
  `create_registry` raises `ValueError`, e.g. `command name 'status' is already used by
  'status'`, when `run()` builds the registry for a GUI frontend (before the window opens).
  The console frontend builds no registry.
* Custom commands get the category `custom` and are listed under CUSTOM by `help`.
* `choices` are checked on the raw text before conversion; a converter's `TypeError` or
  `ValueError` becomes `invalid <name> '<text>': <message>`; defaults are not converted.
* Raise `ht.CommandError` for a failed result with your message. Any other ordinary
  exception becomes a failed result `<Type>: <message>`; training continues either way.
  `KeyboardInterrupt` and `SystemExit` are not caught.
* Touch the model or optimizer only inside an action passed to `context.submit`. While
  training runs, `submit` returns `None` (so the handler prints nothing) and the action's
  message appears once the training thread has run it. After training has ended, `submit`
  runs the action at once and returns its message.
* Keep handlers short. In the window-process frontend they run one at a time on the
  command thread, so a slow handler delays the commands typed after it (acknowledgements
  go through the link thread and are not delayed); in-process they run on the thread that
  trains.

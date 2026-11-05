# Remote Frontend

`hollywood_training/gui/remote.py` · part of [GUI](GUI)

`RemoteFrontend` is the default frontend of `run()`, and it runs in the training process.
It starts the window as a separate Python process and forwards every session event to it
over an authenticated local connection. On synchronous events it blocks the training
thread until the window reports that it has drawn them. Command lines typed in the window
come back and run here, next to the model. Training itself never draws, and the window
keeps animating during a long batch.

## Responsibilities

* Start `python -m hollywood_training.gui.window_process`, connect to it, and wait until
  the window is up.
* Act as the [Sync Channel](Sync-Channel) sink: mirror each event into its own
  `SessionState`, then send it to the window.
* Receive acknowledgements (passed to `SyncChannel.acknowledge`) and command lines
  (passed to `CommandRegistry.execute`, with the result sent back).
* Sample CPU, memory and GPU usage in the training process, twice a second.
* Implement `FrontendActions` for the `clear`, `quit` and `screenshot` commands and the
  window-only commands ([View Requests](Commands-View)).
* End the session cleanly when training finishes, the window is closed, the window
  process dies, or Ctrl+C is pressed.

## API

```python
STARTUP_TIMEOUT = 60.0  # seconds for the window process to start and connect
SHUTDOWN_TIMEOUT = 10.0
USAGE_INTERVAL = 0.5
```

```python
class RemoteFrontend:
    def __init__(
        self,
        *,
        channel: SyncChannel,
        control: TrainingControl,
        registry: CommandRegistry,
        close_on_finish: bool = False,
        title: str | None = None,
    ) -> None

    def run(self, manager: TrainingManager) -> TrainingResult

    # FrontendActions
    def clear_output(self) -> None
    def close(self) -> None
    def screenshot(self, path: Path) -> Path | None
    def view(self, request: ViewRequest) -> str | None
```

* The constructor connects the frontend to `channel` as its sink. `registry` executes
  the command lines. `close_on_finish` closes the window once `manager.run()` has
  returned. `title` is passed to the window as `--title`.
* `session`: a `SessionState` that mirrors what the window shows. Commands such as
  `status` read it.
* `run(manager)` starts the window, runs `manager.run()` on the calling thread, and
  returns its `TrainingResult` once the window has closed. It raises `RuntimeError` when
  there is no display (`ensure_display()`), or when the window process exits before
  connecting, does not connect within `STARTUP_TIMEOUT`, cannot be connected to, or fails
  while opening the window. After Ctrl+C it re-raises `KeyboardInterrupt` once the window
  has closed and the child has been reaped.
* `clear_output()` and `close()` send `("frontend", "clear", None)` and
  `("frontend", "close", None)`.
* `screenshot(path)` sends `("frontend", "screenshot", <absolute path>)`, with
  `expanduser()` and `resolve()` applied, and returns `None`. The window process writes
  the file and reports `screenshot saved to ...` or `screenshot failed: ...` in its
  terminal. The `screenshot` command itself answers `screenshot requested: <name>`.
* `view(request)` sends `("frontend", "view", request)` and returns `None`. The window
  process carries the request out and writes its outcome into its terminal, so the
  command's own result is empty.

## How it works

**Start-up** (`run()` calls `ensure_display()`, then `_start_window()`):

1. A `Listener(authkey=secrets.token_bytes(32))` opens on a random local address: a Unix
   socket, or a named pipe on Windows.
2. `subprocess.Popen` starts the child with the same interpreter (`sys.executable`):

   ```
   python -m hollywood_training.gui.window_process --address <address> [--title <title>]
   ```

   Its environment is a copy of this one, plus `HOLLYWOOD_TRAINING_AUTHKEY` (the key as
   hex) and a `PYTHONPATH` prefixed with the folder that contains the
   `hollywood_training` package. The child inherits stdout and stderr, so its errors
   appear in the same terminal.
3. `accept()` runs on a helper thread. Every 0.1 s the caller checks whether the child has
   exited (`RuntimeError` with its exit code) or `STARTUP_TIMEOUT` has passed (the child
   is terminated, then `RuntimeError`). If `accept()` itself fails, the child is
   terminated and `RuntimeError` reports the error. Afterwards the listener is closed.
4. It waits up to `STARTUP_TIMEOUT` for `("ready",)`. If the connection breaks first (the
   child connected but could not create the window), the child gets 5 s to exit, and
   `RuntimeError` reports its exit code.
5. It sends `("completions", registry.names(include_aliases=True))` for Tab completion.

Then the receiving thread and the command thread start, and `manager.run()` runs on the
caller's thread.

**Threads in the training process:**

| Thread | Work | Sends |
|---|---|---|
| caller (training) | `manager.run()`. The sink `_forward` mirrors each event, starts sampling at `SessionStarted`, and sends the event. `SyncChannel.deliver` blocks here on synchronous events | `("event", envelope)`, and `("close",)` at the end |
| `hollywood-training-link` | receives `("ack", seq)` and calls `channel.acknowledge(seq)`. Puts each `("command", line)` on a `queue.SimpleQueue`. Ends on `("closed",)` or EOF | nothing |
| `hollywood-training-commands` | `_serve_commands`: takes the lines from the queue one at a time and runs `registry.execute(line, context)`. Ends on the `None` sentinel | `("result", ok, output)`, and `("frontend", ...)` from command handlers |
| `hollywood-training-usage` | calls `SystemMonitor(device).sample()` every `USAGE_INTERVAL` s, from the first `SessionStarted` | `("usage", sample)` |

Because the receiving thread only handles acknowledgements and `("closed",)`, a slow
command handler never delays an acknowledgement.

The `CommandContext` targets `manager.process.model` and `manager.process.optimizer`.
Handlers that need them queue the work with `CommandContext.submit`, so it runs on the
training thread between batches ([Commands Context](Commands-Context)).

**Losing the window.** `_lost_window()` runs when the window reports `("closed",)`, the
link reaches EOF, or a send of an event fails. It marks the window closed, calls
`control.stop()`, and calls `channel.close()`, which releases a `deliver()` that is
waiting for an acknowledgement. If the window vanished on its own, meaning the link
broke without `("closed",)` and not because `_shutdown()` closed it, then
`_report_lost_window()` first issues a `RuntimeWarning` once:
`the GUI window process ended unexpectedly; training was stopped`. This happens whether the
receive loop or a failed event send notices it first. A forced close that times out
during shutdown does not warn.

| Situation | What happens |
|---|---|
| Training ends, `close_on_finish=False` | `run()` (the [Runner](Runner)) creates the manager with `linger=True`, so it waits in `FINISHED` until stopped (`epochs <n>` trains further). After `manager.run()` returns, the frontend polls every 0.2 s until the user closes the window. |
| Training ends, `close_on_finish=True` | After `SessionFinished` has been drawn, sends `("close",)`. The window closes 400 ms later. Waits up to `SHUTDOWN_TIMEOUT` for `("closed",)`. |
| Training code raises | The manager ends in `FAILED` without lingering, and the terminal shows the traceback. `run()` then waits for the window as at a normal end, and the [Runner](Runner) raises `TrainingFailedError`. |
| User closes the window | `("closed",)`, then `_lost_window()`. Training stops at the next batch boundary. |
| Window process dies | EOF on the link or a failed event send. Either way the `RuntimeWarning` above (issued once), then `_lost_window()`. |
| Ctrl+C in the training terminal | The manager logs `interrupted`, delivers `SessionFinished` and re-raises ([Training Manager](Training-Manager)). The frontend then sends `("close",)`, waits up to `SHUTDOWN_TIMEOUT`, shuts down and re-raises `KeyboardInterrupt`. A Ctrl+C while waiting for the window also forces the close. The window process ignores SIGINT. |

**Shutdown.** Once the window has started, shutdown always runs at the end of `run()`. It
stops the sampler, ends the command thread with a `None` sentinel, closes the link, joins
the receiving and usage threads (2 s each), and waits `SHUTDOWN_TIMEOUT` for the child to
exit, terminating it otherwise.

## Used by / uses

* Created by `run()` ([Runner](Runner)) when `gui=True` and `isolate_gui=True` (the
  defaults).
* Uses the [Link Protocol](GUI-Link-Protocol), the [Window Process](GUI-Window-Process)
  (the child), `ensure_display` from the [In-Process Frontend](GUI-In-Process-Frontend),
  [Sync Channel](Sync-Channel), [Sync Control](Sync-Control),
  [Session State](Sync-Session-State), [Commands Registry](Commands-Registry),
  [Commands Context](Commands-Context), [System Monitor](Monitor-System) and
  [Training Manager](Training-Manager).

## Example

This is the wiring that `run()` does, as used in the tests:

```python
from hollywood_training.commands import create_registry
from hollywood_training.gui.remote import RemoteFrontend
from hollywood_training.sync import SyncChannel, TrainingControl
from hollywood_training.training import TrainingManager

control = TrainingControl(process.epochs)
channel = SyncChannel()
remote = RemoteFrontend(
    channel=channel, control=control, registry=create_registry(process.commands),
    close_on_finish=True,
)
manager = TrainingManager(process, channel=channel, control=control, linger=False)
result = remote.run(manager)      # returns after the window has closed
print(remote.session.epoch)       # the training-side mirror followed along
```

## Notes

* Commands run one at a time, in the order they were entered, on the command thread. A
  slow handler delays the following commands, but not the epoch hand-off. Work on the
  model still belongs in `context.submit(...)`.
* `session` is written on the training thread (`_forward`) and read by command handlers
  on the command thread.
* Importing this module imports the `hollywood_training.gui` package, and with it
  PySide6. The training process never creates a `QApplication` on this path.

# In-Process Frontend

`hollywood_training/gui/app.py` · part of [GUI](GUI)

`QtFrontend` shows the window inside the training process, with
`run(process, isolate_gui=False)`. It is meant for embedding in an existing Qt
application, or for places where a second process cannot be started. Training and the GUI share the caller's
thread, which must be the main thread. The training loop pumps the Qt event loop through
`QtPump`. Every event is applied, drawn and, if synchronous, acknowledged inside the
channel's sink, before `deliver()` returns. The module also provides `ensure_display()`,
which both frontends call.

## Responsibilities

* Check for a display before Qt is touched (`ensure_display`).
* Provide the manager's [event pump](Sync-Event-Pump) (`QtPump`).
* Create or reuse the `QApplication`, show the `MainWindow`, and run the manager on the
  same thread.
* Apply, draw and acknowledge each event, and start usage sampling in the usage panel.
* Run command lines and show their results. Stop training when the window closes or
  Ctrl+C is pressed.
* After training, keep the window open until it is closed, or close it automatically
  with `close_on_finish`.

## API

```python
CLOSE_DELAY_MS = 400  # with close_on_finish: keep the final state visible briefly
```

```python
def ensure_display() -> None
```

On Linux and the BSDs (`sys.platform` starting with `linux`, `freebsd`, `openbsd` or
`netbsd`), this raises `RuntimeError` unless `QT_QPA_PLATFORM`, `DISPLAY` or
`WAYLAND_DISPLAY` is set. The message starts with
`no display found (DISPLAY and WAYLAND_DISPLAY are unset)` and suggests a desktop
session, `run(process, gui=False)` or `QT_QPA_PLATFORM=offscreen`. On other platforms it
returns at once.

```python
class QtPump:
    POLL_INTERVAL = 0.004  # seconds; polling more often only costs time

    def __init__(self) -> None
    def poll(self) -> None
    def wait(self, timeout: float) -> None
    def interrupt(self) -> None
```

| Method | Behaviour |
|---|---|
| `poll()` | Calls `QApplication.processEvents()` if at least `POLL_INTERVAL` s have passed since the last poll or wait. Otherwise it returns at once. |
| `wait(timeout)` | Runs a nested `QEventLoop`, which a single-shot timer quits after `max(1, int(timeout * 1000))` ms. |
| `interrupt()` | Ends a running `wait()` early. |

```python
class QtFrontend:
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
    def screenshot(self, path: Path) -> Path
    def view(self, request: ViewRequest) -> str
```

* The constructor connects the frontend's sink to `channel`. Attributes: `session` (a
  `SessionState`), `pump` (a `QtPump` to pass to the manager) and `window` (the
  `MainWindow`, or `None` until `run()` creates it).
* `run(manager)` shows the window, runs `manager.run()` on this thread, and returns its
  `TrainingResult` once the window is closed. It raises `RuntimeError` when called off the
  main thread or when `ensure_display()` fails. After Ctrl+C it re-raises
  `KeyboardInterrupt` once the window is closed.
* `clear_output()` clears the terminal. `close()` closes the window at the next
  event-loop iteration.
* `screenshot(path)` returns `window.save_screenshot(path)`. It raises `CommandError`
  when there is no window or the file cannot be written.
* `view(request)` returns the text of `window.handle_view(request)` and raises
  `CommandError` with the window's message when the window refuses the request, or when
  there is no window ([View Requests](Commands-View)).

## How it works

**`run(manager)`:**

1. Checks for the main thread and calls `ensure_display()`.
2. Uses `QApplication.instance()`, or creates `QApplication([sys.argv[0]])` (with
   `"hollywood-training"` when `sys.argv` is empty). It turns `quitOnLastWindowClosed`
   off for the run and restores it at the end, so closing this window does not quit a
   host application.
3. Creates `MainWindow(session, completer=registry.complete, title=title)`, and a
   `CommandContext` with `frontend=self` whose `TrainingTarget` holds
   `manager.process.model` and `manager.process.optimizer`. It connects `closing` and
   `command_submitted`, shows the window and calls `processEvents()`, so the window is
   mapped before the first synchronous render.
4. Runs `manager.run()` inside the Ctrl+C guard. A `KeyboardInterrupt` turns into
   `manager.partial_result()`.
5. Lingers: unless the window is already closed, a nested `QEventLoop` runs until it
   closes. With `close_on_finish`, a timer closes it after `CLOSE_DELAY_MS`.
6. Finally calls `control.stop()`, closes the window if it is still open, calls
   `deleteLater()` and processes events, then restores `quitOnLastWindowClosed`.

**Events.** The sink `_handle` runs on the training thread, which is the GUI thread here:

```
manager: deliver(event)
  └▶ _handle(envelope)
       session.apply(event)
       on SessionStarted: usage.set_sampler(SystemMonitor(info.device).sample)
       window.apply(event)          synchronous events: render_now()
       finally: acknowledge(seq)    synchronous envelopes only
  ◀─ deliver returns: the next epoch can start
```

An exception while applying or drawing is printed with its traceback, and the event is
still acknowledged.

**Pumping.** The manager calls `pump.poll()` at every checkpoint between batches. It
calls `pump.wait(0.1)` while paused, while lingering in `FINISHED`, and while sleeping
for `delay` ([Training Manager](Training-Manager)). Commands run from the window's
`command_submitted` signal during such a poll or wait. After each command,
`pump.interrupt()` ends a running `wait()`, so the manager sees the request at once.
Closing the window sets the closed flag, calls `control.stop()` and interrupts the pump.

**Ctrl+C.** While `manager.run()` and the lingering loop execute, a SIGINT handler marks
the interrupt, calls `control.stop()` and schedules `window.close()`. The handler is
installed only on the interpreter's main thread. A no-op `QTimer` every 200 ms lets
Python run the handler while Qt sits in an event loop. The previous handler is restored
afterwards.

## Used by / uses

* Created by `run()` ([Runner](Runner)) for `isolate_gui=False`. `run()` passes
  `pump=frontend.pump` and `linger=not close_on_finish` to the `TrainingManager`.
* `ensure_display()` is also called by the [Remote Frontend](GUI-Remote-Frontend).
* Uses the [Main Window](GUI-Main-Window), [Sync Channel](Sync-Channel),
  [Sync Control](Sync-Control), [Session State](Sync-Session-State),
  [Event Pump](Sync-Event-Pump), [Commands Registry](Commands-Registry),
  [Commands Context](Commands-Context) and [System Monitor](Monitor-System).

## Example

```python
import hollywood_training as ht

result = ht.run(process, isolate_gui=False)   # call it from the main thread
```

The same wiring by hand, as `run()` does it:

```python
from hollywood_training.commands import create_registry
from hollywood_training.gui.app import QtFrontend
from hollywood_training.sync import SyncChannel, TrainingControl
from hollywood_training.training import TrainingManager

control = TrainingControl(process.epochs)
channel = SyncChannel()
frontend = QtFrontend(channel=channel, control=control, registry=create_registry(process.commands))
manager = TrainingManager(
    process, channel=channel, control=control, linger=True, pump=frontend.pump
)
result = frontend.run(manager)
```

## Notes

* **Main thread only.** Apart from the synchronous repaints, the window refreshes only
  when the training loop pumps. During one very long batch it refreshes when the batch
  ends, and the animation clock's `dt` clamp (0.1 s) keeps animations from jumping
  afterwards.
* **CPU threads.** Drawing between batches takes time on the training thread, and
  PyTorch's spinning thread pool suffers from it. While a window is open, `run()` caps
  PyTorch's intra-op threads at half the CPU cores (`torch_threads="auto"`). See the
  [Performance](Performance) page.
* **Embedding.** An existing `QApplication` is reused, and the stylesheet is applied only
  to the Hollywood Training window.

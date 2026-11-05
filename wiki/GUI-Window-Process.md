# Window Process

`hollywood_training/gui/window_process.py` · part of [GUI](GUI)

This module is the entry point of the GUI process,
`python -m hollywood_training.gui.window_process`. The
[Remote Frontend](GUI-Remote-Frontend) starts it; it is not meant to be run by hand. It
connects back to the training process, shows a [`MainWindow`](GUI-Main-Window), and
applies and draws every event it receives. A synchronous event is acknowledged only
after it has been drawn. The process imports no torch, because everything it receives is
plain data.

## Responsibilities

* Parse `--address` and `--title`, read the authentication key from the environment, and
  connect.
* Own the `QApplication` and the window, and report `("ready",)` once the window is
  shown.
* Receive messages on a background thread and handle them on the GUI thread.
* Acknowledge each synchronous event after `MainWindow.apply()` has repainted it.
* Forward entered command lines, show the results, run frontend actions, and report when
  the window closes.

## API

```python
CLOSE_DELAY_MS = 400
```

The delay between receiving `("close",)` and closing the window, so the final state stays
visible briefly.

```python
class WindowProcess:
    def __init__(self, link: Link, *, title: str | None = None) -> None
    def start(self) -> None
```

* Attributes: `link`, `session` (a new `SessionState`) and `window`
  (`MainWindow(session, completer=..., title=title)`).
* The completer returns the received command names that start with the lower-cased
  prefix.
* `start()` shows the window, processes pending Qt events, starts the daemon receiving
  thread `hollywood-training-link`, and sends `("ready",)`.

```python
def main(argv: list[str] | None = None) -> int
```

* Arguments: `--address` (required) and `--title` (default `None`).
* Reads the hex key from `HOLLYWOOD_TRAINING_AUTHKEY` and removes it from `os.environ`.
  Without a key it prints `hollywood_training.gui.window_process: no authentication key`
  to stderr and returns `2`.
* Ignores SIGINT, connects with `Client(address, authkey=key)`, creates
  `QApplication(["hollywood-training"])`, starts a `WindowProcess` and runs the event
  loop. When the loop ends it closes the link and returns `0`.

Under `if __name__ == "__main__"` the module calls `sys.exit(main())`.

## How it works

```
training process                               window process (this module)
────────────────                               ──────────────────────────────────────────
Listener.accept()  ◀── connect ──────────────  main(): Client(address, authkey)
                                               QApplication, WindowProcess(link).start()
wait for ready     ◀── ("ready",) ───────────  window shown
("completions", …) ──────────────────────────▶ receiving thread: link.receive()
("event", envelope) ─────────────────────────▶   └ _Inbox.received(message)
                                                     └ GUI thread: _dispatch(message)
                   ◀── ("ack", sequence) ────  after the event was drawn
```

The receiving thread only calls `link.receive()` and emits `_Inbox.received(message)`.
`_Inbox` is a `QObject` created on the GUI thread, so the signal reaches `_dispatch` as a
queued call on the GUI thread. All session updates and all drawing therefore happen on
the GUI thread, in the order the messages arrived.

| Message | Handling in `_dispatch` |
|---|---|
| `("event", envelope)` | `session.apply(event)`, then `window.apply(event)`, which repaints synchronous events. An exception is printed with its traceback. Afterwards, if `envelope.synchronous`, sends `("ack", envelope.sequence)` |
| `("usage", sample)` | `window.usage.set_sample(sample)` |
| `("result", ok, text)` | `window.show_command_result(ok, text)` |
| `("completions", names)` | stores the names, sorted, for Tab completion |
| `("frontend", "clear", None)` | `window.terminal.clear()` |
| `("frontend", "close", None)` | closes the window at the next event-loop iteration |
| `("frontend", "screenshot", path)` | `window.save_screenshot(Path(path))`. The terminal shows `screenshot saved to ...` (success) or `screenshot failed: ...` (error) |
| `("frontend", "view", request)` | `window.handle_view(request)` ([View Requests](Commands-View)); the outcome, if not empty, is written to the terminal at level `info`, or `error` when refused |
| `("close",)` | closes the window after `CLOSE_DELAY_MS` |
| `("disconnected",)` | created locally by the receiving thread when the link reaches EOF. Writes `training process disconnected` (warning) once. The window stays open |

Messages to the training process: `window.command_submitted` sends `("command", line)`,
and `window.closing` sends `("closed",)`. When the window closes, the last window is gone
and the application quits. `main()` then closes the link and returns.

## Used by / uses

* Started by the [Remote Frontend](GUI-Remote-Frontend), and speaks the
  [Link Protocol](GUI-Link-Protocol).
* Uses the [Main Window](GUI-Main-Window), [Session State](Sync-Session-State) and
  `Envelope` from the [Sync Channel](Sync-Channel).

## Notes

* **Torch-free.** The module imports PySide6, numpy and the package's data modules.
  Importing it loads neither `torch` nor `psutil`. The usage readings are taken in the
  training process.
* **SIGINT is ignored.** Ctrl+C in a terminal reaches the whole process group, and the
  training process decides when the window closes.
* **The key is removed** from the environment once it has been read.
* **A drawing bug cannot stall training.** An exception while applying or drawing is
  printed, and the event is still acknowledged.
* **A training process that disappears** leaves the window open with a warning in the
  terminal. The window can still be closed normally.

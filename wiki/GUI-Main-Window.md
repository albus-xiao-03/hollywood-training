# Main Window

`hollywood_training/gui/window.py` · part of [GUI](GUI)

`MainWindow` is the Hollywood Training window. It builds the six panels, arranges them
with splitters on a gradient backdrop, and routes session events to them. It never runs
commands or touches training objects itself. It shows what the
[session state](Sync-Session-State) holds and hands entered command lines to whichever
frontend owns it. Both frontends use it unchanged: the
[Window Process](GUI-Window-Process) and the [In-Process Frontend](GUI-In-Process-Frontend).

## Responsibilities

* Create the panels, the shared `AnimationClock`, the window icon and the stylesheet.
* Show each event: terminal lines for state changes, log messages and epoch summaries,
  plus a synchronous repaint (`render_now()`) for every synchronous event.
* Echo entered command lines, emit them as `command_submitted`, and print results handed
  back through `show_command_result()`.
* Carry out window-only command requests (`handle_view()`, [View Requests](Commands-View)).
* Route the keyboard to the terminal: `keyboard`, a [Keyboard Focus](GUI-Keyboard-Focus)
  lock, sends every key typed while the window is active to the command line.
* Save screenshots, and emit `closing` when the window is closed.

## API

```python
STATE_LEVEL = {
    TrainingState.RUNNING: "success",
    TrainingState.PAUSED: "warning",
    TrainingState.FINISHED: "success",
    TrainingState.STOPPED: "warning",
    TrainingState.FAILED: "error",
}
```

The terminal level of a `StateChanged` message, by state. Other states use `"info"`.

```python
class Backdrop(QWidget):
    def __init__(self, parent: QWidget | None = None) -> None
```

The central widget. It draws a radial deep-blue gradient (`background_bottom`, then
`background_top`, then `void` at the edge) with a 36 px grid in which every fourth line
is brighter. The result is cached in a pixmap per width, height and device pixel ratio.

```python
class MainWindow(QMainWindow):
    closing = Signal()
    command_submitted = Signal(str)

    def __init__(
        self,
        session: SessionState,
        *,
        completer: Callable[[str], list[str]] = lambda prefix: [],
        title: str | None = None,
        parent: QWidget | None = None,
    ) -> None

    def apply(self, event: Event) -> None
    def render_now(self) -> None
    def run_command(self, line: str) -> None
    def show_command_result(self, ok: bool, output: str) -> None
    def handle_view(self, request: ViewRequest) -> tuple[bool, str]
    def save_screenshot(self, path: Path) -> Path
    def closeEvent(self, event) -> None
```

* `session`: the state the panels read. The caller folds events into it.
* `completer`: Tab completion for the terminal, mapping a prefix to command names.
* `title`: a fixed window title. Without one, the title is
  `Hollywood Training — <network class>`, or `Hollywood Training` before a session.
* Attributes: `session`, `clock` (`AnimationClock(60, self)`, started by the constructor),
  and the panels `header` ([Header](GUI-Header)), `network`
  ([Network View](GUI-Network-View)), `terminal` ([Terminal](GUI-Terminal)), `status`
  ([Status Panel](GUI-Status-Panel)), `chart` ([Loss Chart](GUI-Loss-Chart)) and `usage`
  ([Usage Panel](GUI-Usage-Panel)), plus `keyboard`
  ([Keyboard Focus](GUI-Keyboard-Focus)), created after the panels.

| Member | Behaviour |
|---|---|
| `apply(event)` | Shows `event`, which must already be applied to `session` (see below). |
| `render_now()` | Calls `repaint()` on `header`, `network`, `status` and `chart`. Each runs its `paintEvent` before the call returns. |
| `run_command(line)` | Enters `line` as if it was typed: echo, then `command_submitted`. |
| `show_command_result(ok, output)` | Writes `output` at level `info` if `ok`, else `error`. Empty output writes nothing. |
| `handle_view(request)` | Carries out a window-only request: calls the handler registered for `request.action` with `*request.arguments`. Returns `(True, text)`, or `(False, message)` when the handler raises `ViewError` or the action is unknown (`unknown window action '<action>'`). |
| `save_screenshot(path)` | Creates the parent folders, saves `grab()` as a PNG and returns the resolved path. Raises `OSError` when the file cannot be written. |
| `closeEvent(event)` | Emits `closing`, stops the clock and the usage panel's timer, then closes. |
| `closing` | Emitted when the window closes. Both frontends stop training on it. |
| `command_submitted(str)` | An entered command line, for the frontend to execute. |

## How it works

**Layout.** Three splitters, each with an 8 px handle, non-collapsible children, and
stretch factors equal to the shares:

| Splitter | Orientation | Children (initial share) |
|---|---|---|
| left | vertical | `network` 70, `terminal` 30 |
| right | vertical | `status` 44, `chart` 36, `usage` 20 |
| main | horizontal | left 62, right 38 |

The backdrop's vertical layout (margins 14/8/14/14 px, spacing 6 px) holds the 80 px
header above the main splitter. The minimum size is 960 × 680. At construction the window
takes 88 % of the screen's available area, clamped to between 960 × 680 and
1680 × 1020 and to the available area, and centres itself. Without a screen it uses
1500 × 920. The terminal's command line holds the keyboard focus and is the window's only
focus target ([Keyboard Focus](GUI-Keyboard-Focus)).

**Events.** `apply()` handles each event type as follows:

| Event | Terminal output | Repaint |
|---|---|---|
| `SessionStarted` | window title updated; the banner `▓▒░ HOLLYWOOD TRAINING :: COMMAND TERMINAL ░▒▓`; network, parameter count and number of connection layers; epochs, batch size, loss and optimizer; `not drawn: ...` for unsupported modules; `type 'help' for commands · PgUp / PgDn scroll this log` | `render_now()` |
| `EpochCompleted` | `fmt.epoch_summary(...)` at level `dim`, for epoch 1, every `max(1, total_epochs // 20)`-th epoch and the last epoch | `render_now()` |
| `TrainingPaused` | none | `render_now()` |
| `StateChanged` | its `message`, if not empty, at the level from `STATE_LEVEL` (default `info`) | none (animated panels show it in the next frame) |
| `LogMessage` | `text` at `level.value` | none |
| `SessionFinished` | `<STATE>: <reason> in HH:MM:SS.mmm` at `error` (failed), `warning` (stopped) or `success` | `render_now()` |
| `BatchProgress` | none | none (the status panel reads batch progress every frame) |

The summary interval is computed once, from `SessionInfo.total_epochs` at session start.

**Commands.** The window only passes lines through:

```
TerminalPanel.command_entered(line) ─▶ terminal.echo(line) ─▶ command_submitted(line)
                                                                   │ frontend executes
show_command_result(ok, output) ◀──────────────────────────────────┘
```

## Used by / uses

* Used by `QtFrontend` ([In-Process Frontend](GUI-In-Process-Frontend)) with
  `completer=registry.complete`, and by `WindowProcess` ([Window Process](GUI-Window-Process))
  with the command names received over the link.
* Uses the panels in `gui.widgets`, [Keyboard Focus](GUI-Keyboard-Focus),
  [Theme](GUI-Theme) (`PALETTE`, `color`, `stylesheet`, `app_icon`),
  [Animation](GUI-Animation), [Formatting](Formatting)
  (`epoch_summary`, `duration`, `count`), [Sync Events](Sync-Events) and
  [Session State](Sync-Session-State).

## Example

```python
from pathlib import Path

from PySide6.QtWidgets import QApplication

from hollywood_training.commands import create_registry
from hollywood_training.gui import MainWindow
from hollywood_training.sync import SessionState

app = QApplication([])
session = SessionState()
window = MainWindow(session, completer=create_registry().complete)
window.command_submitted.connect(lambda line: window.show_command_result(True, f"got {line}"))
window.show()

for event in events:          # SessionStarted, EpochCompleted, ... e.g. from a SyncChannel sink
    session.apply(event)      # the state first
    window.apply(event)       # then the window; synchronous events are repainted here

window.save_screenshot(Path("shots/window.png"))
```

## Notes

* Use the window on the GUI thread only. `apply()` does not update the session. The
  frontend does that first, and `SessionState` is not thread-safe.
* `render_now()` leaves out `usage` and `terminal`, because they are not part of the
  epoch hand-off. Qt's `repaint()` paints at once only while a widget is visible, and
  both frontends show the window before the first event arrives.
* The stylesheet is set on the window, not on the application, so a host Qt application
  keeps its own style.

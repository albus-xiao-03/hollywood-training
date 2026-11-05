# Terminal

`hollywood_training/gui/widgets/terminal.py` · part of [GUI](GUI)

This is the integrated command terminal (SPEC GUI 5): a coloured, read-only output log
and a command line with history (Up / Down) and Tab completion of command names.
PgUp / PgDn and Ctrl+Home / Ctrl+End, typed in the command line, scroll the log. The
panel only collects input and shows text. Entered lines leave through the
`command_entered` signal. The frontend that owns the window executes them with the
[command registry](Commands-Registry) and writes the outcome back. The commands are
listed on the [Commands](Commands) page.

The command line is the window's only keyboard focus target: while the window is active,
every key goes there ([Keyboard Focus](GUI-Keyboard-Focus)).

## Responsibilities

* Keep a scrolling log, with one colour per message level.
* Provide a command line with history and command-name completion.
* Scroll the log from the keyboard (page up / down, top, bottom).
* Consume every key it receives, so that no key travels on to the panels around it.
* Emit entered lines, and leave their execution to the frontend.

## API

```python
LEVEL_COLORS = {
    "info": PALETTE.text,
    "success": PALETTE.success,
    "warning": PALETTE.warning,
    "error": PALETTE.error,
    "command": PALETTE.accent,
    "dim": PALETTE.text_dim,
    "accent": PALETTE.ice,
}

PROMPT = "ht ❯"
```

`LEVEL_COLORS` gives the colour of each output level, and an unknown level uses
`PALETTE.text`. `PROMPT` is shown before the input field and in echoed commands.

```python
PAGE_KEYS = {
    Qt.Key.Key_PageUp: QAbstractSlider.SliderAction.SliderPageStepSub,
    Qt.Key.Key_PageDown: QAbstractSlider.SliderAction.SliderPageStepAdd,
}
EDGE_KEYS = {  # with Ctrl
    Qt.Key.Key_Home: QAbstractSlider.SliderAction.SliderToMinimum,
    Qt.Key.Key_End: QAbstractSlider.SliderAction.SliderToMaximum,
}
```

The keys of the command line that scroll the log, as scroll-bar actions. Shift does not
matter for `PAGE_KEYS`; `EDGE_KEYS` need Ctrl (Home and End alone move the text cursor).

```python
class CommandLine(QLineEdit):
    submitted = Signal(str)
    candidates = Signal(list)
    scroll_requested = Signal(object)  # QAbstractSlider.SliderAction

    def __init__(
        self, completer: Callable[[str], list[str]], parent: QWidget | None = None
    ) -> None

    @property
    def history(self) -> list[str]
    def complete(self) -> None
```

* `completer(prefix)` returns the command names that start with `prefix`.
* `submitted(text)` is emitted on Return when the line is not blank. It carries the
  stripped line, and the field is cleared.
* `candidates(names)` is emitted when several command names match on Tab.
* `scroll_requested(action)` is emitted for a scroll key (`PAGE_KEYS`, Ctrl + `EDGE_KEYS`).
  The key does not edit the line.
* `history` returns a copy of the entered lines. A line equal to the one before it is not
  stored twice.
* `complete()` does what Tab does (see below).
* Every key press and release is accepted after it was handled, including keys the line
  has no use for. Unhandled keys therefore never travel up to the parent panels, which
  [Keyboard Focus](GUI-Keyboard-Focus) relies on to forward stray keys without loops.

```python
class TerminalPanel(HudPanel):
    command_entered = Signal(str)

    def __init__(
        self, completer: Callable[[str], list[str]], parent: QWidget | None = None
    ) -> None

    def write(self, text: str, level: str = "info") -> None
    def scroll(self, action: QAbstractSlider.SliderAction) -> None
    def echo(self, command: str) -> None
    def clear(self) -> None
    def text(self) -> str
```

* Attributes: `output`, a read-only `QPlainTextEdit` that keeps at most 4000 lines and
  wraps at the widget width, and `input`, the `CommandLine`.
* `command_entered(line)` re-emits `input.submitted`.
* `write(text, level)` appends `text`, which may contain newlines, as a new block in the
  colour of `level`. Empty text is ignored. The view scrolls to the end only if it was
  already at the bottom.
* `scroll(action)` triggers `action` on the log's vertical scroll bar, exactly as the
  scroll bar itself would: `SliderPageStepSub` / `SliderPageStepAdd` move one page (the
  visible height), `SliderToMinimum` / `SliderToMaximum` jump to the top or the bottom.
  It is connected to `input.scroll_requested`.
* `echo(command)` writes `ht ❯ <command>` at level `command`.
* `clear()` empties the log, and `text()` returns the whole log as plain text.

## How it works

**Keys in the command line:**

| Key | Effect |
|---|---|
| Return | submits the line (blank lines are ignored) and moves the history cursor past the newest entry |
| Up | shows the previous history entry |
| Down | shows the next entry, or an empty line after the newest |
| Tab | runs `complete()` instead of moving the focus |
| Shift+Tab | does nothing (the focus never leaves the command line) |
| PgUp / PgDn (with or without Shift) | scrolls the log one page up / down |
| Ctrl+Home / Ctrl+End | scrolls the log to the top / bottom |
| any other key | normal line editing (arrows, Home / End, Shift+arrows to select, Ctrl+A / C / X / V / Z); keys without a meaning are consumed |

`complete()` only acts on a single word (a line without spaces). With one match, the line
becomes `<match> ` with a trailing space. With several matches, the line is extended to
their longest common prefix (if that is longer), and `candidates` is emitted. The panel
writes the candidates as one `dim` line, separated by three spaces. With no match,
nothing happens.

**Command flow:**

```
CommandLine (Return) ─ submitted ─▶ TerminalPanel.command_entered ─▶ MainWindow:
    terminal.echo(line), then command_submitted(line) ─▶ frontend: CommandRegistry.execute
frontend ─▶ MainWindow.show_command_result(ok, output) ─▶ terminal.write(output, "info" or "error")
```

The completer depends on the frontend. In-process it is `CommandRegistry.complete`. In
the window process it filters the names received in `("completions", ...)` by the
lower-cased prefix.

**Levels the window uses:** `accent` for the session banner, `info` for session details
and results, `dim` for epoch summaries and completion candidates, `success`, `warning`
and `error` for states, log messages and failures, and `command` for echoed lines.

## Used by / uses

* Used by the [Main Window](GUI-Main-Window) as `window.terminal`. Both frontends also
  call `terminal.clear()` for the `clear` command, and the window process writes
  screenshot and disconnect messages into it.
* [Keyboard Focus](GUI-Keyboard-Focus) keeps the focus on `terminal.input` and forwards
  keys addressed to other widgets to it.
* Uses [HUD Panel](GUI-HUD-Panel) and [Theme](GUI-Theme) (`PALETTE`, `color`,
  `mono_font`). The object names `terminalOutput`, `terminalInput` and `terminalPrompt`
  are styled by `stylesheet()`.

## Example

```python
from PySide6.QtWidgets import QApplication

from hollywood_training.commands import create_registry
from hollywood_training.gui.widgets import TerminalPanel

app = QApplication([])
terminal = TerminalPanel(create_registry().complete)
terminal.command_entered.connect(lambda line: terminal.write(f"would run {line!r}", "dim"))
terminal.write("all systems nominal", "success")

terminal.input.setText("hist")
terminal.input.complete()        # the line becomes "history "
terminal.input.setText("h")
terminal.input.complete()        # writes "help   history" as a dim line
```

## Notes

* The log uses a 9 pt monospace font, and the input and the bold prompt use 10 pt. The
  placeholder text is `type a command - 'help' lists them, Tab completes`.
* Completion covers command names and aliases only, not arguments.

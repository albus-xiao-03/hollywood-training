# View Actions

`hollywood_training/gui/view_actions.py` · part of [GUI](GUI)

`ViewActions` carries out the window-only commands in the window. These commands are the
keyboard replacements for the mouse interactions of the first version (SPEC R2) and are
listed under `VIEW` in `help`. They reach the window as a
[`ViewRequest`](Commands-View): `MainWindow.handle_view(request)` passes it to
`ViewActions.handle()`, which calls the handler registered for the request's action. A
handler changes the window through the panels' own methods and returns the text the
command answers. [Main Window](GUI-Main-Window) creates one instance as
`window.view_actions`.

## Responsibilities

* Map action names to handlers, and carry out a request with its arguments.
* Turn a refusal (`ViewError`) or an unknown action into `(False, reason)`.
* Compose the answers of the window-only commands.

## API

```python
Handler = Callable[..., str]

class ViewActions:
    def __init__(self, window: MainWindow) -> None

    @property
    def names(self) -> list[str]
    def register(self, action: str, handler: Handler) -> None
    def handle(self, request: ViewRequest) -> tuple[bool, str]

    # the built-in actions
    def scale(self, mode: str | None) -> str
    def probe(self, epoch: int | None) -> str
    def neuron(self, column: str | None, index: int | None) -> str
    def layout(self, panel: str | None, percent: int | None) -> str
```

* `names`: the registered action names, sorted.
* `register(action, handler)` adds a handler, or replaces the one registered for
  `action`.
* `handle(request)` calls the handler of `request.action` with `*request.arguments`. It
  returns `(True, answer)`, or `(False, reason)` when the handler raises `ViewError` or
  the action is unknown (`unknown window action '<action>'`). Other exceptions propagate:
  they are bugs, and the frontends report them like any other failure.

## How it works

| Action | Command | Handler | Effect and answer |
|---|---|---|---|
| `scale` | `scale [lin\|log]` | `scale(mode)` | `chart.set_log_scale(...)`: `log` / `lin`, or the opposite of the current scale for `None`. Answers `loss axis: linear` or `loss axis: logarithmic`, plus `(drawn linear until a loss is positive)` when `chart.has_positive_loss()` is false |
| `layout` | `layout [panel] [percent]`, `layout reset` | `layout(panel, percent)` | No arguments: every panel's share, one line per splitter (`columns  left 62% · right 38%`, `left column  network 70% · terminal 30%`, `right column  status 44% · chart 36% · usage 20%`) and a usage hint. A panel only: `<panel>: <share>% of the right column`. A panel and a percentage: `window.set_panel_share(...)`, answering the new shares of that splitter, plus `(limited by the panels' minimum sizes)` when the panel's share ends up more than 1 point off. `reset`: `window.reset_layout()`, answering `layout reset` and the shares. Panel names ignore case. Refuses `no panel 'x' - panels: …` and `'layout reset' takes no percentage` |
| `neuron` | `neuron [column] [index]`, `neuron off` | `neuron(column, index)` | No arguments: lists the drawn columns (`drawn columns ('neuron <column> <index>' highlights a neuron):`, then one line per column with its title, caption and drawn neuron indices, e.g. `neurons 0-5` or `neurons 0 … 39 (8 drawn)`). A column only: `<COLUMN> · drawn neurons 0, 6, 11-12, …`. A column and an index: `network.select_neuron(...)`, answering `network.neuron_lines(...)`. `off`: removes the highlight (`neuron highlight removed`, or `no neuron highlighted`). Refuses `no network drawn yet`, `no column 'x' - columns: …`, `<COLUMN> has no drawn neuron n - drawn: …` |
| `probe` | `probe [epoch\|off]` | `probe(epoch)` | `chart.set_probe(epoch)` for an epoch, answering nothing (the command reports the epoch's values). For `None` (`probe off`): removes the line and answers `probe line removed`, or `no probe line to remove` |

The arguments arrive already checked by the command handler (see
[Built-in Commands](Commands-Builtin)). What only the window knows, such as which
neurons are drawn, is checked here.

**Column names.** `neuron` finds a column by the title the network view draws above it:
`INPUT`, a layer name in upper case such as `FC1`, or `<LAYER> IN` for a reshaped input.
The comparison ignores case, and `-` or `_` stand for a space (`neuron fc2-in 3`).
Columns have no numbers: the layers of an `nn.Sequential` are named `0`, `1`, `2`, ...
themselves, so a number would be ambiguous.

## Used by / uses

* Used by [Main Window](GUI-Main-Window) (`handle_view`), and through it by both frontends
  and the [Window Process](GUI-Window-Process).
* Uses `ViewRequest` and `ViewError` ([View Requests](Commands-View)) and the panels of
  the window: [Loss Chart](GUI-Loss-Chart), [Network View](GUI-Network-View), and the
  splitters of the [Main Window](GUI-Main-Window) (`panel_shares`, `set_panel_share`,
  `reset_layout`).

## Example

```python
from hollywood_training.commands import ViewRequest

window.handle_view(ViewRequest("scale", ("log",)))   # (True, "loss axis: logarithmic")
window.handle_view(ViewRequest("zoom"))               # (False, "unknown window action 'zoom'")

window.view_actions.register("hello", lambda: "hello from the window")
window.handle_view(ViewRequest("hello"))              # (True, "hello from the window")
```

## Notes

* Handlers run on the GUI thread, in both frontends.
* The answers are written by the window process itself when the window has its own
  process, and are the command's output in-process.

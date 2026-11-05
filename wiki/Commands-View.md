# View Requests

`hollywood_training/commands/view.py` · part of [Commands](Commands)

Most commands act on the training run. A few only change what the window shows or does:
they are the keyboard replacements for the mouse interactions of the first version
(SPEC R2), listed under `VIEW` in `help`. Their handlers check the arguments and hand a
`ViewRequest` to `CommandContext.show()`. The frontend passes the request on to
`MainWindow.handle_view()`, across the process link when the window runs in its own
process, and the window carries it out.

## Responsibilities

* Describe a window-only action as plain, picklable data that needs neither torch nor Qt.
* Give the window a way to refuse a request with a message (`ViewError`).

## API

```python
class ViewError(Exception): ...

@dataclass(frozen=True, slots=True)
class ViewRequest:
    action: str
    arguments: tuple[object, ...] = ()
```

* `ViewRequest.action` names what the window should do. `arguments` are already
  converted and validated by the command handler (numbers, keywords, `None`).
* `ViewError` is raised by the window's action handlers when a request cannot be carried
  out with the current window state, for example a neuron that is not drawn. Its message
  becomes the command's error output.

Both are exported by `hollywood_training.commands`.

## How it works

```
handler(ctx, ...) ─▶ ctx.show(ViewRequest(action, arguments))
                       │ require_frontend(action).view(request)
        ┌──────────────┴───────────────────────────────┐
  QtFrontend.view                                 RemoteFrontend.view
  window.handle_view(request) ─▶ (ok, text)       sends ("frontend", "view", request); returns None
  ok: returns text ─▶ command output              window process: window.handle_view(request)
  not ok: raises CommandError(text)               writes text to its terminal (info / error)
```

* `CommandContext.show(request)` returns the outcome text, or `""` when the window reports
  the outcome itself (window process). Without a frontend (no window) it raises
  `CommandError("<action> is not available in this frontend")`.
* `MainWindow.handle_view(request)` passes the request to `ViewActions.handle()`
  ([View Actions](GUI-View-Actions)), which looks the action up and calls its handler
  with `*request.arguments`. It returns `(True, text)`, or
  `(False, message)` for a `ViewError` or an unknown action
  (`unknown window action '<action>'`).

## Used by / uses

* Used by [Commands Context](Commands-Context) (`FrontendActions.view`,
  `CommandContext.show`), the window-only commands in [Built-in Commands](Commands-Builtin),
  both frontends ([In-Process Frontend](GUI-In-Process-Frontend),
  [Remote Frontend](GUI-Remote-Frontend)), the [Window Process](GUI-Window-Process), the
  [Main Window](GUI-Main-Window) and [View Actions](GUI-View-Actions), which carries the
  requests out.
* Sent across the [link](GUI-Link-Protocol) as `("frontend", "view", request)`.
* Uses only the standard library.

## Example

```python
from hollywood_training.commands import Command, ViewRequest

def scale_log(ctx):
    # the window switches the loss chart to a logarithmic axis and reports it
    return ctx.show(ViewRequest("scale", ("log",)))

command = Command("logaxis", "logarithmic loss axis", scale_log, category="view")
```

## Notes

* A request is fire-and-forget across the link: the command's own result is empty, and
  the window process writes the outcome into its terminal right before that empty result
  arrives. In-process, the outcome is the command's result.
* The handlers in the window run on the GUI thread, in both frontends.

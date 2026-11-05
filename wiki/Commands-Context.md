# Command Context

`hollywood_training/commands/context.py` · part of [Commands](Commands)

`CommandContext` is what a command handler receives as its first argument. It gives
read access to the session through [`SessionState`](Sync-Session-State), requests changes
through [`TrainingControl`](Sync-Control), and routes anything that touches the model or
optimizer through `submit()`, which hands it to the training thread (or runs it directly
once training has ended). `FrontendActions` describes the optional abilities of the
frontend that runs the commands. Commands that only change the window hand a
[`ViewRequest`](Commands-View) to `show()`.

## Responsibilities

* Bundle the control, the session state, the registry and the frontend for handlers.
* Keep the model and optimizer out of direct reach: only queued actions receive them.
* Report frontend features that are not available as a `CommandError`.
* Pass window-only requests to the frontend (`show()`).

## API

```python
class FrontendActions(Protocol):
    def clear_output(self) -> None: ...
    def close(self) -> None: ...
    def screenshot(self, path: Path) -> Path | None:
        """Save a PNG; returns its path, or None when it is written asynchronously."""
    def view(self, request: ViewRequest) -> str | None:
        """Carry out a window-only request; returns its outcome, or None when the window
        reports the outcome itself (asynchronously). Raises CommandError when refused."""
```

| Implementation | `clear_output()` | `close()` | `screenshot(path)` | `view(request)` |
|---|---|---|---|---|
| `QtFrontend` ([in-process](GUI-In-Process-Frontend)) | clears the terminal | closes the window (queued with `QTimer.singleShot(0, …)`) | saves synchronously and returns the resolved path; raises `CommandError` on failure | `window.handle_view(request)`: returns the text, or raises `CommandError` with the window's message |
| `RemoteFrontend` ([window process](GUI-Remote-Frontend)) | sends `("frontend", "clear", None)` | sends `("frontend", "close", None)` | sends `("frontend", "screenshot", <absolute path>)` and returns `None`; the window reports the result | sends `("frontend", "view", request)` and returns `None`; the window reports the outcome |

```python
class CommandContext:
    def __init__(
        self,
        *,
        control: TrainingControl,
        session: SessionState,
        registry: CommandRegistry,
        target: TrainingTarget,
        frontend: FrontendActions | None = None,
    ) -> None: ...

    @property
    def control(self) -> TrainingControl: ...
    @property
    def session(self) -> SessionState: ...
    @property
    def registry(self) -> CommandRegistry: ...
    @property
    def frontend(self) -> FrontendActions | None: ...

    def submit(self, action: ControlAction, description: str) -> str | None: ...
    def require_frontend(self, feature: str) -> FrontendActions: ...
    def show(self, request: ViewRequest) -> str: ...
```

* `submit(action, description)` queues `action` with `control.submit(...)` and returns
  `None`; the training thread runs `action(target)` at its next safe point and reports the
  returned message as a log event. If the control is already closed (training has ended),
  it calls `action(target)` immediately and returns its message, or `""` for `None`.
  Exceptions from an immediate run propagate to the registry, which turns them into a
  failed result.
* `require_frontend(feature)` returns the frontend, or raises
  `CommandError(f"{feature} is not available in this frontend")` when there is none.
* `show(request)` returns `require_frontend(request.action).view(request)`, or `""` when
  the frontend returns `None` (the window reports the outcome itself). See
  [View Requests](Commands-View).

The `target` is not exposed as a property; handlers reach it only through `submit()`.

## How it works

A frontend creates one context per session, with
`TrainingTarget(manager.process.model, manager.process.optimizer)` and itself as
`frontend`, and passes it to every `registry.execute(line, context)`.

| Frontend | Handlers run on | `submit` while training runs | `submit` after training ended |
|---|---|---|---|
| window process (default) | the training process's command thread (fed by the link thread through a queue) | queued for the training thread | runs on the command thread |
| in-process window | the main thread, during the pump | queued, applied right after the pump returns | runs on the main thread |

## Used by / uses

* Uses [`TrainingControl`, `TrainingTarget`, `ControlAction`](Sync-Control),
  [`SessionState`](Sync-Session-State), `CommandError` / `CommandRegistry` from
  [Commands Registry](Commands-Registry) and `ViewRequest` from
  [View Requests](Commands-View). It imports no torch.
* Created by [GUI Remote Frontend](GUI-Remote-Frontend) and
  [GUI In-Process Frontend](GUI-In-Process-Frontend).
* Every handler in [Built-in Commands](Commands-Builtin) and every custom command
  ([Commands](Commands)) receives it. Exported eagerly by `hollywood_training.commands`
  (together with `FrontendActions`) and lazily as `ht.CommandContext`.

## Example

```python
from pathlib import Path

import torch
import hollywood_training as ht

@process.command("probe", summary="mean output of the model for a zero input")
def probe(context: ht.CommandContext):
    epoch = context.session.epoch                     # read the session in the handler

    def run(target):                                  # training thread, between batches
        device = next(target.model.parameters()).device
        was_training = target.model.training
        target.model.eval()
        with torch.no_grad():                         # a model with two inputs
            value = target.model(torch.zeros(1, 2, device=device)).mean().item()
        target.model.train(was_training)
        return f"mean output {value:.4f} after epoch {epoch}"

    return context.submit(run, "probe the model")     # None while queued

@process.command("snap", summary="screenshot into ./shots")
def snap(context):
    frontend = context.require_frontend("snap")       # CommandError without a frontend
    saved = frontend.screenshot(Path("shots/latest.png"))
    return f"saved {saved}" if saved is not None else "requested"
```

## Notes

* Handlers run on the thread that serves command input. With the in-process window that is
  the GUI thread, which is also the thread that trains. With the default window process
  they run in the training process, next to the model, on a dedicated command thread; the
  window only sends the line and shows the result.
* Handlers should return quickly. The command thread runs them one at a time, so a slow
  handler delays the commands queued after it. Acknowledgements are received by the link
  thread and are not delayed.

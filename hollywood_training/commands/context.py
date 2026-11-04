"""``CommandContext``: what a command handler can see and do.

Handlers run on the thread that serves command input: the GUI thread for the
in-process window, and a command thread of the training process for the default window
process (the line is typed in the window and sent back). The console frontend takes no
commands. Handlers read the session through
:class:`~hollywood_training.sync.session.SessionState`, request changes through
:class:`~hollywood_training.sync.control.TrainingControl`, and anything touching the
model or optimizer goes through :meth:`CommandContext.submit`, which hands it to the
training thread (or runs it directly once training has ended).
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

from ..sync.control import ControlAction, TrainingControl, TrainingTarget
from ..sync.session import SessionState
from .registry import CommandError, CommandRegistry


class FrontendActions(Protocol):
    """Optional abilities of the frontend that runs the commands."""

    def clear_output(self) -> None: ...

    def close(self) -> None: ...

    def screenshot(self, path: Path) -> Path | None:
        """Save a PNG; returns its path, or None when it is written asynchronously."""


class CommandContext:
    """Everything a command handler receives as its first argument."""

    def __init__(
        self,
        *,
        control: TrainingControl,
        session: SessionState,
        registry: CommandRegistry,
        target: TrainingTarget,
        frontend: FrontendActions | None = None,
    ) -> None:
        self._control = control
        self._session = session
        self._registry = registry
        self._target = target
        self._frontend = frontend

    @property
    def control(self) -> TrainingControl:
        return self._control

    @property
    def session(self) -> SessionState:
        return self._session

    @property
    def registry(self) -> CommandRegistry:
        return self._registry

    @property
    def frontend(self) -> FrontendActions | None:
        return self._frontend

    def submit(self, action: ControlAction, description: str) -> str | None:
        """Run ``action(target)`` on the training thread at its next safe point.

        Returns None when queued (its message is reported later as a log event), or the
        action's own message when it ran immediately because training has ended.
        """
        if self._control.submit(action, description):
            return None
        return action(self._target) or ""

    def require_frontend(self, feature: str) -> FrontendActions:
        if self._frontend is None:
            raise CommandError(f"{feature} is not available in this frontend")
        return self._frontend

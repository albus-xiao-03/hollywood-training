"""``ViewActions``: what the window-only commands do in the window.

The ``view`` commands (SPEC R2: the keyboard replacements for the mouse interactions of
the first version) reach the window as a
:class:`~hollywood_training.commands.view.ViewRequest`. :meth:`MainWindow.handle_view
<hollywood_training.gui.window.MainWindow.handle_view>` passes the request to
:meth:`ViewActions.handle`, which calls the handler registered for its action. A handler
changes the window through the panels' own methods and returns the text the command
answers, or raises :class:`~hollywood_training.commands.view.ViewError` when the request
cannot be carried out.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

from ..commands.view import ViewError, ViewRequest

if TYPE_CHECKING:
    from .window import MainWindow

Handler = Callable[..., str]


class ViewActions:
    """The window-only actions of one :class:`MainWindow`, by action name."""

    def __init__(self, window: MainWindow) -> None:
        self._window = window
        self._handlers: dict[str, Handler] = {
            "scale": self.scale,
            "probe": self.probe,
        }

    @property
    def names(self) -> list[str]:
        """The registered action names, sorted."""
        return sorted(self._handlers)

    def register(self, action: str, handler: Handler) -> None:
        """Add (or replace) the handler of ``action``."""
        self._handlers[action] = handler

    def handle(self, request: ViewRequest) -> tuple[bool, str]:
        """Carry out ``request``: ``(True, answer)``, or ``(False, reason)`` when refused."""
        handler = self._handlers.get(request.action)
        if handler is None:
            return False, f"unknown window action {request.action!r}"
        try:
            return True, handler(*request.arguments)
        except ViewError as exc:
            return False, str(exc)

    # -------------------------------------------------------------- loss chart
    def scale(self, mode: str | None) -> str:
        """``scale [lin|log]``: the loss axis (replaces clicking the chart)."""
        chart = self._window.chart
        log = not chart.log_scale if mode is None else mode == "log"
        chart.set_log_scale(log)
        if not log:
            return "loss axis: linear"
        if chart.has_positive_loss():
            return "loss axis: logarithmic"
        return "loss axis: logarithmic (drawn linear until a loss is positive)"

    def probe(self, epoch: int | None) -> str:
        """``probe [epoch|off]``: the probe line on the loss chart (replaces hovering it)."""
        chart = self._window.chart
        if epoch is not None:
            chart.set_probe(epoch)
            return ""  # the command itself reports the epoch's values
        if chart.probe_epoch is None:
            return "no probe line to remove"
        chart.set_probe(None)
        return "probe line removed"

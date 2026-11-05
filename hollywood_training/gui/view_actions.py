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

import numpy as np

from ..commands.view import ViewError, ViewRequest

if TYPE_CHECKING:
    from .widgets.network_view import Column
    from .window import MainWindow

Handler = Callable[..., str]

# How the `layout` command names the splitters.
SPLIT_LABELS = {"columns": "columns", "left": "left column", "right": "right column"}
SPLIT_NAMES = {"columns": "the width", "left": "the left column", "right": "the right column"}


class ViewActions:
    """The window-only actions of one :class:`MainWindow`, by action name."""

    def __init__(self, window: MainWindow) -> None:
        self._window = window
        self._handlers: dict[str, Handler] = {
            "scale": self.scale,
            "probe": self.probe,
            "neuron": self.neuron,
            "layout": self.layout,
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

    # ------------------------------------------------------------ network view
    def neuron(self, column: str | None, index: int | None) -> str:
        """``neuron [column] [index]`` / ``neuron off``: highlight a neuron (replaces its
        tooltip). Without arguments: the drawn columns; with a column only: its neurons."""
        network = self._window.network
        if column == "off" and index is None:
            if network.selected_neuron is None:
                return "no neuron highlighted"
            network.select_neuron(None)
            return "neuron highlight removed"
        columns = network.columns()
        if not columns:
            raise ViewError("no network drawn yet")
        if column is None:
            width = max(len(each.title) for each in columns)
            lines = ["drawn columns ('neuron <column> <index>' highlights a neuron):"]
            lines += [
                f"  {each.title:<{width}}  {each.caption:<20}  neurons {_summary(each)}"
                for each in columns
            ]
            return "\n".join(lines)
        chosen = _column(columns, column)
        if index is None:
            return f"{chosen.title} · drawn neurons {_indices(chosen)}"
        hits = np.flatnonzero(chosen.indices == index)
        if not hits.size:
            raise ViewError(
                f"{chosen.title} has no drawn neuron {index} - drawn: {_indices(chosen)}"
            )
        network.select_neuron(chosen.title, index)
        return "\n".join(network.neuron_lines(chosen, int(hits[0])))

    # ------------------------------------------------------------------ layout
    def layout(self, panel: str | None, percent: int | None) -> str:
        """``layout [panel] [percent]`` / ``layout reset``: panel sizes (replaces dragging
        the splitters). Without arguments: every panel's share."""
        window = self._window
        if panel is None:
            return f"{self._shares()}\n'layout <panel> <percent>' resizes, 'layout reset' restores"
        panel = panel.lower()
        if panel == "reset":
            if percent is not None:
                raise ViewError("'layout reset' takes no percentage")
            window.reset_layout()
            return f"layout reset\n{self._shares()}"
        try:
            splitter = window.panel_splitter(panel)
        except ValueError as exc:
            raise ViewError(str(exc)) from None
        if percent is None:
            share = window.panel_shares()[splitter][panel]
            return f"{panel}: {share}% of {SPLIT_NAMES[splitter]}"
        window.set_panel_share(panel, percent)
        shares = window.panel_shares()[splitter]
        line = f"{SPLIT_LABELS[splitter]}  {_share_line(shares)}"
        if abs(shares[panel] - percent) > 1:
            line += " (limited by the panels' minimum sizes)"
        return line

    def _shares(self) -> str:
        width = max(len(label) for label in SPLIT_LABELS.values())
        return "\n".join(
            f"{SPLIT_LABELS[name]:<{width}}  {_share_line(shares)}"
            for name, shares in self._window.panel_shares().items()
        )


def _share_line(shares: dict[str, int]) -> str:
    return " · ".join(f"{panel} {share}%" for panel, share in shares.items())


def _normal(title: str) -> str:
    """Column titles compare without case; '-' and '_' stand for spaces."""
    return title.casefold().replace("-", " ").replace("_", " ")


def _column(columns: list[Column], name: str) -> Column:
    for column in columns:
        if _normal(column.title) == _normal(name):
            return column
    titles = ", ".join(column.title for column in columns)
    raise ViewError(f"no column {name!r} - columns: {titles}")


def _runs(indices: np.ndarray) -> list[str]:
    """Consecutive indices as ranges: [0, 1, 2, 5] -> ['0-2', '5']."""
    values = [int(value) for value in indices]
    runs: list[str] = []
    start = previous = values[0]
    for value in [*values[1:], None]:
        if value is not None and value == previous + 1:
            previous = value
            continue
        runs.append(str(start) if start == previous else f"{start}-{previous}")
        if value is not None:
            start = previous = value
    return runs


def _indices(column: Column) -> str:
    """Every drawn neuron index of ``column``, consecutive ones as ranges."""
    return ", ".join(_runs(column.indices)) if len(column.indices) else "none"


def _summary(column: Column) -> str:
    """The drawn indices in short: a range, or first to last with the count."""
    runs = _runs(column.indices) if len(column.indices) else []
    if len(runs) <= 1:
        return runs[0] if runs else "none"
    first, last = int(column.indices[0]), int(column.indices[-1])
    return f"{first} … {last} ({len(column.indices)} drawn)"

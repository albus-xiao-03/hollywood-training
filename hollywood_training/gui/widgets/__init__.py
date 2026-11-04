"""The panels of the main window."""

from .header import HeaderBar
from .hud import HudPanel, paint_bar
from .loss_chart import Axes, LossChart, decade_ticks, nice_ticks
from .network_view import Column, Link, NetworkView, Signals, build_columns
from .status_panel import StatusPanel
from .terminal import CommandLine, TerminalPanel
from .usage_panel import UsagePanel

__all__ = [
    "Axes",
    "Column",
    "CommandLine",
    "HeaderBar",
    "HudPanel",
    "Link",
    "LossChart",
    "NetworkView",
    "Signals",
    "StatusPanel",
    "TerminalPanel",
    "UsagePanel",
    "build_columns",
    "decade_ticks",
    "nice_ticks",
    "paint_bar",
]

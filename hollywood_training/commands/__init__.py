"""Runtime commands for controlling a managed training session.

The declarations and the parser (:mod:`.registry`) need no torch; the built-in command
set and the handler context are imported on first use.
"""

from importlib import import_module

from .registry import (
    Argument,
    Command,
    CommandError,
    CommandRegistry,
    CommandResult,
    split_command_line,
)

_LAZY = {
    "builtin_commands": ".builtin",
    "create_registry": ".builtin",
    "CommandContext": ".context",
    "FrontendActions": ".context",
}

__all__ = [
    "Argument",
    "Command",
    "CommandContext",
    "CommandError",
    "CommandRegistry",
    "CommandResult",
    "FrontendActions",
    "builtin_commands",
    "create_registry",
    "split_command_line",
]


def __getattr__(name: str):
    # See hollywood_training/__init__.py: torch-dependent exports load on first use.
    module = _LAZY.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(module, __name__), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(__all__)

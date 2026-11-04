"""The command language: declarations, parsing and dispatch.

A :class:`Command` declares its name, aliases, typed :class:`Argument` s and a handler
``handler(context, *arguments) -> str | None``. :class:`CommandRegistry` parses a command
line (shell-like quoting), converts and validates the arguments, calls the handler and
wraps the outcome in a :class:`CommandResult`. Ordinary errors never reach the frontend:
they become failed results with a helpful message (``KeyboardInterrupt`` / ``SystemExit``
still propagate).
"""

from __future__ import annotations

import difflib
import os
import shlex
from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import dataclass
from typing import Any


class CommandError(Exception):
    """An invalid command line, or a request the session cannot honour right now."""


@dataclass(frozen=True, slots=True)
class Argument:
    """One positional argument of a command."""

    name: str
    converter: Callable[[str], Any] = str
    help: str = ""
    optional: bool = False
    default: Any = None
    choices: tuple[str, ...] | None = None

    @property
    def usage(self) -> str:
        return f"[{self.name}]" if self.optional else f"<{self.name}>"

    def convert(self, raw: str) -> Any:
        if self.choices is not None and raw not in self.choices:
            raise CommandError(f"{self.name} must be one of: {', '.join(self.choices)}")
        try:
            return self.converter(raw)
        except (TypeError, ValueError) as exc:
            raise CommandError(f"invalid {self.name} {raw!r}: {exc}") from None


Handler = Callable[..., "str | None"]


@dataclass(frozen=True, slots=True)
class Command:
    """A named runtime command."""

    name: str
    summary: str
    handler: Handler
    arguments: tuple[Argument, ...] = ()
    aliases: tuple[str, ...] = ()
    category: str = "general"
    details: str = ""

    def __post_init__(self) -> None:
        for word in (self.name, *self.aliases):
            if not word or any(ch.isspace() for ch in word) or word != word.lower():
                raise ValueError(f"invalid command name {word!r}: use one lowercase word")
        optional_seen = False
        for argument in self.arguments:
            if optional_seen and not argument.optional:
                raise ValueError(f"{self.name}: required argument after an optional one")
            optional_seen = optional_seen or argument.optional

    @property
    def usage(self) -> str:
        return " ".join([self.name, *(argument.usage for argument in self.arguments)])

    def bind(self, tokens: Sequence[str]) -> list[Any]:
        """Convert raw argument tokens into handler arguments."""
        required = [argument for argument in self.arguments if not argument.optional]
        if len(tokens) < len(required):
            missing = required[len(tokens)].name
            raise CommandError(f"missing {missing} - usage: {self.usage}")
        if len(tokens) > len(self.arguments):
            raise CommandError(f"too many arguments - usage: {self.usage}")
        return [
            argument.convert(tokens[index]) if index < len(tokens) else argument.default
            for index, argument in enumerate(self.arguments)
        ]


@dataclass(frozen=True, slots=True)
class CommandResult:
    """Outcome of executing one command line."""

    ok: bool
    output: str
    command: Command | None = None


def split_command_line(line: str) -> list[str]:
    """Split like a shell. Backslashes are kept on Windows so paths work unquoted."""
    posix = os.name != "nt"
    try:
        tokens = shlex.split(line, posix=posix)
    except ValueError as exc:
        raise CommandError(f"cannot parse command: {exc}") from None
    if not posix:
        tokens = [t[1:-1] if len(t) >= 2 and t[0] == t[-1] and t[0] in "\"'" else t for t in tokens]
    return tokens


class CommandRegistry:
    """A set of commands addressable by name or alias."""

    def __init__(self, commands: Iterable[Command] = ()) -> None:
        self._commands: dict[str, Command] = {}
        self._aliases: dict[str, str] = {}
        for command in commands:
            self.register(command)

    def register(self, command: Command, *, replace: bool = False) -> Command:
        """Add ``command``; with ``replace`` it may take over existing names."""
        words = (command.name, *command.aliases)
        for word in words:
            owner = self._aliases.get(word, word if word in self._commands else None)
            if owner is not None and not replace:
                raise ValueError(f"command name {word!r} is already used by {owner!r}")
            if owner is not None:
                self.unregister(owner)
        self._commands[command.name] = command
        for alias in command.aliases:
            self._aliases[alias] = command.name
        return command

    def unregister(self, name: str) -> None:
        command = self.get(name)
        if command is None:
            return
        del self._commands[command.name]
        for alias in command.aliases:
            self._aliases.pop(alias, None)

    def get(self, name: str) -> Command | None:
        name = name.lower()
        return self._commands.get(self._aliases.get(name, name))

    def __contains__(self, name: object) -> bool:
        return isinstance(name, str) and self.get(name) is not None

    def __iter__(self) -> Iterator[Command]:
        return iter(list(self._commands.values()))

    def __len__(self) -> int:
        return len(self._commands)

    def names(self, *, include_aliases: bool = False) -> list[str]:
        names = list(self._commands)
        if include_aliases:
            names += list(self._aliases)
        return sorted(names)

    def complete(self, prefix: str) -> list[str]:
        """Command names and aliases starting with ``prefix``."""
        prefix = prefix.lower()
        return [name for name in self.names(include_aliases=True) if name.startswith(prefix)]

    def suggest(self, name: str) -> list[str]:
        return difflib.get_close_matches(name.lower(), self.names(include_aliases=True), n=3)

    def parse(self, line: str) -> tuple[Command, list[Any]]:
        tokens = split_command_line(line)
        if not tokens:
            raise CommandError("empty command")
        command = self.get(tokens[0])
        if command is None:
            hint = self.suggest(tokens[0])
            suffix = f" - did you mean {', '.join(hint)}?" if hint else " - type 'help'"
            raise CommandError(f"unknown command {tokens[0]!r}{suffix}")
        return command, command.bind(tokens[1:])

    def execute(self, line: str, context: Any) -> CommandResult:
        """Parse and run ``line``; ordinary errors become a failed result."""
        command: Command | None = None
        try:
            command, arguments = self.parse(line)
            output = command.handler(context, *arguments)
        except CommandError as exc:
            return CommandResult(False, str(exc), command)
        except Exception as exc:  # a handler bug must not take the frontend down
            return CommandResult(False, f"{type(exc).__name__}: {exc}", command)
        return CommandResult(True, output or "", command)

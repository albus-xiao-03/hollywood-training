# Command Registry

`hollywood_training/commands/registry.py` · part of [Commands](Commands)

The command language: declarations, parsing and dispatch. A `Command` declares its name,
aliases, typed `Argument`s and a handler `handler(context, *arguments) -> str | None`.
`CommandRegistry` parses a command line with shell-like quoting, converts and validates
the arguments, calls the handler and wraps the outcome in a `CommandResult`. Ordinary
errors become failed results with a helpful message instead of exceptions;
`KeyboardInterrupt` and `SystemExit` still propagate. The module needs no torch.

## Responsibilities

* Validate command declarations (names, argument order).
* Split command lines (`split_command_line`) and bind tokens to typed arguments.
* Look up names and aliases, complete prefixes, suggest close matches.
* Run a handler and turn every outcome into a `CommandResult`.

## API

```python
class CommandError(Exception): ...
```

An invalid command line, or a request the session cannot honour right now. Its message
is shown to the user. Exported as `ht.CommandError`.

```python
@dataclass(frozen=True, slots=True)
class Argument:
    name: str
    converter: Callable[[str], Any] = str
    help: str = ""
    optional: bool = False
    default: Any = None
    choices: tuple[str, ...] | None = None

    @property
    def usage(self) -> str: ...  # "<name>", or "[name]" when optional
    def convert(self, raw: str) -> Any: ...
```

One positional argument. `convert` raises `CommandError` with `"<name> must be one of: a,
b"` when `choices` is set and `raw` is not among them (checked before converting), and
with `"invalid <name> '<raw>': <error>"` when `converter` raises `TypeError` or `ValueError`.
Exported as `ht.Argument`.

```python
Handler = Callable[..., "str | None"]

@dataclass(frozen=True, slots=True)
class Command:
    name: str
    summary: str
    handler: Handler
    arguments: tuple[Argument, ...] = ()
    aliases: tuple[str, ...] = ()
    category: str = "general"
    details: str = ""

    @property
    def usage(self) -> str: ...  # e.g. "step [epochs]"
    def bind(self, tokens: Sequence[str]) -> list[Any]: ...
```

Construction raises `ValueError` when the name or an alias is empty, contains whitespace
or is not lowercase (`"invalid command name 'X': use one lowercase word"`), or when a
required argument follows an optional one. `bind` converts the argument tokens, using
`default` for omitted optional arguments. It raises `CommandError` with
`"missing <name> - usage: …"` or `"too many arguments - usage: …"`. Exported as
`ht.Command`.

```python
@dataclass(frozen=True, slots=True)
class CommandResult:
    ok: bool
    output: str
    command: Command | None = None
```

`command` is the parsed command, or `None` when the line could not be parsed or bound.

```python
def split_command_line(line: str) -> list[str]: ...
```

Splits like a shell with `shlex.split`, in POSIX mode except on Windows (`os.name ==
"nt"`), where backslashes are kept and one pair of surrounding quotes is stripped from
each token. Raises `CommandError("cannot parse command: …")`, e.g. for an unclosed quote.

```python
class CommandRegistry:
    def __init__(self, commands: Iterable[Command] = ()) -> None: ...
    def register(self, command: Command, *, replace: bool = False) -> Command: ...
    def unregister(self, name: str) -> None: ...
    def get(self, name: str) -> Command | None: ...
    def __contains__(self, name: object) -> bool: ...
    def __iter__(self) -> Iterator[Command]: ...
    def __len__(self) -> int: ...
    def names(self, *, include_aliases: bool = False) -> list[str]: ...
    def complete(self, prefix: str) -> list[str]: ...
    def suggest(self, name: str) -> list[str]: ...
    def parse(self, line: str) -> tuple[Command, list[Any]]: ...
    def execute(self, line: str, context: Any) -> CommandResult: ...
```

| Member | Behaviour |
|---|---|
| `register` | Adds `command` and returns it. A name or alias already in use raises `ValueError("command name 'x' is already used by 'y'")`; with `replace=True` the owning command is unregistered first, with all its aliases. |
| `unregister` | Removes the command with this name or alias; unknown names are ignored. |
| `get`, `in` | Case-insensitive lookup by name or alias. |
| `iter`, `len` | Commands in registration order; aliases are not counted. |
| `names` | Sorted command names, plus aliases with `include_aliases=True`. |
| `complete` | Sorted names and aliases starting with `prefix` (case-insensitive). |
| `suggest` | Up to three close names or aliases (`difflib.get_close_matches`). |
| `parse` | Splits the line and binds the arguments. Raises `CommandError`: `"empty command"`, or `"unknown command 'x' - did you mean a, b?"` (`" - type 'help'"` when nothing is close). |
| `execute` | Parses and runs the handler with `context` as first argument. Returns `CommandResult(True, output or "", command)`; a `CommandError` gives `CommandResult(False, message, command)`, any other `Exception` gives `CommandResult(False, "<Type>: <message>", command)`. |

## How it works

`execute(line, context)` runs: `split_command_line` → look up the first token →
`Command.bind` (each `Argument.convert`) → `handler(context, *values)`. The registry never
inspects `context`; [`CommandContext`](Commands-Context) is what the frontends pass.

## Used by / uses

* Standard library only (`shlex`, `difflib`, `os`).
* [Built-in Commands](Commands-Builtin) declare their commands with `Command` and `Argument`;
  `create_registry` builds the session's registry.
* [Training Process](Training-Process): `@process.command(...)` creates `Command` objects
  with category `"custom"`.
* Frontends call `execute` ([GUI In-Process Frontend](GUI-In-Process-Frontend),
  [GUI Remote Frontend](GUI-Remote-Frontend)) and `complete` / `names(include_aliases=True)`
  for Tab completion ([GUI Terminal](GUI-Terminal)).

## Example

```python
from hollywood_training.commands import Argument, Command, CommandRegistry

def add(context, a, b):
    return f"{a + b:g}"

registry = CommandRegistry([
    Command("add", "add two numbers", add,
            arguments=(Argument("a", float), Argument("b", float, optional=True, default=1.0))),
])
registry.execute("add 2", None)            # CommandResult(ok=True, output='3', command=...)
registry.execute("add 2 3.5", None).output # '5.5'
registry.execute("ad 2", None).output      # "unknown command 'ad' - did you mean add?"
registry.execute("add x", None).output     # "invalid a 'x': could not convert string to float: 'x'"
```

## Notes

* `execute` catches `Exception`, so a buggy handler cannot take the frontend down;
  `KeyboardInterrupt` and `SystemExit` still propagate.
* A handler's return value is passed on unchanged (`output or ""`); return a `str` or
  `None`.
* `register(..., replace=True)` removes the *whole* command that owned a clashing name or
  alias, not just that word.

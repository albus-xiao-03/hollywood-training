"""R1 requirement 9 - "The package **MUST** be **COMPLETELY** decompoised into modules with
clear architecture."

"Completely decomposed": the package is split into subpackages by concern, each module has
one documented purpose and none is a catch-all ("god module": more than 1000 lines here).
"Clear architecture": the dependencies are one-way and explainable -

* no import cycle between the subpackages (module-level imports);
* the core (tracking, training, synchronisation, commands, monitoring, formatting, console)
  never imports the GUI; only ``runner.py``, the composition root, does - lazily;
* the data that crosses from training to a frontend needs no torch, and the window process
  runs without torch; importing the package loads no GUI toolkit;
* every module imports on its own (no hidden ordering).
"""

from __future__ import annotations

import ast
import textwrap
from collections import defaultdict
from pathlib import Path

import pytest
from htspec import paths
from htspec.subproc import run_python

pytestmark = pytest.mark.req("R1.9")

SUBPACKAGES = ("tracking", "training", "sync", "commands", "monitor", "gui")
GOD_MODULE_LINES = 1000
TORCH_FREE = (
    "hollywood_training.sync.events", "hollywood_training.sync.session",
    "hollywood_training.sync.channel", "hollywood_training.sync.control",
    "hollywood_training.sync.pump", "hollywood_training.tracking.architecture",
    "hollywood_training.tracking.weights", "hollywood_training.tracking.roles",
    "hollywood_training.monitor.usage", "hollywood_training.commands.registry",
    "hollywood_training.commands.context", "hollywood_training.commands.view",
    "hollywood_training.formatting",
)  # fmt: skip


def modules() -> list[Path]:
    return sorted(paths.PACKAGE_DIR.rglob("*.py"))


def dotted(path: Path) -> str:
    parts = list(path.relative_to(paths.PROJECT_ROOT).with_suffix("").parts)
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


def unit_of(name: str) -> str:
    """The subpackage (or top-level module) a dotted module name belongs to."""
    parts = name.split(".")
    return parts[1] if len(parts) > 1 else "__init__"


def resolve(node: ast.ImportFrom, package: str) -> set[str]:
    """Absolute names a ``from ... import ...`` refers to (``from .. import formatting``
    is the module ``<package>.formatting``)."""
    if node.level:
        base = package.split(".")
        base = base[: len(base) - (node.level - 1)]
        module = ".".join(base + ([node.module] if node.module else []))
    else:
        module = node.module or ""
    if node.module is None:
        return {f"{module}.{alias.name}" for alias in node.names}
    return {module}


def module_level_imports(path: Path) -> set[str]:
    """Package-internal modules imported at module level (not in functions, not under
    ``if TYPE_CHECKING:``), as absolute dotted names."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    package = dotted(path) if path.name == "__init__.py" else dotted(path).rsplit(".", 1)[0]
    found: set[str] = set()

    def visit(body) -> None:
        for node in body:
            if isinstance(node, ast.If) and "TYPE_CHECKING" in ast.unparse(node.test):
                continue
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                continue
            if isinstance(node, ast.Import):
                found.update(a.name for a in node.names if a.name.startswith("hollywood_training"))
            elif isinstance(node, ast.ImportFrom):
                names = resolve(node, package)
                found.update(n for n in names if n.startswith("hollywood_training"))
            elif isinstance(node, (ast.If, ast.Try, ast.With)):
                visit(getattr(node, "body", []))
                visit(getattr(node, "orelse", []))
                visit(getattr(node, "finalbody", []))
                for handler in getattr(node, "handlers", []):
                    visit(handler.body)

    visit(tree.body)
    return found


def all_imports(path: Path) -> set[str]:
    """Every package-internal or GUI-toolkit import anywhere in the module (incl. functions),
    except under ``if TYPE_CHECKING:``."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    package = dotted(path) if path.name == "__init__.py" else dotted(path).rsplit(".", 1)[0]
    guarded = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.If) and "TYPE_CHECKING" in ast.unparse(node.test):
            guarded.update(id(child) for child in ast.walk(node))
    found = set()
    for node in ast.walk(tree):
        if id(node) in guarded:
            continue
        if isinstance(node, ast.Import):
            found.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            found.update(resolve(node, package))
    return found


# ---------------------------------------------------------------- decomposition
def test_the_package_is_split_into_subpackages_by_concern():
    for name in SUBPACKAGES:
        init = paths.PACKAGE_DIR / name / "__init__.py"
        assert init.is_file(), name
        assert ast.get_docstring(ast.parse(init.read_text(encoding="utf-8"))), name
        assert len(list((paths.PACKAGE_DIR / name).glob("*.py"))) >= 2, name


def module_id(path: Path) -> str:
    return path.relative_to(paths.PACKAGE_DIR).as_posix()


@pytest.mark.parametrize("path", modules(), ids=module_id)
def test_every_module_documents_its_purpose(path):
    docstring = ast.get_docstring(ast.parse(path.read_text(encoding="utf-8")))
    assert docstring and len(docstring.split()) >= 3, path.name


@pytest.mark.parametrize("path", modules(), ids=module_id)
def test_no_module_is_a_catch_all(path):
    lines = len(path.read_text(encoding="utf-8").splitlines())
    assert lines <= GOD_MODULE_LINES, f"{path.name}: {lines} lines"


# ------------------------------------------------------------------ dependencies
def test_there_is_no_import_cycle_between_the_subpackages():
    graph: dict[str, set[str]] = defaultdict(set)
    for path in modules():
        source = unit_of(dotted(path))
        if source == "__init__":
            continue  # the root re-exports; it is not a layer of its own
        for imported in module_level_imports(path):
            target = unit_of(imported)
            if target not in (source, "__init__"):
                graph[source].add(target)
    cycle = _find_cycle(graph)
    assert cycle is None, f"import cycle: {' -> '.join(cycle)} (graph: {dict(graph)})"


def _find_cycle(graph: dict[str, set[str]]) -> list[str] | None:
    state: dict[str, int] = {}
    stack: list[str] = []

    def visit(node: str) -> list[str] | None:
        state[node] = 1
        stack.append(node)
        for nxt in sorted(graph.get(node, ())):
            if state.get(nxt) == 1:
                return [*stack[stack.index(nxt) :], nxt]
            if state.get(nxt) is None:
                found = visit(nxt)
                if found:
                    return found
        stack.pop()
        state[node] = 2
        return None

    for node in sorted(graph):
        if state.get(node) is None:
            found = visit(node)
            if found:
                return found
    return None


@pytest.mark.parametrize(
    "path",
    [p for p in modules() if unit_of(dotted(p)) != "gui" and p.name != "runner.py"],
    ids=module_id,
)
def test_the_core_never_imports_the_gui(path):
    gui = [
        name for name in all_imports(path)
        if name.startswith(("hollywood_training.gui", "PySide6", "PyQt", "shiboken"))
    ]  # fmt: skip
    assert gui == []


def test_only_the_composition_root_reaches_the_gui_and_only_lazily():
    runner = paths.PACKAGE_DIR / "runner.py"
    eager = {n for n in module_level_imports(runner) if n.startswith("hollywood_training.gui")}
    assert eager == set()
    assert any(n.startswith("hollywood_training.gui") for n in all_imports(runner))


# ------------------------------------------------------------------- run-time
@pytest.mark.subprocess
def test_the_data_crossing_to_a_frontend_needs_no_torch():
    code = textwrap.dedent(
        f"""
        import importlib, sys
        sys.modules["torch"] = None  # any "import torch" fails from here on
        for name in {list(TORCH_FREE)!r}:
            importlib.import_module(name)
        print("TORCH-FREE OK")
        """
    )
    completed = run_python(["-c", code], timeout=120)
    assert completed.returncode == 0, completed.stderr[-3000:]
    assert "TORCH-FREE OK" in completed.stdout


@pytest.mark.subprocess
def test_the_window_process_runs_without_torch():
    code = textwrap.dedent(
        """
        import sys
        sys.modules["torch"] = None
        import hollywood_training.gui.window_process
        import hollywood_training.gui.window
        print("WINDOW-WITHOUT-TORCH OK")
        """
    )
    completed = run_python(["-c", code], timeout=120)
    assert completed.returncode == 0, completed.stderr[-3000:]
    assert "WINDOW-WITHOUT-TORCH OK" in completed.stdout


@pytest.mark.subprocess
def test_importing_the_package_loads_no_gui_toolkit():
    code = textwrap.dedent(
        """
        import sys
        import hollywood_training as ht
        ht.track, ht.TrainingProcess, ht.run, ht.NetworkTracker  # the public API
        print("GUI-MODULES", sorted(m for m in sys.modules if m.startswith("PySide6")))
        """
    )
    completed = run_python(["-c", code], timeout=120)
    assert completed.returncode == 0, completed.stderr[-3000:]
    assert "GUI-MODULES []" in completed.stdout


@pytest.mark.subprocess
@pytest.mark.slow
@pytest.mark.timeout(360)
def test_every_module_imports_on_its_own():
    names = [dotted(path) for path in modules()]
    code = textwrap.dedent(
        f"""
        import importlib, sys, traceback
        failed = []
        for name in {names!r}:
            for loaded in [m for m in sys.modules if m.startswith("hollywood_training")]:
                del sys.modules[loaded]  # as if this module were the first one imported
            try:
                importlib.import_module(name)
            except Exception:
                failed.append((name, traceback.format_exc(limit=3)))
        print("FAILED", len(failed))
        for name, error in failed:
            print(name, error)
        """
    )
    completed = run_python(["-c", code], timeout=300)
    assert completed.returncode == 0, completed.stderr[-3000:]
    assert "FAILED 0" in completed.stdout, completed.stdout[-4000:]

"""R9: the package is completely decomposed into modules with a clear architecture:
one subpackage per concern, documented modules, no import cycles, dependencies that only
point downward (wiki: Architecture), and a GUI that is separate from the training core."""

from __future__ import annotations

import ast
import importlib
import json
import textwrap

import pytest

from .helpers import PACKAGE_DIR, module_name, package_sources, parse_source, run_python

pytestmark = pytest.mark.spec("R9")

SUBPACKAGES = ("tracking", "training", "sync", "commands", "monitor", "gui")
CORE = ("tracking", "training", "sync", "commands", "monitor", "formatting", "console")

# "Layers depend only downward" (wiki/Architecture.md): what each layer must never import.
FORBIDDEN_UPWARD = {
    "tracking": ("sync", "training", "commands", "gui", "console", "runner"),
    "monitor": ("tracking", "sync", "training", "commands", "gui", "console", "runner"),
    "formatting": ("tracking", "sync", "training", "commands", "gui", "console", "runner"),
    "sync": ("training", "commands", "gui", "console", "runner"),
    "training": ("gui", "console", "runner"),
    "commands": ("gui", "console", "runner"),
    "console": ("gui", "runner"),
    "gui": ("runner",),
}


def _is_type_checking(test: ast.expr) -> bool:
    return (isinstance(test, ast.Name) and test.id == "TYPE_CHECKING") or (
        isinstance(test, ast.Attribute) and test.attr == "TYPE_CHECKING"
    )


class _ImportCollector(ast.NodeVisitor):
    """Package-internal modules a module imports when it is imported (function bodies and
    ``if TYPE_CHECKING:`` blocks run later or never, so they are skipped)."""

    def __init__(self, module: str, is_package: bool) -> None:
        self.package = module if is_package else module.rpartition(".")[0]
        self.found: set[str] = set()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        return None

    visit_AsyncFunctionDef = visit_FunctionDef  # noqa: N815 - ast.NodeVisitor naming

    def visit_If(self, node: ast.If) -> None:
        if _is_type_checking(node.test):
            for child in node.orelse:
                self.visit(child)
            return
        self.generic_visit(node)

    def visit_Import(self, node: ast.Import) -> None:
        self.found.update(alias.name for alias in node.names)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if node.level:
            parts = self.package.split(".")
            base_parts = parts[: len(parts) - (node.level - 1)]
            base = ".".join(base_parts + ([node.module] if node.module else []))
        else:
            base = node.module or ""
        self.found.add(base)
        self.found.update(f"{base}.{alias.name}" for alias in node.names)


@pytest.fixture(scope="module")
def graph() -> dict[str, set[str]]:
    modules = {module_name(path): path for path in package_sources()}
    edges: dict[str, set[str]] = {}
    for name, path in modules.items():
        collector = _ImportCollector(name, path.name == "__init__.py")
        collector.visit(parse_source(path))
        edges[name] = {target for target in collector.found if target in modules} - {name}
    return edges


def _layer(module: str) -> str:
    parts = module.split(".")
    return parts[1] if len(parts) > 1 else ""


# ------------------------------------------------------------------ structure
def test_the_package_is_split_into_one_subpackage_per_concern():
    for name in SUBPACKAGES:
        assert (PACKAGE_DIR / name / "__init__.py").is_file(), f"missing subpackage {name}"
    assert (PACKAGE_DIR / "gui" / "widgets" / "__init__.py").is_file()
    assert (PACKAGE_DIR / "runner.py").is_file()  # the management method


def test_every_module_explains_itself():
    undocumented = [
        module_name(path) for path in package_sources() if not ast.get_docstring(parse_source(path))
    ]
    assert undocumented == []


def test_there_are_no_circular_imports(graph):
    cycles: list[list[str]] = []
    state: dict[str, int] = {}  # 1 = on the stack, 2 = done

    def visit(node: str, stack: list[str]) -> None:
        state[node] = 1
        stack.append(node)
        for target in sorted(graph[node]):
            if state.get(target) == 1:
                cycles.append([*stack[stack.index(target) :], target])
            elif target not in state:
                visit(target, stack)
        stack.pop()
        state[node] = 2

    for node in sorted(graph):
        if node not in state:
            visit(node, [])
    assert cycles == []


def test_dependencies_only_point_downward(graph):
    violations = []
    for module, targets in graph.items():
        for target in targets:
            if _layer(target) in FORBIDDEN_UPWARD.get(_layer(module), ()):
                violations.append(f"{module} -> {target}")
    assert violations == []


def test_the_training_core_knows_nothing_about_the_gui():
    offenders = []
    for path in package_sources():
        module = module_name(path)
        if _layer(module) not in CORE:
            continue
        for node in ast.walk(parse_source(path)):  # anywhere, even inside functions
            names = []
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [("." * node.level) + (node.module or "")]
            for name in names:
                if name.startswith("PySide6") or ".gui" in name or name.startswith("gui"):
                    offenders.append(f"{module}: {name}")
    assert offenders == []


# ---------------------------------------------------------- documented contracts
_PROBE = textwrap.dedent(
    """
    import importlib, json, sys
    for name in sys.argv[1:]:
        importlib.import_module(name)
    print(json.dumps({"torch": "torch" in sys.modules,
                      "qt": any(m.startswith("PySide6") for m in sys.modules)}))
    """
)


def _probe(tmp_path, *modules: str) -> dict:
    script = tmp_path / "probe.py"
    script.write_text(_PROBE, encoding="utf-8")
    completed = run_python([str(script), *modules], timeout=120)
    assert completed.returncode == 0, completed.stderr
    return json.loads(completed.stdout.strip().splitlines()[-1])


def test_the_window_process_does_not_load_pytorch(tmp_path):
    """README / wiki: the window runs in a light process that imports no torch."""
    assert _probe(tmp_path, "hollywood_training.gui.window_process")["torch"] is False


def test_the_plain_data_modules_do_not_load_pytorch(tmp_path):
    """wiki Architecture, "Data contract": what crosses to a frontend is torch-free data."""
    report = _probe(
        tmp_path,
        "hollywood_training.tracking.architecture",
        "hollywood_training.tracking.weights",
        "hollywood_training.tracking.roles",
        "hollywood_training.sync",
        "hollywood_training.monitor.usage",
        "hollywood_training.formatting",
        "hollywood_training.commands.registry",
        "hollywood_training.commands.context",
    )
    assert report == {"torch": False, "qt": False}


def test_every_module_can_be_imported(tmp_path):
    names = sorted(module_name(path) for path in package_sources())
    report = _probe(tmp_path, *names)
    assert report["torch"] is True  # the training side, of course, uses torch


@pytest.mark.parametrize(
    "package",
    ["hollywood_training", *(f"hollywood_training.{name}" for name in SUBPACKAGES)],
)
def test_every_exported_name_resolves(package):
    module = importlib.import_module(package)
    exported = getattr(module, "__all__", ())
    assert exported, f"{package} declares no public API"
    missing = [name for name in exported if not hasattr(module, name)]
    assert missing == []

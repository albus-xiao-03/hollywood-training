"""R8: no "dark magic" - the package never retrieves or analyses the user's code, never
traces or patches the interpreter or PyTorch, and works only through public objects."""

from __future__ import annotations

import ast
import functools
import inspect
import sys
import textwrap

import pytest
import torch
from torch import nn

import hollywood_training as ht

from .helpers import TinyNet, build_process, module_name, package_sources, parse_source, xor_data

pytestmark = pytest.mark.spec("R8")

FORBIDDEN_MODULES = {
    "inspect": "retrieves source code, signatures and frames of user functions",
    "dis": "disassembles bytecode",
    "ast": "parses source code",
    "tokenize": "tokenizes source code",
    "linecache": "reads source lines",
    "gc": "scans live objects",
    "ctypes": "pokes interpreter memory",
    "code": "compiles and runs code",
    "codeop": "compiles code",
    "trace": "traces execution",
    "bdb": "debugger hooks",
    "pdb": "debugger hooks",
    "symtable": "analyses source code",
}
FORBIDDEN_ATTRIBUTES = {
    "__code__", "__globals__", "__closure__", "__defaults__", "__kwdefaults__",
    "co_code", "co_names", "co_varnames", "co_consts", "co_filename", "co_firstlineno",
    "f_locals", "f_globals", "f_back", "f_code", "f_lineno", "gi_frame", "tb_frame",
    "_getframe", "settrace", "setprofile", "meta_path", "path_hooks",
}  # fmt: skip
FORBIDDEN_CALLS = {"eval", "exec", "compile", "__import__", "breakpoint", "setattr", "delattr"}
GLOBAL_HOOKS = {
    "register_module_forward_hook",
    "register_module_forward_pre_hook",
    "register_module_full_backward_hook",
    "register_module_full_backward_pre_hook",
    "register_module_backward_hook",
}


def _imports(tree: ast.Module) -> set[str]:
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            names.add(node.module.split(".")[0])
    return names


def _imported_names(tree: ast.Module) -> set[str]:
    """Names bound by absolute imports (modules and objects from other packages)."""
    bound: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            bound.update((alias.asname or alias.name).split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0:
            bound.update(alias.asname or alias.name for alias in node.names)
    return bound


def _root_name(node: ast.AST) -> str | None:
    while isinstance(node, (ast.Attribute, ast.Subscript, ast.Call)):
        node = node.value if not isinstance(node, ast.Call) else node.func
    return node.id if isinstance(node, ast.Name) else None


@pytest.fixture(scope="module")
def sources() -> dict[str, ast.Module]:
    return {module_name(path): parse_source(path) for path in package_sources()}


# -------------------------------------------------------------------- static
def test_no_module_imports_code_introspection_tools(sources):
    found = {
        module: sorted(set(FORBIDDEN_MODULES) & _imports(tree)) for module, tree in sources.items()
    }
    assert {module: names for module, names in found.items() if names} == {}


def test_no_code_object_frame_or_interpreter_hook_is_touched(sources):
    found: dict[str, list[str]] = {}
    for module, tree in sources.items():
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute) and node.attr in FORBIDDEN_ATTRIBUTES:
                found.setdefault(module, []).append(f"line {node.lineno}: .{node.attr}")
    assert found == {}


def test_no_dynamic_code_execution(sources):
    found: dict[str, list[str]] = {}
    for module, tree in sources.items():
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id in FORBIDDEN_CALLS
            ):
                found.setdefault(module, []).append(f"line {node.lineno}: {node.func.id}()")
    assert found == {}


def test_nothing_from_other_libraries_is_monkey_patched(sources):
    """No assignment to an attribute of an imported module or class (torch, Qt, ...)."""
    found: dict[str, list[str]] = {}
    for module, tree in sources.items():
        imported = _imported_names(tree)
        for node in ast.walk(tree):
            targets = []
            if isinstance(node, ast.Assign):
                targets = node.targets
            elif isinstance(node, (ast.AugAssign, ast.AnnAssign)):
                targets = [node.target]
            elif isinstance(node, ast.Delete):
                targets = node.targets
            for target in targets:
                if isinstance(target, ast.Attribute) and _root_name(target) in imported:
                    found.setdefault(module, []).append(f"line {node.lineno}")
    assert found == {}


def test_no_global_module_hooks_are_installed(sources):
    """Global hooks would change every network of the program, not just the tracked one."""
    found = {
        module: sorted(
            node.attr
            for node in ast.walk(tree)
            if isinstance(node, ast.Attribute) and node.attr in GLOBAL_HOOKS
        )
        for module, tree in sources.items()
    }
    assert {module: names for module, names in found.items() if names} == {}


# --------------------------------------------------------------- behavioural
def _native_step(model, optimizer, loss_fn, batch):
    inputs, targets = batch
    optimizer.zero_grad()
    loss = loss_fn(model(inputs), targets)
    loss.backward()
    optimizer.step()
    return loss


class CallableStep:
    def __init__(self, model, optimizer, loss_fn) -> None:
        self.arguments = (model, optimizer, loss_fn)

    def __call__(self, batch):
        return _native_step(*self.arguments, batch)


def _exec_defined_step(model, optimizer, loss_fn):
    namespace = {"model": model, "optimizer": optimizer, "loss_fn": loss_fn}
    source = textwrap.dedent(
        """
        def step(batch):
            inputs, targets = batch
            optimizer.zero_grad()
            loss = loss_fn(model(inputs), targets)
            loss.backward()
            optimizer.step()
            return loss
        """
    )
    exec(compile(source, "<no file>", "exec"), namespace)  # a function without source file
    return namespace["step"]


@pytest.mark.parametrize("kind", ["lambda", "partial", "callable-object", "no-source"])
def test_steps_without_retrievable_source_work(kind):
    torch.manual_seed(0)
    model = TinyNet()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    loss_fn = nn.CrossEntropyLoss()
    step = {
        "lambda": lambda: lambda batch: _native_step(model, optimizer, loss_fn, batch),
        "partial": lambda: functools.partial(_native_step, model, optimizer, loss_fn),
        "callable-object": lambda: CallableStep(model, optimizer, loss_fn),
        "no-source": lambda: _exec_defined_step(model, optimizer, loss_fn),
    }[kind]()
    process = ht.TrainingProcess(
        model, optimizer, [tuple(xor_data(16).tensors)], epochs=3, train_step=step
    )
    result = ht.run(process, gui=False)
    assert result.epochs_completed == 3
    assert result.train_losses[-1] < result.train_losses[0]


def _forbid_source_access(monkeypatch) -> list[str]:
    calls: list[str] = []

    def forbidden(name):
        def refuse(*args, **kwargs):
            calls.append(name)
            raise AssertionError(f"the package called {name}")

        return refuse

    # the calls that retrieve a function's definition (pytest itself tolerates their failure)
    for name in ("getsource", "getsourcelines", "findsource"):
        monkeypatch.setattr(inspect, name, forbidden(f"inspect.{name}"))
    return calls


def test_the_users_code_is_never_read(monkeypatch):
    calls = _forbid_source_access(monkeypatch)
    bundle = build_process(epochs=2)
    result = ht.run(bundle.process, gui=False)
    assert result.epochs_completed == 2 and calls == []


@pytest.mark.gui
def test_the_users_code_is_never_read_with_the_window(qapp, monkeypatch):
    calls = _forbid_source_access(monkeypatch)
    bundle = build_process(epochs=2)
    result = ht.run(bundle.process, isolate_gui=False, close_on_finish=True)
    assert result.epochs_completed == 2 and calls == []


def test_no_tracer_or_profiler_runs_while_the_user_code_runs():
    outside = (sys.gettrace(), sys.getprofile())
    inside = []
    bundle = build_process(epochs=2, native_steps=False)

    @bundle.process.train_step
    def train_step(batch):
        inside.append((sys.gettrace(), sys.getprofile()))
        return _native_step(bundle.model, bundle.optimizer, bundle.loss_fn, batch)

    ht.run(bundle.process, gui=False)
    assert inside and set(inside) == {outside}


def test_the_users_functions_are_left_exactly_as_written():
    bundle = build_process(epochs=2, native_steps=False)

    def train_step(batch, scale=1.0):
        return _native_step(bundle.model, bundle.optimizer, bundle.loss_fn, batch) * scale

    def hook(record):
        return None

    snapshot = {
        f: (f.__code__, f.__defaults__, dict(vars(f)), f.__name__, f.__qualname__)
        for f in (train_step, hook)
    }
    bundle.process.train_step(train_step)
    bundle.process.on_epoch_end(hook)
    ht.run(bundle.process, gui=False)
    for function, before in snapshot.items():
        after = (
            function.__code__,
            function.__defaults__,
            dict(vars(function)),
            function.__name__,
            function.__qualname__,
        )
        assert after == before

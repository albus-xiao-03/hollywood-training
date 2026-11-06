"""R1 requirement 8 - "**ALL** `dark magic` Python methods are **STRICTLY PROHIBITED**.
Example of `dark magic`: Dynamically retrive the definition of user's training function,
analyze user's intention with function definition by `regex`, then determine the necessary
information of training process."

* statically (every module of the package, by its syntax tree): no source retrieval
  (``inspect.getsource`` & co., ``linecache``), no frame / code-object introspection
  (``sys._getframe``, ``__code__``, ``co_*``, ``f_locals``, ...), no tracing / profiling
  hooks, no ``exec`` / ``eval`` / ``compile``, no ``ast`` / ``dis`` / ``tokenize``, no garbage
  collector reference walking, no import hooks, no monkeypatching of torch or builtins;
* by behaviour: steps that have **no source at all** (generated with ``exec``, partials,
  callable objects, bound methods, lambdas) train exactly like ordinary functions, also
  while every source-retrieval function is booby-trapped; the user's functions and classes
  are left untouched by a run.
"""

from __future__ import annotations

import ast
import functools
import inspect
import linecache
import sys
import threading

import pytest
import torch
from htspec import paths
from htspec.models import TinyNet, loader
from torch import nn

import hollywood_training as ht

pytestmark = pytest.mark.req("R1.8")

FORBIDDEN_ATTRIBUTES = {
    ("inspect", name)
    for name in (
        "getsource", "getsourcelines", "getsourcefile", "getfile", "findsource", "getcomments",
        "getclosurevars", "currentframe", "stack", "trace", "getouterframes", "getinnerframes",
        "getframeinfo", "getargvalues",
    )
} | {
    ("sys", "_getframe"), ("sys", "settrace"), ("sys", "setprofile"), ("sys", "meta_path"),
    ("sys", "path_hooks"), ("threading", "settrace"), ("threading", "setprofile"),
    ("gc", "get_referrers"), ("gc", "get_referents"), ("gc", "get_objects"),
    ("linecache", "getline"), ("linecache", "getlines"),
}  # fmt: skip
FORBIDDEN_MODULES = {"dis", "ast", "tokenize", "linecache", "ctypes", "bytecode", "codeop"}
FORBIDDEN_DUNDERS = {
    "__code__", "co_code", "co_consts", "co_names", "co_varnames", "co_filename", "co_lines",
    "__globals__", "__closure__", "f_locals", "f_globals", "f_back", "f_code", "gi_frame",
    "cr_frame", "__builtins__",
}  # fmt: skip
FORBIDDEN_BUILTINS = {"exec", "eval", "compile", "__import__"}
PATCHED_NAMES = {"torch", "nn", "builtins", "functional", "F", "optim", "inspect"}


def package_trees():
    for path in sorted(paths.PACKAGE_DIR.rglob("*.py")):
        yield path.relative_to(paths.PROJECT_ROOT), ast.parse(path.read_text(encoding="utf-8"))


def findings_in(tree: ast.AST) -> list[str]:
    found = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found += [
                f"import {a.name}" for a in node.names if a.name.split(".")[0] in FORBIDDEN_MODULES
            ]
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            root = node.module.split(".")[0]
            if root in FORBIDDEN_MODULES:
                found.append(f"from {node.module} import ...")
            for alias in node.names:
                if (node.module, alias.name) in FORBIDDEN_ATTRIBUTES:
                    found.append(f"from {node.module} import {alias.name}")
        elif isinstance(node, ast.Attribute):
            owner = node.value.id if isinstance(node.value, ast.Name) else None
            if (owner, node.attr) in FORBIDDEN_ATTRIBUTES:
                found.append(f"{owner}.{node.attr}")
            if node.attr in FORBIDDEN_DUNDERS:
                found.append(f".{node.attr}")
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id in FORBIDDEN_BUILTINS:
                found.append(f"{node.func.id}(...)")
            if node.func.id == "setattr" and node.args:
                target = node.args[0]
                base = target.value if isinstance(target, ast.Attribute) else target
                if isinstance(base, ast.Name) and base.id in PATCHED_NAMES:
                    found.append(f"setattr({ast.unparse(target)}, ...)")
        elif isinstance(node, (ast.Assign, ast.AugAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                base = target
                while isinstance(base, ast.Attribute):
                    base = base.value
                attribute_of_name = isinstance(target, ast.Attribute) and isinstance(base, ast.Name)
                if attribute_of_name and base.id in PATCHED_NAMES:
                    found.append(f"assignment to {ast.unparse(target)}")
    return found


@pytest.mark.parametrize(
    "module", [path for path, _tree in package_trees()], ids=lambda p: str(p.as_posix())
)
def test_the_module_uses_no_dark_magic(module):
    tree = ast.parse((paths.PROJECT_ROOT / module).read_text(encoding="utf-8"))
    assert findings_in(tree) == []


def test_the_scan_would_notice_dark_magic():
    """The scanner itself finds what it is looking for (it is not vacuous)."""
    sample = ast.parse(
        "import inspect, re\n"
        "def guess(step):\n"
        "    source = inspect.getsource(step)\n"
        "    frame = sys._getframe(1)\n"
        "    code = step.__code__.co_names\n"
        "    eval('1')\n"
        "    torch.nn.Module.__call__ = None\n"
        "    return re.findall('optimizer', source)\n"
    )
    found = findings_in(sample)
    for expected in ("inspect.getsource", "sys._getframe", ".__code__", ".co_names", "eval(...)",
                     "assignment to torch.nn.Module.__call__"):  # fmt: skip
        assert expected in found, expected


# ------------------------------------------------------------------- behaviour
class TrainOnce:
    """A callable object as the training step (no function source to read)."""

    def __init__(self, model, optimizer, loss_fn) -> None:
        self.model, self.optimizer, self.loss_fn = model, optimizer, loss_fn

    def __call__(self, batch):
        return _native_step(self.model, self.optimizer, self.loss_fn, batch)

    def method(self, batch):
        return _native_step(self.model, self.optimizer, self.loss_fn, batch)


def _native_step(model, optimizer, loss_fn, batch):
    optimizer.zero_grad()
    loss = loss_fn(model(batch[0]), batch[1])
    loss.backward()
    optimizer.step()
    return loss


GENERATED = """
def generated_step(batch):
    optimizer.zero_grad()
    loss = loss_fn(model(batch[0]), batch[1])
    loss.backward()
    optimizer.step()
    return loss
"""


def make_step(kind: str, model, optimizer, loss_fn):
    if kind == "generated":
        namespace = {"model": model, "optimizer": optimizer, "loss_fn": loss_fn}
        exec(compile(GENERATED, "<generated in memory>", "exec"), namespace)  # test side only
        return namespace["generated_step"]
    if kind == "partial":
        return functools.partial(_native_step, model, optimizer, loss_fn)
    if kind == "callable object":
        return TrainOnce(model, optimizer, loss_fn)
    if kind == "bound method":
        return TrainOnce(model, optimizer, loss_fn).method
    return lambda batch: _native_step(model, optimizer, loss_fn, batch)  # "lambda"


KINDS = ["generated", "partial", "callable object", "bound method", "lambda"]


@pytest.mark.parametrize("kind", KINDS)
def test_steps_without_retrievable_source_train_normally(kind):
    torch.manual_seed(0)
    model = TinyNet()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    loss_fn = nn.CrossEntropyLoss()
    step = make_step(kind, model, optimizer, loss_fn)
    if kind == "generated":
        with pytest.raises(OSError):
            inspect.getsource(step)  # there is no source anywhere
    before = model.fc1.weight.detach().clone()
    process = ht.TrainingProcess(model, optimizer, loader(), epochs=2, train_step=step)
    result = ht.run(process, gui=False)
    assert result.epochs_completed == 2
    assert not torch.equal(model.fc1.weight, before)


def test_a_run_never_reads_source_code_or_installs_tracing(monkeypatch):
    """Every source-reading and tracing function is wrapped: a call *from the package's own
    code* is recorded (PyTorch itself may legitimately use some of them internally, e.g.
    when it imports ``torch._dynamo`` lazily - those calls go through unchanged)."""
    violations: list[str] = []

    def trapped(owner, name):
        original = getattr(owner, name)

        def wrapper(*args, **kwargs):
            caller = sys._getframe(1).f_globals.get("__name__", "")
            if caller.startswith("hollywood_training"):
                violations.append(f"{caller} called {owner.__name__}.{name}")
            return original(*args, **kwargs)

        monkeypatch.setattr(owner, name, wrapper)

    for name in ("getsource", "getsourcelines", "getsourcefile", "findsource", "getfile",
                 "getclosurevars", "currentframe", "stack"):  # fmt: skip
        trapped(inspect, name)
    for owner, name in ((linecache, "getlines"), (linecache, "getline"), (sys, "settrace"),
                        (sys, "setprofile"), (threading, "settrace")):  # fmt: skip
        trapped(owner, name)
    torch.manual_seed(0)
    model = TinyNet()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    loss_fn = nn.CrossEntropyLoss()
    process = ht.TrainingProcess(model, optimizer, loader(), loss_fn=loss_fn, val_data=loader(),
                                 epochs=2)  # fmt: skip
    process.train_step(lambda batch: _native_step(model, optimizer, loss_fn, batch))
    process.on_epoch_end(lambda record: None)
    result = ht.run(process, gui=False)
    assert result.epochs_completed == 2
    assert violations == []


def test_the_users_functions_and_classes_are_left_untouched():
    torch.manual_seed(0)
    model = TinyNet()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    loss_fn = nn.CrossEntropyLoss()

    def train_step(batch):
        return _native_step(model, optimizer, loss_fn, batch)

    def validation_step(batch):
        return loss_fn(model(batch[0]), batch[1])

    def hook(record):
        pass

    functions = (train_step, validation_step, hook)
    before = [(f.__code__, dict(vars(f)), f.__defaults__, f.__name__) for f in functions]
    forward, module_call = TinyNet.forward, nn.Module.__call__
    process = ht.TrainingProcess(model, optimizer, loader(), val_data=loader(), epochs=2,
                                 train_step=train_step, validation_step=validation_step,
                                 on_epoch_end=hook)  # fmt: skip
    ht.run(process, gui=False)
    after = [(f.__code__, dict(vars(f)), f.__defaults__, f.__name__) for f in functions]
    assert after == before
    assert TinyNet.forward is forward
    assert nn.Module.__call__ is module_call  # PyTorch itself was not patched either
    assert type(model).__call__ is module_call

"""Shared building blocks for the SPEC harness: user-style networks and processes,
frontend stand-ins, a scripted GUI user, paint recorders, image analysis and the
window-process probe."""

from __future__ import annotations

import ast
import contextlib
import json
import os
import subprocess
import sys
import threading
import time
import warnings
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

import hollywood_training as ht
from hollywood_training.commands import CommandContext, CommandResult, create_registry
from hollywood_training.sync import (
    Envelope,
    EpochCompleted,
    EpochRecord,
    SessionFinished,
    SessionStarted,
    SessionState,
    SyncChannel,
    TrainingControl,
    TrainingPaused,
    TrainingState,
    TrainingTarget,
)
from hollywood_training.tracking import NetworkTracker
from hollywood_training.training import TrainingManager
from hollywood_training.training.describe import build_session_info

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_DIR = PROJECT_ROOT / "hollywood_training"
DEMOS_DIR = PROJECT_ROOT / "demos"
WIKI_DIR = PROJECT_ROOT / "wiki"
README = PROJECT_ROOT / "README.md"
CHILD_HOOKS_DIR = Path(__file__).resolve().parent / "child_hooks"

G2_LABELS = ("BATCH SIZE", "LEARNING RATE", "LOSS FUNCTION", "OPTIMIZER", "NORMALIZATION")


# --------------------------------------------------------------------- networks
class TinyNet(nn.Module):
    """2 inputs -> 6 hidden (Tanh) -> 2 outputs: small enough to be drawn in full."""

    def __init__(self, hidden: int = 6) -> None:
        super().__init__()
        self.fc1 = nn.Linear(2, hidden)
        self.act = nn.Tanh()
        self.fc2 = nn.Linear(hidden, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


class RichNet(nn.Module):
    """BatchNorm, Dropout and two activations: exercises train/eval modes and the RNG."""

    def __init__(self, hidden: int = 8) -> None:
        super().__init__()
        self.fc1 = nn.Linear(2, hidden)
        self.norm = nn.BatchNorm1d(hidden)
        self.act1 = nn.ReLU()
        self.drop = nn.Dropout(0.2)
        self.fc2 = nn.Linear(hidden, hidden)
        self.act2 = nn.Tanh()
        self.out = nn.Linear(hidden, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.drop(self.act1(self.norm(self.fc1(x))))
        return self.out(self.act2(self.fc2(x)))


def xor_data(samples: int = 96, seed: int = 0) -> TensorDataset:
    """2-D points labelled by the sign of x*y (a small classification task)."""
    generator = torch.Generator().manual_seed(seed)
    inputs = torch.randn(samples, 2, generator=generator)
    targets = (inputs[:, 0] * inputs[:, 1] > 0).long()
    return TensorDataset(inputs, targets)


# --------------------------------------------------------------------- processes
@dataclass
class Bundle:
    """A user's training script, as objects: what was built and what happened."""

    process: ht.TrainingProcess
    model: nn.Module
    optimizer: Any
    loss_fn: Any
    train_loader: Any
    val_loader: Any
    calls: list[tuple[str, float, int]] = field(default_factory=list)  # phase, time, done
    step_lrs: list[tuple[float, float]] = field(default_factory=list)  # lr before/after step
    completed: list[int] = field(default_factory=lambda: [0])  # epochs completed (hook)
    weights: dict[int, dict[str, torch.Tensor]] = field(default_factory=dict)  # per epoch
    initial: dict[str, torch.Tensor] = field(default_factory=dict)

    @property
    def batches_per_epoch(self) -> int:
        return len(self.train_loader)

    def train_calls(self) -> list[tuple[str, float, int]]:
        return [call for call in self.calls if call[0] == "train"]

    def first_batch_time(self, epoch: int) -> float | None:
        """When the first training batch of ``epoch`` (1-based) started."""
        times = [t for phase, t, done in self.calls if phase == "train" and done == epoch - 1]
        return min(times) if times else None


def snapshot_parameters(model: nn.Module) -> dict[str, torch.Tensor]:
    return {name: parameter.detach().clone() for name, parameter in model.named_parameters()}


def build_process(
    epochs: int = 3,
    *,
    model: nn.Module | None = None,
    validation: bool = True,
    lr: float = 0.1,
    batch: int = 16,
    samples: int = 96,
    seed: int = 0,
    step_delay: float = 0.0,
    native_steps: bool = True,
    optimizer: Any = None,
    **options: Any,
) -> Bundle:
    """A small process written the way a user writes one: native steps on decorators."""
    torch.manual_seed(seed)
    model = model if model is not None else TinyNet()
    optimizer = optimizer if optimizer is not None else torch.optim.SGD(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    train_loader = DataLoader(xor_data(samples, seed), batch_size=batch)
    val_loader = (
        DataLoader(xor_data(max(2, samples // 2), seed + 1), batch_size=batch * 2)
        if validation
        else None
    )
    process = ht.TrainingProcess(
        model,
        optimizer,
        train_loader,
        epochs=epochs,
        loss_fn=loss_fn,
        val_data=val_loader,
        **options,
    )
    bundle = Bundle(process, model, optimizer, loss_fn, train_loader, val_loader)
    bundle.initial = snapshot_parameters(model)

    if native_steps:

        @process.train_step
        def train_step(batch_data):
            lr_before = optimizer.param_groups[0]["lr"]
            bundle.calls.append(("train", time.monotonic(), bundle.completed[0]))
            if step_delay:
                time.sleep(step_delay)
            inputs, targets = batch_data
            optimizer.zero_grad()
            loss = loss_fn(model(inputs), targets)
            loss.backward()
            optimizer.step()
            bundle.step_lrs.append((lr_before, optimizer.param_groups[0]["lr"]))
            return loss

        if validation:

            @process.validation_step
            def validation_step(batch_data):
                bundle.calls.append(("validation", time.monotonic(), bundle.completed[0]))
                inputs, targets = batch_data
                return loss_fn(model(inputs), targets)

    @process.on_epoch_end
    def remember(record: EpochRecord) -> None:
        bundle.completed[0] = record.epoch
        bundle.weights[record.epoch] = snapshot_parameters(model)

    return bundle


# ------------------------------------------------------------- frontend stand-ins
class RecordingFrontend:
    """Frontend stand-in on a :class:`SyncChannel`: records every envelope, keeps the
    frontend-side :class:`SessionState` and acknowledges synchronous envelopes (after
    ``ack_delay`` seconds, to imitate drawing time)."""

    def __init__(
        self,
        channel: SyncChannel,
        *,
        on_event: Callable[[Any], None] | None = None,
        ack_delay: float = 0.0,
    ) -> None:
        self.channel = channel
        self.session = SessionState()
        self.envelopes: list[Envelope] = []
        self.acks: list[tuple[int, float]] = []
        self._on_event = on_event
        self._ack_delay = ack_delay
        channel.connect(self)

    def __call__(self, envelope: Envelope) -> None:
        try:
            self.envelopes.append(envelope)
            self.session.apply(envelope.event)
            if self._on_event is not None:
                self._on_event(envelope.event)
            if envelope.synchronous and self._ack_delay:
                time.sleep(self._ack_delay)
        finally:
            if envelope.synchronous:
                self.acks.append((envelope.sequence, time.monotonic()))
                self.channel.acknowledge(envelope.sequence)

    @property
    def events(self) -> list[Any]:
        return [envelope.event for envelope in list(self.envelopes)]

    def of_type(self, kind: type) -> list[Any]:
        return [event for event in self.events if isinstance(event, kind)]

    def names(self) -> list[str]:
        return [type(event).__name__ for event in self.events]


class ManagedRun:
    """A TrainingManager on a worker thread, with the test thread acting as the command
    thread of a frontend (exactly how the default window process serves commands)."""

    def __init__(
        self,
        process: ht.TrainingProcess,
        *,
        linger: bool = False,
        start_paused: bool = False,
        ack_delay: float = 0.0,
        on_event: Callable[[Any], None] | None = None,
    ) -> None:
        self.control = TrainingControl(process.epochs, start_paused=start_paused)
        self.channel = SyncChannel()
        self.frontend = RecordingFrontend(self.channel, ack_delay=ack_delay, on_event=on_event)
        self.manager = TrainingManager(
            process, channel=self.channel, control=self.control, linger=linger
        )
        self.registry = create_registry(process.commands)
        self.context = CommandContext(
            control=self.control,
            session=self.frontend.session,
            registry=self.registry,
            target=TrainingTarget(process.model, process.optimizer),
        )
        self.result: Any = None
        self.error: BaseException | None = None
        self.thread = threading.Thread(target=self._run, name="test-training-thread", daemon=True)

    def _run(self) -> None:
        try:
            self.result = self.manager.run()
        except BaseException as exc:  # reported by join()
            self.error = exc

    def start(self) -> ManagedRun:
        self.thread.start()
        return self

    def command(self, line: str) -> CommandResult:
        return self.registry.execute(line, self.context)

    @property
    def session(self) -> SessionState:
        return self.frontend.session

    def wait_for(self, predicate: Callable[[], bool], what: str, timeout: float = 15.0) -> None:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if predicate():
                return
            if not self.thread.is_alive() and not predicate():
                raise AssertionError(f"training ended before: {what} (error: {self.error!r})")
            time.sleep(0.005)
        raise AssertionError(f"timed out waiting for: {what}")

    def join(self, timeout: float = 30.0) -> Any:
        self.thread.join(timeout)
        assert not self.thread.is_alive(), "the training thread did not finish"
        if self.error is not None:
            raise self.error
        return self.result

    def __enter__(self) -> ManagedRun:
        return self.start()

    def __exit__(self, exc_type: object, *exc_info: object) -> None:
        self.control.stop()
        self.thread.join(15.0)
        if self.thread.is_alive():  # never leave a training thread running into later tests
            message = "the training thread did not stop within 15 s"
            if exc_type is None:
                raise AssertionError(message)
            warnings.warn(message, RuntimeWarning, stacklevel=2)


@dataclass(frozen=True)
class Delivery:
    """One blocking hand-off seen on the training side."""

    event: str
    epoch: int | None
    start: float
    end: float
    acknowledged: bool


def event_epoch(event: Any) -> int | None:
    if isinstance(event, EpochCompleted):
        return event.record.epoch
    if isinstance(event, TrainingPaused):
        return event.epoch
    if isinstance(event, SessionFinished):
        return event.epochs_completed
    if isinstance(event, SessionStarted):
        return 0
    return None


def record_deliveries(monkeypatch: Any) -> list[Delivery]:
    """Record every ``SyncChannel.deliver`` call (start/end time, outcome) in this process."""
    deliveries: list[Delivery] = []
    original = SyncChannel.deliver

    def deliver(self: SyncChannel, event: Any) -> bool:
        start = time.monotonic()
        acknowledged = original(self, event)
        deliveries.append(
            Delivery(
                type(event).__name__, event_epoch(event), start, time.monotonic(), acknowledged
            )
        )
        return acknowledged

    monkeypatch.setattr(SyncChannel, "deliver", deliver)
    return deliveries


def wait_until(predicate: Callable[[], bool], what: str, timeout: float = 10.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.005)
    raise AssertionError(f"timed out waiting for: {what}")


# ------------------------------------------------------------------ session data
def session_started(
    process: ht.TrainingProcess,
    *,
    epochs: int | None = None,
    max_neurons: int = 20,
    state: TrainingState = TrainingState.RUNNING,
) -> tuple[NetworkTracker, SessionStarted]:
    """The ``SessionStarted`` event a manager would send for ``process``."""
    tracker = NetworkTracker(process.model, max_neurons=max_neurons)
    info = build_session_info(
        process,
        tracker.architecture.describe(),
        total_epochs=epochs or process.epochs,
        neuron_limit=max_neurons,
    )
    return tracker, SessionStarted(info, tracker.snapshot(0), state)


def epoch_completed(
    tracker: NetworkTracker,
    epoch: int,
    *,
    total: int,
    train_loss: float,
    val_loss: float | None = None,
    lr: float = 0.1,
    batch_size: int | None = 16,
    train_metrics: dict[str, float] | None = None,
    val_metrics: dict[str, float] | None = None,
) -> EpochCompleted:
    record = EpochRecord(
        epoch,
        train_loss,
        val_loss,
        train_metrics or {},
        val_metrics or {},
        lr,
        0.01,
        6,
        2 if val_loss is not None else 0,
    )
    return EpochCompleted(record, tracker.snapshot(epoch), total, lr, batch_size, epoch * 0.05)


# ---------------------------------------------------------------------- GUI user
def active_window() -> Any:
    """The visible :class:`MainWindow` of this process, if any."""
    from PySide6.QtWidgets import QApplication

    from hollywood_training.gui import MainWindow

    app = QApplication.instance()
    if app is None:
        return None
    windows = [w for w in app.topLevelWidgets() if isinstance(w, MainWindow) and w.isVisible()]
    return windows[-1] if windows else None


def type_command(window: Any, line: str) -> None:
    """Type ``line`` into the window's terminal and press Enter, like a user."""
    from PySide6.QtCore import Qt
    from PySide6.QtTest import QTest

    field_ = window.terminal.input
    field_.clear()
    QTest.keyClicks(field_, line)
    QTest.keyClick(field_, Qt.Key.Key_Return)


Step = tuple[str, Callable[[Any], bool], Callable[[Any], None]]


class GuiDriver:
    """A scripted user of the in-process window.

    Polls on the GUI thread with a ``QTimer`` (which fires while the training loop pumps
    Qt events) and performs the steps in order: ``(name, ready(window), act(window))``.
    On a timeout or an error it closes the window, which stops the run, so a broken
    expectation fails the test instead of hanging it.
    """

    def __init__(self, steps: Iterable[Step], *, timeout: float = 60.0, interval_ms: int = 20):
        from PySide6.QtCore import QTimer

        self._steps = list(steps)
        self.completed: list[str] = []
        self.errors: list[str] = []
        self._deadline = time.monotonic() + timeout
        self._timer = QTimer()
        self._timer.setInterval(interval_ms)
        self._timer.timeout.connect(self._tick)
        self._timer.start()

    def _tick(self) -> None:
        if not self._steps:
            self._timer.stop()
            return
        window = active_window()
        name, ready, act = self._steps[0]
        if time.monotonic() > self._deadline:
            self.errors.append(f"timed out waiting for step {name!r}")
            self._abort(window)
            return
        if window is None:
            return
        try:
            if not ready(window):
                return
            self._steps.pop(0)
            act(window)
            self.completed.append(name)
        except Exception as exc:
            self.errors.append(f"step {name!r}: {type(exc).__name__}: {exc}")
            self._abort(window)

    def _abort(self, window: Any) -> None:
        self._timer.stop()
        if window is not None:
            window.close()

    def stop(self) -> None:
        self._timer.stop()

    def check(self) -> None:
        self.stop()
        assert not self.errors, self.errors
        assert not self._steps, f"steps never reached: {[step[0] for step in self._steps]}"


# --------------------------------------------------------------- paint recorders
def text_recorder(monkeypatch: Any, *modules: Any) -> list[str]:
    """Replace the ``QPainter`` used by the given widget modules with a subclass that
    records every string passed to ``drawText``. Returns the live list of strings."""
    from PySide6.QtGui import QPainter

    texts: list[str] = []

    class RecordingPainter(QPainter):
        def drawText(self, *args: Any) -> Any:
            texts.extend(arg for arg in args if isinstance(arg, str))
            return super().drawText(*args)

    for module in modules:
        monkeypatch.setattr(module, "QPainter", RecordingPainter)
    return texts


def call_recorder(monkeypatch: Any, module: Any, name: str) -> list[tuple[tuple, dict]]:
    """Wrap the module-level function ``module.name``; returns the list of its calls."""
    calls: list[tuple[tuple, dict]] = []
    original = getattr(module, name)

    def wrapper(*args: Any, **kwargs: Any) -> Any:
        calls.append((args, kwargs))
        return original(*args, **kwargs)

    monkeypatch.setattr(module, name, wrapper)
    return calls


# ---------------------------------------------------------------- image analysis
def image_rgb(image: Any) -> np.ndarray:
    """``(height, width, 3)`` uint8 array of a ``QImage`` or ``QPixmap``."""
    from PySide6.QtGui import QImage, QPixmap

    if isinstance(image, QPixmap):
        image = image.toImage()
    image = image.convertToFormat(QImage.Format.Format_RGBA8888)
    width, height = image.width(), image.height()
    try:
        raw = np.frombuffer(image.constBits(), dtype=np.uint8, count=image.sizeInBytes())
        rows = raw.reshape(height, image.bytesPerLine())[:, : width * 4]
        return rows.reshape(height, width, 4)[:, :, :3].copy()
    except Exception:  # a binding without buffer access: sample pixels instead
        step = max(1, max(width, height) // 400)
        out = np.zeros((len(range(0, height, step)), len(range(0, width, step)), 3), np.uint8)
        for row, y in enumerate(range(0, height, step)):
            for column, x in enumerate(range(0, width, step)):
                pixel = image.pixelColor(x, y)
                out[row, column] = (pixel.red(), pixel.green(), pixel.blue())
        return out


def hue_chroma(rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per pixel: hue in degrees (NaN for greys), chroma (max-min) and value (max)."""
    values = rgb.reshape(-1, 3).astype(float)
    r, g, b = values[:, 0], values[:, 1], values[:, 2]
    maximum = values.max(axis=1)
    minimum = values.min(axis=1)
    chroma = maximum - minimum
    safe = np.where(chroma == 0, 1.0, chroma)
    hue_r = (60.0 * (g - b) / safe) % 360.0
    hue_g = 60.0 * (b - r) / safe + 120.0
    hue_b = 60.0 * (r - g) / safe + 240.0
    hue = np.where(maximum == r, hue_r, np.where(maximum == g, hue_g, hue_b))
    hue = np.where(chroma == 0, np.nan, hue)
    return hue, chroma, maximum


def blue_share(rgb: np.ndarray, *, low: float = 170.0, high: float = 270.0) -> float:
    """Share of clearly coloured pixels whose hue lies in the blue band ``[low, high]``."""
    hue, chroma, value = hue_chroma(rgb)
    coloured = (chroma >= 40) & (value >= 50)
    if not coloured.any():
        return 0.0
    in_band = (hue >= low) & (hue <= high)
    return float((in_band & coloured).sum() / coloured.sum())


# ----------------------------------------------------------- window-process probe
@dataclass
class ChildProbe:
    """Reads what ``child_hooks/ht_child_probe.py`` logged inside the window process."""

    log: Path

    def records(self, kind: str | None = None) -> list[dict[str, Any]]:
        if not self.log.exists():
            return []
        rows = [
            json.loads(line)
            for line in self.log.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        return [row for row in rows if kind is None or row.get("kind") == kind]

    def applied(self, event: str) -> list[dict[str, Any]]:
        return [row for row in self.records("applied") if row.get("event") == event]

    def terminal_text(self) -> str:
        return "\n".join(str(row.get("text", "")) for row in self.records("terminal"))

    def assert_installed(self) -> None:
        errors = self.records("error")
        assert self.records("installed"), (
            f"the window-process probe did not start (log {self.log}); errors: {errors}"
        )


def install_child_probe(
    monkeypatch: Any, tmp_path: Path, rules: Iterable[dict[str, Any]] = (), deadline: float = 90
) -> ChildProbe:
    """Instrument the next GUI window process started by ``ht.run`` (via PYTHONPATH).

    ``rules`` are ``{"on": trigger, "do": [action, ...]}`` dicts executed inside the
    window, e.g. ``{"on": {"event": "EpochCompleted", "epoch": 3}, "do": [{"close": True}]}``;
    see ``child_hooks/ht_child_probe.py`` for the full list. ``deadline`` closes the
    window after that many seconds, so a missed trigger cannot hang a test.
    """
    log = tmp_path / "window-process.jsonl"
    pythonpath = os.pathsep.join(filter(None, [str(CHILD_HOOKS_DIR), os.environ.get("PYTHONPATH")]))
    monkeypatch.setenv("PYTHONPATH", pythonpath)
    monkeypatch.setenv("HT_TEST_CHILD_LOG", str(log))
    monkeypatch.setenv("HT_TEST_CHILD_SCRIPT", json.dumps(list(rules)))
    monkeypatch.setenv("HT_TEST_CHILD_DEADLINE", str(deadline))
    return ChildProbe(log)


# -------------------------------------------------------------------- processes
def run_python(
    args: list[str],
    *,
    timeout: float = 240,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess:
    """Run this interpreter in a subprocess (headless Qt unless a platform is chosen)."""
    environment = dict(os.environ)
    environment.setdefault("QT_QPA_PLATFORM", "offscreen")
    if env:
        environment.update(env)
    return subprocess.run(
        [sys.executable, *args],
        capture_output=True,
        text=True,
        timeout=timeout,
        cwd=str(cwd) if cwd else None,
        env=environment,
        check=False,
    )


def window_process_children() -> list[Any]:
    """Live child processes of this process that run the GUI window."""
    import psutil

    found = []
    for child in psutil.Process().children(recursive=True):
        try:
            if "hollywood_training.gui.window_process" in " ".join(child.cmdline()):
                found.append(child)
        except (psutil.Error, OSError):
            continue
    return found


# ------------------------------------------------------------------ source code
def package_sources() -> list[Path]:
    return sorted(p for p in PACKAGE_DIR.rglob("*.py") if "__pycache__" not in p.parts)


def module_name(path: Path) -> str:
    parts = list(path.resolve().relative_to(PROJECT_ROOT).with_suffix("").parts)
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


def parse_source(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


# ---------------------------------------------------------- fixtures' payloads
@dataclass
class GuiSession:
    """A shown ``MainWindow`` displaying a session that the test feeds event by event."""

    window: Any
    session: SessionState
    tracker: NetworkTracker
    bundle: Bundle
    started: SessionStarted

    def push(self, event: Any) -> Any:
        """Apply ``event`` the way every frontend does: session state first, then window."""
        self.session.apply(event)
        self.window.apply(event)
        return event

    def complete_epoch(
        self, epoch: int, *, train_loss: float = 0.9, val_loss: float | None = 0.8, **kwargs: Any
    ) -> EpochCompleted:
        total = kwargs.pop("total", self.session.total_epochs or self.bundle.process.epochs)
        return self.push(
            epoch_completed(
                self.tracker, epoch, total=total, train_loss=train_loss, val_loss=val_loss, **kwargs
            )
        )


class FakeFrontend:
    """``FrontendActions`` stand-in that records what the commands asked of the window."""

    def __init__(self) -> None:
        self.cleared = 0
        self.closed = 0
        self.screenshots: list[Path] = []

    def clear_output(self) -> None:
        self.cleared += 1

    def close(self) -> None:
        self.closed += 1

    def screenshot(self, path: Path) -> Path:
        self.screenshots.append(Path(path))
        return Path(path)


@dataclass
class CommandEnv:
    """A command context over a session that has completed one epoch (of five)."""

    context: CommandContext
    control: TrainingControl
    session: SessionState
    bundle: Bundle
    frontend: FakeFrontend | None
    tracker: NetworkTracker

    def run(self, line: str) -> CommandResult:
        return self.context.registry.execute(line, self.context)

    def apply_queued_actions(self) -> list[str | None]:
        """Do what the training thread does at its next safe point."""
        target = TrainingTarget(self.bundle.model, self.bundle.optimizer)
        return [action(target) for action, _description in self.control.take_actions()]


def make_command_env(*, frontend: bool = True) -> CommandEnv:
    bundle = build_process(epochs=5)
    control = TrainingControl(5)
    tracker, started = session_started(bundle.process)
    session = SessionState()
    session.apply(started)
    session.apply(
        epoch_completed(
            tracker, 1, total=5, train_loss=0.9, val_loss=0.8, train_metrics={"accuracy": 0.5}
        )
    )
    control.epoch_completed(1)
    fake = FakeFrontend() if frontend else None
    context = CommandContext(
        control=control,
        session=session,
        registry=create_registry(bundle.process.commands),
        target=TrainingTarget(bundle.model, bundle.optimizer),
        frontend=fake,
    )
    return CommandEnv(context, control, session, bundle, fake, tracker)


class WindowProcessHarness:
    """The GUI side of the default window process (``WindowProcess``), run inside the test
    process over a ``multiprocessing.Pipe``: the test plays the training process.

    ``sent`` records every message the window sends together with what the window had
    painted at that moment: ``(message, network_painted_epoch, chart_painted_epoch)``.
    """

    def __init__(self, qtbot: Any) -> None:
        from multiprocessing import Pipe

        from hollywood_training.gui.link import Link
        from hollywood_training.gui.window_process import WindowProcess

        self.qtbot = qtbot
        self.connection, window_end = Pipe()
        link = Link(window_end)
        self.sent: list[tuple[tuple, Any, Any]] = []
        original_send = link.send

        def send(*message: Any) -> bool:
            window = self.process.window if hasattr(self, "process") else None
            network = window.network.last_painted_epoch if window is not None else None
            chart = window.chart.last_painted_epoch if window is not None else None
            self.sent.append((message, network, chart))
            return original_send(*message)

        link.send = send  # instance attribute: what WindowProcess calls
        self.process = WindowProcess(link)
        self.window = self.process.window
        qtbot.addWidget(self.window)

    def start(self) -> None:
        self.process.start()

    def send(self, *message: Any) -> None:
        """Send a message as the training process would."""
        self.connection.send(message)

    def receive(self, timeout: float = 5.0) -> tuple:
        """Next message from the window (Qt events are processed while waiting)."""
        self.qtbot.waitUntil(lambda: self.connection.poll(), timeout=int(timeout * 1000))
        return self.connection.recv()

    def wait_for_text(self, text: str, timeout: float = 5.0) -> None:
        self.qtbot.waitUntil(
            lambda: text in self.window.terminal.text(), timeout=int(timeout * 1000)
        )

    def shutdown(self) -> None:
        """End the link like a finished training process; the window notices."""
        self.connection.close()
        with contextlib.suppress(Exception):
            self.wait_for_text("disconnected")

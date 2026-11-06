"""Python subprocesses with the harness environment (``htspec.env.subprocess_env``)."""

from __future__ import annotations

import contextlib
import json
import subprocess
import sys
import threading
import time
import weakref
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from . import paths
from .env import subprocess_env

_SPAWNED: weakref.WeakSet[Spawned] = weakref.WeakSet()  # see terminate_leftovers


def terminate_leftovers() -> int:
    """Terminate every spawned process that is still running; how many were. The safety
    net in ``tests/conftest.py`` calls this at the end of every module."""
    leftover = [spawned for spawned in list(_SPAWNED) if spawned.alive()]
    for spawned in leftover:
        spawned.terminate()
    return len(leftover)


def run_python(
    args: Sequence[str],
    *,
    env: dict[str, str] | None = None,
    timeout: float = 180.0,
    cwd: Path = paths.PROJECT_ROOT,
    display: str | None = None,
) -> subprocess.CompletedProcess:
    """Run ``python <args>`` to completion; stdout / stderr captured as text."""
    return subprocess.run(
        [sys.executable, *args],
        cwd=cwd,
        env=subprocess_env(env, display=display),
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )


class Spawned:
    """A running ``python`` subprocess whose stdout is collected line by line.

    Lines that are JSON objects are also parsed (the scenario scripts report that way).
    """

    def __init__(self, popen: subprocess.Popen) -> None:
        self.popen = popen
        self.lines: list[str] = []
        self.messages: list[dict[str, Any]] = []
        self.stderr_lines: list[str] = []
        self._condition = threading.Condition()
        self._reader = threading.Thread(target=self._read, daemon=True)
        self._reader.start()
        self._err_reader = threading.Thread(target=self._read_err, daemon=True)
        self._err_reader.start()

    def _read(self) -> None:
        assert self.popen.stdout is not None
        for line in self.popen.stdout:
            line = line.rstrip("\n")
            with self._condition:
                self.lines.append(line)
                if line.startswith("{"):
                    with contextlib.suppress(json.JSONDecodeError):
                        self.messages.append(json.loads(line))
                self._condition.notify_all()

    def _read_err(self) -> None:
        assert self.popen.stderr is not None
        for line in self.popen.stderr:
            with self._condition:
                self.stderr_lines.append(line.rstrip("\n"))

    @property
    def pid(self) -> int:
        return self.popen.pid

    def alive(self) -> bool:
        return self.popen.poll() is None

    def wait_message(
        self, predicate: Callable[[dict[str, Any]], bool], timeout: float = 30.0
    ) -> dict[str, Any]:
        deadline = time.monotonic() + timeout
        with self._condition:
            while True:
                for message in self.messages:
                    if predicate(message):
                        return message
                remaining = deadline - time.monotonic()
                ended = self.popen.poll() is not None and not self._reader.is_alive()
                if remaining <= 0 or ended:
                    raise AssertionError(
                        f"no matching message within {timeout} s (exit code "
                        f"{self.popen.poll()}); stdout tail: {self.lines[-15:]}; stderr tail: "
                        f"{self.stderr_lines[-25:]}"
                    )
                self._condition.wait(min(0.1, remaining))

    def of_kind(self, event: str) -> list[dict[str, Any]]:
        with self._condition:
            return [m for m in self.messages if m.get("event") == event]

    def wait(self, timeout: float = 30.0) -> int:
        code = self.popen.wait(timeout)
        self._reader.join(5)
        self._err_reader.join(5)
        return code

    def terminate(self, timeout: float = 10.0) -> None:
        if self.popen.poll() is None:
            self.popen.terminate()
            try:
                self.popen.wait(timeout)
            except subprocess.TimeoutExpired:
                self.popen.kill()
                self.popen.wait(5)

    def stderr_text(self) -> str:
        with self._condition:
            return "\n".join(self.stderr_lines)


def spawn_python(
    args: Sequence[str],
    *,
    env: dict[str, str] | None = None,
    cwd: Path = paths.PROJECT_ROOT,
    display: str | None = None,
) -> Spawned:
    popen = subprocess.Popen(
        [sys.executable, *args],
        cwd=cwd,
        env=subprocess_env(env, display=display),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        stdin=subprocess.DEVNULL,
        text=True,
        bufsize=1,
    )
    spawned = Spawned(popen)
    _SPAWNED.add(spawned)
    return spawned

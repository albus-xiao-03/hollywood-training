"""Test-side access to the window-process probe (``htspec/probe``).

:func:`probe_env` returns the environment variables that switch the probe on for the
window process started by ``ht.run``; :class:`ProbeLog` reads what it recorded.

An isolated run that lingers (the default ``close_on_finish=False``) must be ended by the
test, e.g. with the autopilot step ``LINGER_CLOSE``.
"""

from __future__ import annotations

import contextlib
import json
import os
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from . import paths

# Autopilot step: close the window once a lingering run reports that it has finished.
LINGER_CLOSE = {"on": "state", "value": "finished", "do": "close", "delay_ms": 300}


def probe_env(
    log_path: Path,
    *,
    script: dict | None = None,
    weights: bool = False,
    interval_ms: int = 100,
    base: dict[str, str] | None = None,
) -> dict[str, str]:
    """Variables that enable the probe in the window process (inherited through ``ht.run``)."""
    source = os.environ if base is None else base
    pythonpath = [str(paths.PROBE_DIR), str(paths.HARNESS_DIR), str(paths.PROJECT_ROOT)]
    if source.get("PYTHONPATH"):
        pythonpath += [p for p in source["PYTHONPATH"].split(os.pathsep) if p not in pythonpath]
    env = {
        "PYTHONPATH": os.pathsep.join(pythonpath),
        "HT_SPEC_PROBE_LOG": str(log_path),
        "HT_SPEC_PROBE_INTERVAL": str(interval_ms),
        "HT_SPEC_PROBE_WEIGHTS": "1" if weights else "0",
    }
    if script is not None:
        env["HT_SPEC_PROBE_SCRIPT"] = json.dumps(script)
    return env


class ProbeLog:
    """The JSON-lines log written by the probe."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self._records: list[dict[str, Any]] = []
        self._offset = 0  # bytes of the file already parsed (complete lines only)

    def _read_new(self) -> None:
        if not self.path.exists():
            return
        with open(self.path, "rb") as stream:
            stream.seek(self._offset)
            chunk = stream.read()
        end = chunk.rfind(b"\n")
        if end < 0:
            return  # no complete new line yet
        self._offset += end + 1
        for line in chunk[: end + 1].decode("utf-8").splitlines():
            with contextlib.suppress(json.JSONDecodeError):
                self._records.append(json.loads(line))

    def records(self, kind: str | None = None) -> list[dict[str, Any]]:
        """Every record so far (of ``kind``, if given)."""
        self._read_new()
        return [r for r in self._records if kind is None or r.get("kind") == kind]

    def states(self) -> list[dict[str, Any]]:
        return self.records("state")

    def latest(self) -> dict[str, Any] | None:
        states = self.states()
        return states[-1] if states else None

    def renders(self) -> list[dict[str, Any]]:
        return self.records("render")

    def sends(self, msg: str | None = None) -> list[dict[str, Any]]:
        return [r for r in self.records("send") if msg is None or r.get("msg") == msg]

    def events(self, type_name: str | None = None) -> list[dict[str, Any]]:
        return [r for r in self.records("event") if type_name is None or r["type"] == type_name]

    def views(self) -> list[dict[str, Any]]:
        return self.records("view")

    def errors(self) -> list[dict[str, Any]]:
        return self.records("probe-error")

    def wait(
        self,
        predicate: Callable[[dict[str, Any]], bool],
        *,
        kind: str | None = "state",
        timeout: float = 30.0,
        after: float | None = None,
    ) -> dict[str, Any]:
        """The first record of ``kind`` (any kind for None; newer than ``after``, a monotonic
        time) for which ``predicate`` holds; waits for it up to ``timeout`` seconds."""
        deadline = time.monotonic() + timeout
        while True:
            for record in self.records(kind):
                if after is not None and record.get("t", 0) <= after:
                    continue
                if predicate(record):
                    return record
            if time.monotonic() > deadline:
                latest = self.latest()
                raise AssertionError(
                    f"no {kind!r} record matched within {timeout} s; latest state: "
                    f"{json.dumps(latest, default=str)[:2000] if latest else None}"
                )
            time.sleep(0.05)

    def wait_fresh_state(self, timeout: float = 10.0) -> dict[str, Any]:
        """A state snapshot taken after this call (so it reflects what just happened)."""
        now = time.monotonic()
        return self.wait(lambda r: True, after=now, timeout=timeout)

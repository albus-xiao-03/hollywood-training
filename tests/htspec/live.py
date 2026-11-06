"""``LiveApp``: a real user script running ``ht.run`` on the private X server.

The script is ``htspec/scenarios/train_app.py`` - native PyTorch training handed to
``ht.run(process)`` with the default, isolated window process. The window runs on the
harness's private Xvfb with Qt's real X11 backend (``xcb``), so the tests can use the real
pointer (XTest), read the cursor the X server shows (XFixes) and type on the real keyboard
path. The window process is observed by the read-only probe (``htspec/probe``); the
training process reports on stdout.

There is no window manager: :meth:`LiveApp.focus` gives the window the keyboard focus the
way a window manager does when the user switches to it.
"""

from __future__ import annotations

import contextlib
import itertools
import json
import time
from pathlib import Path
from typing import Any

from . import paths
from .probe_log import ProbeLog, probe_env
from .subproc import spawn_python
from .x11 import XClient, XServer

_counter = itertools.count(1)
PANELS = ("header", "network", "terminal", "output", "input", "status", "chart", "usage")


class LiveApp:
    """One running Hollywood Training session on the private X server."""

    def __init__(
        self,
        xserver: XServer,
        xclient: XClient,
        folder: Path,
        *,
        config: dict[str, Any] | None = None,
        focus: bool = True,
        startup_timeout: float = 60.0,
    ) -> None:
        self.x = xclient
        self.display = xserver.display
        self.log = ProbeLog(Path(folder) / f"live-{next(_counter)}.jsonl")
        self.config = {"epochs": 1_000_000, "batches": 4, "batch_sleep": 0.004, **(config or {})}
        self.win_id: int | None = None  # known once the window has appeared
        self.window_pid: int | None = None
        self._closed = False
        env = probe_env(self.log.path, base={})
        self.process = spawn_python(
            [str(paths.SCENARIO_DIR / "train_app.py"), json.dumps(self.config)],
            env=env,
            display=xserver.display,
        )
        try:
            first = self.log.wait(
                lambda r: bool(r.get("visible") and r.get("win_id")), timeout=startup_timeout
            )
        except AssertionError:
            self.close()
            raise AssertionError(
                f"the window did not appear; stderr: {self.process.stderr_text()[-3000:]}"
            ) from None
        self.win_id = first["win_id"]
        hello = self.log.records("hello")
        self.window_pid = hello[0]["pid"] if hello else None
        if focus:
            try:
                self.focus()
            except BaseException:
                self.close()  # nobody gets this session: it must not stay on the X server
                raise

    # ---------------------------------------------------------------- state
    @staticmethod
    def now() -> float:
        return time.monotonic()

    def state(self) -> dict[str, Any]:
        """A probe snapshot taken after this call."""
        return self.log.wait_fresh_state(timeout=15)

    def wait(self, predicate, *, after: float | None = None, timeout: float = 20.0):
        return self.log.wait(predicate, after=after, timeout=timeout)

    def shown(self) -> bool:
        """Whether a window of this session is viewable on the X server."""
        return any(
            window.window == self.win_id
            or (self.window_pid is not None and window.pid == self.window_pid)
            for window in self.x.viewable_windows()
        )

    def wait_gone(self, timeout: float = 15.0) -> bool:
        """Wait until no window of this session is viewable any more; whether that happened
        within ``timeout`` seconds."""
        deadline = time.monotonic() + timeout
        while self.shown():
            if time.monotonic() >= deadline:
                return False
            time.sleep(0.05)
        return True

    def rect(self, panel: str, state: dict[str, Any] | None = None) -> list[int]:
        return (state or self.state())["panels"][panel]

    def center(self, panel: str, state: dict[str, Any] | None = None) -> tuple[int, int]:
        x, y, w, h = self.rect(panel, state)
        return x + w // 2, y + h // 2

    def handle_center(self, name: str, state: dict[str, Any] | None = None) -> tuple[int, int]:
        x, y, w, h = (state or self.state())["handles"][name]
        return x + w // 2, y + h // 2

    def window_rect(self, state: dict[str, Any] | None = None) -> list[int]:
        """``[x, y, width, height]`` of the window as the X server has it (the truth for
        pointer positions; without a window manager Qt's own idea can differ after a
        window-mode change). Falls back to the probe's geometry."""
        for window in self.x.top_levels():
            if window.window == self.win_id and window.viewable:
                return [window.x, window.y, window.width, window.height]
        return (state or self.state())["geometry"]

    def grid(self, columns: int = 12, rows: int = 8, inset: int = 2) -> list[tuple[int, int]]:
        """Points spread over the whole window, its very edges ``inset`` px inside."""
        x, y, w, h = self.window_rect()
        return [
            (x + inset + round((w - 1 - 2 * inset) * i / (columns - 1)),
             y + inset + round((h - 1 - 2 * inset) * j / (rows - 1)))
            for i in range(columns)
            for j in range(rows)
        ]  # fmt: skip

    # -------------------------------------------------------------- actions
    def focus(self, timeout: float = 15.0) -> dict[str, Any]:
        """Switch to the window (what a window manager does on a click or Alt+Tab)."""
        started = self.now()
        self.x.focus(self.win_id)
        return self.wait(
            lambda r: bool(r["active"] and r["focus"] and r["focus"]["class"] == "CommandLine"),
            after=started,
            timeout=timeout,
        )

    def type(self, text: str) -> None:
        self.x.type_text(text)

    def enter(self, line: str, *, expect: str | None = None, timeout: float = 20.0):
        """Type ``line`` + Return on the real keyboard and wait for the command's answer.

        Waits until the window process has sent this line to the training process and the
        answer has come back (and, with ``expect``, until the answer - or the window-only
        outcome of a view command - contains ``expect``); returns a probe snapshot taken
        after that.
        """
        started = self.now()
        self.type(line + "\n")
        # the probe log's records of a kind ("send", "result"); self.wait() reads "state" only
        sent = self.log.wait(
            lambda r: r.get("msg") == "command" and r.get("value") == line,
            kind="send", after=started, timeout=timeout,
        )  # fmt: skip
        answer = self.log.wait(lambda r: True, kind="result", after=sent["t"], timeout=timeout)
        if expect is not None:

            def says(record) -> bool:
                text = record.get("output") if record["kind"] == "result" else record.get("text")
                return expect in (text or "")

            if not says(answer):
                self.log.wait(says, kind=None, after=sent["t"], timeout=timeout)
        return self.log.wait_fresh_state(timeout=timeout)

    def epochs_reported(self) -> int:
        return len(self.process.of_kind("epoch"))

    def wait_epochs(self, count: int, timeout: float = 30.0) -> None:
        target = self.epochs_reported() + count
        self.process.wait_message(
            lambda m: m.get("event") == "epoch" and m["epoch"] >= target, timeout=timeout
        )

    def quit(self, timeout: float = 30.0) -> dict[str, Any]:
        self.focus()
        self.type("quit\n")
        result = self.process.wait_message(lambda m: m.get("event") == "result", timeout=timeout)
        self.process.wait(timeout)
        return result

    def close(self, timeout: float = 10.0) -> None:
        """End the session: the window process first (killing only the script would orphan
        it on the shared X server), then the script; wait until the window is gone. Closing
        a session again does nothing."""
        import psutil

        if self._closed:
            return
        self._closed = True
        doomed = []
        with contextlib.suppress(psutil.Error):
            doomed = psutil.Process(self.process.pid).children(recursive=True)
        if self.window_pid is not None and all(p.pid != self.window_pid for p in doomed):
            with contextlib.suppress(psutil.Error):
                doomed.append(psutil.Process(self.window_pid))
        for process in doomed:
            with contextlib.suppress(psutil.Error):
                process.terminate()
        _gone, alive = psutil.wait_procs(doomed, timeout=5)
        for process in alive:
            with contextlib.suppress(psutil.Error):
                process.kill()
        self.process.terminate()
        self.wait_gone(timeout)  # this session's window only, also when its pid is unknown

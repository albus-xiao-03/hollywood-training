"""R2 requirement 7 - "**ALL** keyboard input **MUST** be automatically rerouted to internal
terminal inside the `Hollywood Training` window for interaction, **WHEN** users stay on the
`Hollywood Training` window."

"Users stay on the window" = the window is the active one. Then every key reaches the
terminal's command line - automatically (no click first), whichever widget of the window it
was addressed to, whatever the key - and the terminal acts on it. When another window is
active, its keys are its own (the reroute applies only while users stay on the window).

* Offscreen, in-process: the platform input path (native ``QWindow``), keys addressed to
  every single widget of the window, key releases, programmatic focus changes, Tab /
  Shift+Tab, every printable character, special keys, window (re)activation, other windows
  of the application, widgets added later; a real in-process run; the isolated window.
* X11 (``x11``): real key presses on the real window - with the pointer anywhere, command
  keys, special keys - and another application that has the focus keeps its keys.
"""

from __future__ import annotations

import string
import time

import pytest
from htspec import keys, paths
from htspec.gui_state import all_widgets, fill_log
from htspec.live import PANELS, LiveApp
from htspec.models import make_setup
from htspec.probe_log import LINGER_CLOSE
from htspec.subproc import spawn_python
from PySide6.QtCore import QEvent, QObject
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QLineEdit, QVBoxLayout, QWidget

from hollywood_training.sync.events import TrainingState

pytestmark = pytest.mark.req("R2.7")

PRINTABLE = "".join(ch for ch in string.printable if ch.isprintable())  # incl. the space


def describe(widget) -> str:
    return f"{type(widget).__name__}({widget.objectName()!r})"


class KeySpy(QObject):
    """Counts the key events the command line itself receives."""

    def __init__(self, target) -> None:
        super().__init__(target)
        self.presses = 0
        self.releases = 0
        target.installEventFilter(self)

    def eventFilter(self, _watched, event) -> bool:
        if event.type() == QEvent.Type.KeyPress:
            self.presses += 1
        elif event.type() == QEvent.Type.KeyRelease:
            self.releases += 1
        return False


class OtherWindow(QWidget):
    """Another top-level window of the same application, with a text field."""

    def __init__(self) -> None:
        super().__init__()
        self.edit = QLineEdit(self)
        QVBoxLayout(self).addWidget(self.edit)
        self.setGeometry(30, 30, 300, 120)


@pytest.fixture
def other(qtbot):
    window = OtherWindow()
    qtbot.addWidget(window)
    return window


def activate(widget) -> bool:
    widget.activateWindow()
    widget.raise_()
    return QTest.qWaitForWindowActive(widget, 5000)


# ------------------------------------------------------------------- offscreen
def test_the_command_line_has_the_focus_as_soon_as_the_window_is_active(make_rig):
    rig = make_rig()  # shown and activated - nothing clicked
    assert rig.focus_widget() is rig.window.terminal.input


def test_typed_text_reaches_the_command_line_through_the_platform_input_path(rig):
    rig.type("pause")
    assert rig.input_text == "pause"
    rig.type("<Return>")
    assert rig.results[-1][1].command.name == "pause"


def test_keys_addressed_to_any_widget_of_the_window_reach_the_command_line(rig):
    line = rig.window.terminal.input
    lost = []
    for widget in all_widgets(rig.window):
        if widget is line:
            continue
        line.clear()
        QTest.keyClicks(widget, "ab1")
        QApplication.processEvents()
        if line.text() != "ab1" or QApplication.focusWidget() is not line:
            lost.append(f"{describe(widget)}: {line.text()!r}")
    assert lost == []


def test_key_releases_are_rerouted_too(rig):
    line = rig.window.terminal.input
    spy = KeySpy(line)
    for widget in (rig.window, rig.window.network, rig.window.chart, rig.window.status):
        QTest.keyClick(widget, "x")
    QApplication.processEvents()
    assert spy.presses == 4 and spy.releases == 4


def test_focus_given_to_another_widget_comes_back_to_the_command_line(rig):
    line = rig.window.terminal.input
    stolen = []
    for widget in all_widgets(rig.window):
        widget.setFocus()
        QApplication.processEvents()
        if QApplication.focusWidget() is not line:
            stolen.append(describe(widget))
    assert stolen == []


def test_tab_and_shift_tab_never_move_the_focus(rig):
    line = rig.window.terminal.input
    for sequence in ("<Tab>", "<Backtab>", "<S-Tab>", "<Tab><Tab><Tab>", "<C-Tab>"):
        rig.type(sequence)
        assert QApplication.focusWidget() is line, sequence


def test_every_printable_character_arrives_unchanged(rig):
    rig.type(keys.escape(PRINTABLE))
    assert rig.input_text == PRINTABLE


@pytest.mark.parametrize(
    "sequence",
    ["<Esc>", "<F1>", "<F5>", "<F10>", "<F12>", "<Insert>", "<C-s>", "<C-q>", "<C-w>",
     "<C-o>", "<A-f>", "<A-x>", "<M-a>", "<Left><Right><Up><Down>", "<Home><End>"],
)  # fmt: skip
def test_special_keys_neither_move_the_focus_nor_reach_another_widget(rig, sequence):
    line = rig.window.terminal.input
    rig.type(sequence)
    assert QApplication.focusWidget() is line
    assert rig.window.isVisible() and rig.window.isActiveWindow()
    rig.type("ok")
    assert rig.input_text.endswith("ok")


def test_command_and_scroll_keys_work_from_any_widget(rig):
    """Return, PgUp and Ctrl+Home addressed to other widgets still act in the terminal."""
    window = rig.window
    fill_log(window)
    bar = window.terminal.output.verticalScrollBar()
    QTest.keyClicks(window.network, "history")
    QTest.keyClick(window.chart, keys.SPECIAL["Return"])
    assert rig.results[-1][1].command.name == "history"
    top = bar.value()
    keys.send(window.status, "<PgUp>")
    assert bar.value() < top
    keys.send(window.usage, "<C-Home>")
    assert bar.value() == 0
    keys.send(window.header, "<C-End>")
    assert bar.value() == bar.maximum()


def test_reactivating_the_window_puts_the_focus_back_on_the_command_line(rig, other):
    other.show()
    assert activate(other)
    other.edit.setFocus()
    keys.send(keys.native(other), "elsewhere")
    assert other.edit.text() == "elsewhere"
    assert rig.input_text == ""
    assert activate(rig.window)
    assert QApplication.focusWidget() is rig.window.terminal.input
    rig.type("here")
    assert rig.input_text == "here"


def test_keys_stay_with_another_window_while_it_is_active(rig, other):
    """No reroute and no focus stealing while the user is on another window - even while
    the session goes on (epochs, state changes, log lines)."""
    other.show()
    assert activate(other)
    other.edit.setFocus()
    rig.feeder.epochs(2)
    rig.feeder.state(TrainingState.PAUSED, "paused")
    rig.feeder.log("log line")
    QApplication.processEvents()
    assert other.isActiveWindow() and not rig.window.isActiveWindow()
    keys.send(keys.native(other), "mine")
    assert other.edit.text() == "mine"
    assert rig.input_text == ""


def test_widgets_added_later_cannot_keep_the_keyboard(rig):
    line = rig.window.terminal.input
    newcomer = QLineEdit(rig.window.status)
    newcomer.show()
    QApplication.processEvents()
    newcomer.setFocus()
    QApplication.processEvents()
    assert QApplication.focusWidget() is line
    QTest.keyClicks(newcomer, "new")
    assert newcomer.text() == ""
    assert line.text() == "new"


def test_keys_reach_the_terminal_during_a_real_in_process_run(pilot):
    observed: dict = {}

    def type_while_training(window) -> None:
        assert activate(window)
        observed["focus"] = QApplication.focusWidget() is window.terminal.input
        QTest.keyClicks(window.network, "paus")  # addressed to another widget
        keys.send(keys.native(window), "e<Return>")  # the platform path

    pilot.on_epoch(2, type_while_training)
    pilot.on_state(TrainingState.PAUSED, pilot.after(0.3, pilot.line("resume")))
    pilot.on_finish(pilot.close)
    result = pilot.run(make_setup(epochs=5).process)
    assert observed == {"focus": True}
    assert result.epochs_completed == 5
    assert "ht ❯ pause" in pilot.final_log and "ht ❯ resume" in pilot.final_log


@pytest.mark.subprocess
def test_keys_reach_the_terminal_in_the_isolated_window(probe):
    import hollywood_training as ht

    steps = [
        {"on": "epoch", "value": 1, "do": "keys", "keys": "status<Return>", "target": "network"},
        {"on": "step", "value": 0, "do": "keys", "keys": "pause<Return>", "target": "chart",
         "delay_ms": 300},
        {"on": "state", "value": "paused", "do": "keys", "keys": "resume<Return>",
         "target": "native", "delay_ms": 500},
        LINGER_CLOSE,
    ]  # fmt: skip
    log = probe(script={"steps": steps})
    # ~40 ms per epoch: still training when the delayed "pause" arrives
    result = ht.run(make_setup(epochs=40, train_hook=lambda _b: time.sleep(0.005)).process)
    assert result.epochs_completed == 40
    commands = [r["value"] for r in log.sends("command")]
    assert commands == ["status", "pause", "resume"]
    states = [s for s in log.states() if s["visible"] and s["active"]]
    assert states and all(s["focus"]["class"] == "CommandLine" for s in states)
    assert any(s["session"]["state"] == "paused" for s in log.states())
    assert any(any(line.startswith("state ") for line in s["output"]["tail"]) for s in states)


# ---------------------------------------------------------------------- X11 level
@pytest.fixture(scope="module")
def live(xserver, xclient, tmp_path_factory):
    app = LiveApp(xserver, xclient, tmp_path_factory.mktemp("r2_7_live"), focus=False)
    yield app
    app.close()


@pytest.mark.x11
@pytest.mark.subprocess
def test_real_keys_reach_the_terminal_without_a_click(live):
    """Switching to the window (what the window manager does) is all it takes."""
    state = live.focus()
    assert state["focus"]["class"] == "CommandLine"
    live.enter("ping", expect="pong 1")


@pytest.mark.x11
@pytest.mark.subprocess
def test_real_keys_reach_the_terminal_wherever_the_pointer_is(live, xclient):
    live.focus()
    state = live.state()
    typed = ""
    for index, panel in enumerate(PANELS):
        xclient.move(*live.center(panel, state))
        letter = "abcdefgh"[index]
        live.type(letter)
        typed += letter
    xclient.move(5, 5)  # even outside the window, as long as the window has the focus
    live.type("z")
    typed += "z"
    time.sleep(0.3)
    try:
        assert live.state()["input"]["text"] == typed
    finally:
        for _ in typed:
            xclient.tap("BackSpace")


@pytest.mark.x11
@pytest.mark.subprocess
def test_real_command_keys_control_the_training(live):
    live.focus()
    live.enter("pause", expect="paus")
    live.wait(lambda r: r["session"]["state"] == "paused", after=live.now())
    live.enter("lr 0.0125", expect="learning rate")
    live.enter("resume", expect="resum")
    live.process.wait_message(
        lambda m: m.get("event") == "epoch" and m["lr"] == pytest.approx(0.0125), timeout=30
    )


@pytest.mark.x11
@pytest.mark.subprocess
def test_real_special_keys_act_in_the_terminal(live, xclient):
    live.focus()
    live.enter("history 1")
    xclient.tap("Up")  # history
    live.wait(lambda r: r["input"]["text"] == "history 1", after=live.now())
    xclient.tap("a", ctrl=True)
    xclient.tap("BackSpace")
    live.type("pin")
    xclient.tap("Tab")  # completion
    live.wait(lambda r: r["input"]["text"] == "ping ", after=live.now())
    xclient.tap("Escape")
    state = live.state()
    assert state["focus"]["class"] == "CommandLine" and state["input"]["text"] == "ping "
    xclient.tap("a", ctrl=True)
    xclient.tap("BackSpace")
    for _ in range(3):
        live.enter("help")
    bottom = live.state()["output"]["scroll"]
    xclient.tap("Prior")  # PgUp
    live.wait(lambda r: r["output"]["scroll"] < bottom, after=live.now())
    xclient.tap("End", ctrl=True)
    live.wait(lambda r: r["output"]["scroll"] == r["output"]["max"], after=live.now())
    xclient.tap("a", ctrl=True)
    xclient.tap("BackSpace")


@pytest.mark.x11
@pytest.mark.subprocess
def test_real_keys_belong_to_another_application_while_it_has_the_focus(live, xserver, xclient):
    neighbour = spawn_python(
        [str(paths.SCENARIO_DIR / "other_app.py"), "20", "20", "380", "220"],
        display=xserver.display,
    )
    try:
        ready = neighbour.wait_message(lambda m: m.get("event") == "ready", timeout=60)
        before = live.state()["input"]["text"]
        xclient.focus(ready["win_id"])
        time.sleep(0.3)
        start = len(neighbour.messages)
        x, y = live.center("network")
        xclient.move(x, y)  # the pointer over the Hollywood window does not matter
        live.type("theirs")
        time.sleep(0.4)
        typed = "".join(m["text"] for m in neighbour.messages[start:] if m["event"] == "key")
        assert typed == "theirs"
        assert live.state()["input"]["text"] == before
        live.focus()  # back on the Hollywood window: its terminal gets the keys again
        live.enter("ping", expect="pong")
    finally:
        neighbour.terminate()


@pytest.mark.x11
@pytest.mark.subprocess
def test_no_keyboard_grab_while_the_window_has_the_focus(live, xclient):
    live.focus()
    assert xclient.try_grab_keyboard() == 0
    live.type("q")
    assert xclient.try_grab_keyboard() == 0
    xclient.tap("BackSpace")

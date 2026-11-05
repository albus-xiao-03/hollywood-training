"""Window-only commands: ViewRequest from a command handler to MainWindow.handle_view."""

from __future__ import annotations

from multiprocessing import Pipe

import pytest

from hollywood_training.commands import (
    Command,
    CommandContext,
    CommandError,
    ViewError,
    ViewRequest,
    create_registry,
)
from hollywood_training.gui import MainWindow
from hollywood_training.gui.app import QtFrontend
from hollywood_training.gui.link import Link
from hollywood_training.gui.remote import RemoteFrontend
from hollywood_training.gui.window_process import WindowProcess
from hollywood_training.sync import SessionState, SyncChannel, TrainingControl, TrainingTarget


def _shout(*words: object) -> str:
    if not words:
        raise ViewError("nothing to shout")
    return " ".join(str(word) for word in words).upper()


def _context(frontend=None) -> CommandContext:
    registry = create_registry(
        [Command("shout", "test view command", lambda ctx: ctx.show(ViewRequest("shout", ("hi",))))]
    )
    return CommandContext(
        control=TrainingControl(1),
        session=SessionState(),
        registry=registry,
        target=TrainingTarget(None, None),
        frontend=frontend,
    )


def test_view_requests_need_a_frontend():
    result = _context().registry.execute("shout", _context())
    assert not result.ok and result.output == "shout is not available in this frontend"


def test_show_returns_the_outcome_or_nothing():
    class Frontend:
        def __init__(self, reply):
            self.reply, self.requests = reply, []

        def view(self, request):
            self.requests.append(request)
            return self.reply

    for reply, expected in (("done", "done"), (None, "")):
        frontend = Frontend(reply)
        assert _context(frontend).show(ViewRequest("shout", ("hi",))) == expected
        assert frontend.requests == [ViewRequest("shout", ("hi",))]


@pytest.fixture
def window(qtbot):
    win = MainWindow(SessionState())
    qtbot.addWidget(win)
    win._view_actions["shout"] = _shout  # a test action; features register their own
    return win


def test_the_window_dispatches_view_requests(window):
    assert window.handle_view(ViewRequest("shout", ("hi", 2))) == (True, "HI 2")
    assert window.handle_view(ViewRequest("shout")) == (False, "nothing to shout")
    assert window.handle_view(ViewRequest("nope")) == (False, "unknown window action 'nope'")


def test_the_in_process_frontend_runs_requests_on_its_window(window):
    frontend = QtFrontend(
        channel=SyncChannel(), control=TrainingControl(1), registry=create_registry()
    )
    with pytest.raises(CommandError, match="no window"):
        frontend.view(ViewRequest("shout", ("hi",)))
    frontend.window = window
    assert frontend.view(ViewRequest("shout", ("hi",))) == "HI"
    with pytest.raises(CommandError, match="nothing to shout"):
        frontend.view(ViewRequest("shout"))


def test_the_remote_frontend_sends_requests_to_the_window_process(qtbot):
    training_end, window_end = Pipe()
    remote = RemoteFrontend(
        channel=SyncChannel(), control=TrainingControl(1), registry=create_registry()
    )
    assert remote.view(ViewRequest("shout", ("hi",))) is None  # no link yet: nothing sent
    remote._link = Link(training_end)
    gui = WindowProcess(Link(window_end))
    qtbot.addWidget(gui.window)
    gui.window._view_actions["shout"] = _shout
    for request in (ViewRequest("shout", ("hi",)), ViewRequest("shout"), ViewRequest("nope")):
        assert remote.view(request) is None  # the window reports the outcome itself
        message = gui.link.receive()
        assert message == ("frontend", "view", request)
        gui._dispatch(message)
    text = gui.window.terminal.text().splitlines()
    assert text == ["HI", "nothing to shout", "unknown window action 'nope'"]

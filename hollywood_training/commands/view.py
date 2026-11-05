"""``ViewRequest``: a command for the window itself rather than for the training run.

Commands run where the model lives: on a command thread of the training process for the
default window process, on the GUI thread for the in-process window. A few commands only
change what the window shows or does - the keyboard replacements for the R1 mouse
interactions (SPEC R2). Their handlers validate the arguments and hand a
:class:`ViewRequest` to :meth:`CommandContext.show
<hollywood_training.commands.context.CommandContext.show>`. The frontend passes the request
to ``MainWindow.handle_view``, across the process link when the window has its own
process, and the window carries it out.

A request is plain, picklable data and needs neither torch nor Qt. The window rejects a
request it cannot carry out (for example a neuron that is not drawn) by raising
:class:`ViewError`; the message becomes the command's error output.
"""

from __future__ import annotations

from dataclasses import dataclass


class ViewError(Exception):
    """The window cannot carry out a view request; the message explains why."""


@dataclass(frozen=True, slots=True)
class ViewRequest:
    """One window-only action: its name and its (already validated) arguments."""

    action: str
    arguments: tuple[object, ...] = ()

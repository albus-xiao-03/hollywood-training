"""``SyncChannel``: the hand-off between the training thread and a frontend.

The training thread calls :meth:`SyncChannel.deliver` for synchronous events and blocks
until the frontend calls :meth:`SyncChannel.acknowledge` (after applying and drawing the
event), or until the channel is closed. :meth:`SyncChannel.notify` sends without waiting.

A frontend connects a *sink*: a callable that receives every :class:`Envelope` on the
training thread and must hand it over to the frontend's own thread (a Qt queued signal,
for instance) or handle it immediately (a console frontend). The channel knows nothing
about Qt, so the training core can be used and tested without a GUI.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from dataclasses import dataclass

from .events import Event


@dataclass(frozen=True, slots=True, eq=False)
class Envelope:
    """An event plus its sequence number and whether the sender waits for it."""

    sequence: int
    event: Event
    synchronous: bool


Sink = Callable[[Envelope], None]


class SyncChannel:
    """Ordered, optionally blocking delivery of events to one frontend."""

    def __init__(self) -> None:
        self._condition = threading.Condition()
        self._sequence = 0
        self._acknowledged: set[int] = set()
        self._closed = False
        self._sink: Sink | None = None

    def connect(self, sink: Sink) -> None:
        """Attach the frontend's sink (replacing any previous one)."""
        with self._condition:
            self._sink = sink

    @property
    def closed(self) -> bool:
        return self._closed

    def deliver(self, event: Event) -> bool:
        """Send ``event`` and block until it is acknowledged.

        Returns True once acknowledged, False if the channel is (or becomes) closed first.
        """
        envelope, sink = self._prepare(event, synchronous=True)
        if envelope is None:
            return False
        sink(envelope)
        with self._condition:
            while envelope.sequence not in self._acknowledged and not self._closed:
                self._condition.wait()
            acknowledged = envelope.sequence in self._acknowledged
            self._acknowledged.discard(envelope.sequence)
            return acknowledged

    def notify(self, event: Event) -> bool:
        """Send ``event`` without waiting. Returns False if the channel is closed."""
        envelope, sink = self._prepare(event, synchronous=False)
        if envelope is None:
            return False
        sink(envelope)
        return True

    def acknowledge(self, sequence: int) -> None:
        """Called by the frontend once a synchronous envelope has been fully applied."""
        with self._condition:
            self._acknowledged.add(sequence)
            self._condition.notify_all()

    def close(self) -> None:
        """Stop delivering; wake any sender that is waiting for an acknowledgement."""
        with self._condition:
            self._closed = True
            self._condition.notify_all()

    def _prepare(self, event: Event, *, synchronous: bool) -> tuple[Envelope | None, Sink]:
        with self._condition:
            if self._sink is None:
                raise RuntimeError("SyncChannel has no frontend connected")
            if self._closed:
                return None, self._sink
            self._sequence += 1
            return Envelope(self._sequence, event, synchronous), self._sink

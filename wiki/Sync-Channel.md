# Sync Channel

`hollywood_training/sync/channel.py` · part of [Synchronization](Synchronization)

`SyncChannel` is the hand-off between the training thread and one frontend. It numbers
every event, passes it to the frontend's *sink*, and for synchronous events blocks the
sender until the frontend acknowledges that sequence number or the channel is closed.
It knows nothing about Qt, so the training core can be used and tested without a GUI.

## Responsibilities

* Wrap each event in an `Envelope` with an increasing sequence number (starting at 1).
* Call the connected sink on the sender's thread.
* Block `deliver` until `acknowledge(sequence)` or `close()`; never block `notify`.
* After `close()`, drop events and release a waiting sender.

## API

```python
@dataclass(frozen=True, slots=True, eq=False)
class Envelope:
    sequence: int
    event: Event
    synchronous: bool
```

An event plus its sequence number and whether the sender waits for it: `synchronous` is
`True` for `deliver`, `False` for `notify`.

```python
Sink = Callable[[Envelope], None]
```

The frontend's receiver. It is called on the training thread for every envelope and must
handle it right away (console) or hand it to the frontend's own thread or process.

```python
class SyncChannel:
    def __init__(self) -> None: ...
    def connect(self, sink: Sink) -> None: ...
    @property
    def closed(self) -> bool: ...
    def deliver(self, event: Event) -> bool: ...
    def notify(self, event: Event) -> bool: ...
    def acknowledge(self, sequence: int) -> None: ...
    def close(self) -> None: ...
```

| Member | Behaviour |
|---|---|
| `connect(sink)` | Attaches the frontend's sink, replacing any previous one. |
| `closed` | `True` after `close()`; a channel cannot be reopened. |
| `deliver(event)` | Sends a synchronous envelope and blocks. Returns `True` once acknowledged, `False` if the channel is closed first (the sink is not called when it was already closed). No timeout. |
| `notify(event)` | Sends an asynchronous envelope. Returns `True`, or `False` without calling the sink when the channel is closed. |
| `acknowledge(sequence)` | Marks `sequence` as fully applied and wakes the waiting sender. Callable from any thread, including from inside the sink. |
| `close()` | Stops delivering and wakes a sender waiting for an acknowledgement. |

Raises: `deliver` and `notify` raise `RuntimeError("SyncChannel has no frontend
connected")` when no sink was connected, even after `close()`. An exception raised by
the sink propagates to the sender.

## How it works

* One `threading.Condition` guards the sequence counter, the set of acknowledged
  sequence numbers, the closed flag and the sink.
* Under the lock, `_prepare` checks for a sink, returns no envelope when closed, and
  otherwise increments the counter and builds the `Envelope`.
* The sink is called outside the lock. A sink may therefore call `acknowledge` before it
  returns, which the console and in-process frontends do.
* `deliver` then waits until its sequence is in the acknowledged set or the channel is
  closed, removes the sequence from the set and returns whether it was acknowledged.

## Used by / uses

* Uses `Event` ([Sync Events](Sync-Events)).
* [Runner](Runner): `run()` creates one channel per session.
* [Training Manager](Training-Manager): `deliver` for `SessionStarted`, `EpochCompleted`,
  `TrainingPaused` and `SessionFinished`, `notify` for the others. When one of the first
  three returns `False`, it calls `control.stop()`.
* [Console Frontend](Console-Frontend) and [GUI In-Process Frontend](GUI-In-Process-Frontend):
  the sink applies, prints or repaints, then acknowledges.
* [GUI Remote Frontend](GUI-Remote-Frontend): the sink forwards the envelope to the
  [window process](GUI-Window-Process). The link thread calls `acknowledge` for every
  `("ack", seq)` and `close()` when the window is closed or the link is lost.

## Example

A frontend thread that receives envelopes through a queue:

```python
import queue
import threading

from hollywood_training.sync import LogMessage, SyncChannel

channel = SyncChannel()
inbox: queue.Queue = queue.Queue()
channel.connect(inbox.put)                    # sink: hand over to the frontend thread

def frontend() -> None:
    while (envelope := inbox.get()) is not None:
        print("draw", envelope.event)         # apply and draw first ...
        if envelope.synchronous:
            channel.acknowledge(envelope.sequence)   # ... then release the sender

threading.Thread(target=frontend, daemon=True).start()

# Any event can be delivered; the manager delivers only the four synchronous ones.
assert channel.deliver(LogMessage("drawn before deliver returns"))
assert channel.notify(LogMessage("no acknowledgement needed"))
channel.close()
assert channel.deliver(LogMessage("too late")) is False   # sink not called
inbox.put(None)
```

## Notes

* Whether a send waits is the sender's choice, and `Envelope.synchronous` records it.
  A frontend acknowledges based on that flag, not on the event type.
* A sink that fails must still acknowledge, or the training thread stays blocked. The
  bundled frontends acknowledge in a `finally` block.
* Only `acknowledge` or `close` ends a wait. The window-process frontend closes the
  channel (and requests a stop) when the window is closed or the link breaks, which
  releases a waiting training thread.

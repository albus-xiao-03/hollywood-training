# Synchronization

`hollywood_training/sync/` · layer of the [Architecture](Architecture)

The `sync` package is the contract between the training thread and a frontend. It
defines what the training thread reports (events), how a report is handed over (a
channel that can block until the frontend has drawn it), how requests travel back (a
thread-safe control object), how a frontend that shares the training thread gets time
to run (event pumps), and the state a frontend assembles from the events. It imports
neither Qt nor torch.

## What crosses which way

```
 training thread (TrainingManager)                       frontend
 ─────────────────────────────────                       ──────────────────────────────────
 SyncChannel.deliver(event) ── Envelope(seq, event, True) ──▶ sink → SessionState.apply → draw
   … blocks until …          ◀──────────── acknowledge(seq) ─┘ (only after drawing)
 SyncChannel.notify(event)  ── Envelope(seq, event, False) ─▶ sink (no reply)
 safe point between batches ◀── TrainingControl: pause · resume · step · stop · epochs ·
                                delay · submit(action)  ◀── commands, window close, Ctrl+C
 EventPump.poll() / wait()  ──▶ a frontend on the same thread handles its own events
```

| Direction | What | How | Page |
|---|---|---|---|
| training → frontend | events, each wrapped in an `Envelope` with a sequence number | `SyncChannel.deliver` (waits) or `SyncChannel.notify` (does not wait); both call the frontend's sink | [Sync Events](Sync-Events), [Sync Channel](Sync-Channel) |
| frontend → training | acknowledgement of a synchronous envelope, after it was applied and drawn | `SyncChannel.acknowledge(sequence)` | [Sync Channel](Sync-Channel) |
| frontend → training | requests (pause, resume, step, stop, total epochs, delay) and queued actions `action(TrainingTarget) -> str \| None` | `TrainingControl`; the manager applies them at its next safe point | [Sync Control](Sync-Control) |
| training → frontend, same thread | time to process input and animate | `EventPump.poll()` between batches, `EventPump.wait(timeout)` while idle | [Sync Event Pump](Sync-Event-Pump) |
| inside the frontend | the frontend's picture of the session | `SessionState.apply(event)` | [Sync Session State](Sync-Session-State) |

## Synchronous and asynchronous events

| Event | Synchronous | Sent with | When |
|---|---|---|---|
| `SessionStarted` | yes | `deliver` | once, before the first epoch (initial weights) |
| `EpochCompleted` | yes | `deliver` | after every epoch (new weights, losses, …) |
| `TrainingPaused` | yes | `deliver` | when a pause takes effect (current weights) |
| `SessionFinished` | yes | `deliver` | once, at the end (final weights, outcome) |
| `BatchProgress` | no | `notify` | during an epoch, throttled |
| `StateChanged` | no | `notify` | state transitions and setting changes |
| `LogMessage` | no | `notify` | text for the frontend's log or terminal |

For a synchronous event the training thread blocks in `deliver` until the frontend
acknowledges the envelope's sequence number, or until the channel is closed. The GUI
frontends acknowledge only after they have applied the event and repainted the affected
panels (the console after printing), so epoch N+1 cannot start before epoch N is on
screen. An asynchronous event costs the training thread only the sink call.

## Frontends

| Frontend | The sink runs on | Acknowledged by | Pump |
|---|---|---|---|
| [Console](Console-Frontend) (`gui=False`) | training thread: apply, print | the sink itself, immediately | `WaitingPump` |
| [In-process window](GUI-In-Process-Frontend) (`isolate_gui=False`) | training thread (the main thread): apply, repaint | the sink itself, after repainting | `QtPump` |
| [Window process](GUI-Remote-Frontend) (default) | training thread: apply to a mirror `SessionState`, send `("event", envelope)` over the [link](GUI-Link-Protocol) | the [window process](GUI-Window-Process) replies `("ack", seq)` after drawing; the training process's link thread calls `acknowledge` | `WaitingPump` |

## Modules

| Module | Page | Contents |
|---|---|---|
| `events.py` | [Sync Events](Sync-Events) | `TrainingState`, `Phase`, `LogLevel`, `EpochRecord`, `best_epoch`, `SessionInfo`, the seven events, `Event`, `SYNCHRONOUS_EVENTS`, `is_synchronous` |
| `channel.py` | [Sync Channel](Sync-Channel) | `Envelope`, `Sink`, `SyncChannel` |
| `control.py` | [Sync Control](Sync-Control) | `TrainingTarget`, `ControlAction`, `TrainingControl` |
| `pump.py` | [Sync Event Pump](Sync-Event-Pump) | `EventPump`, `WaitingPump` |
| `session.py` | [Sync Session State](Sync-Session-State) | `SessionState` |

## Package exports

`sync/__init__.py` imports all five modules eagerly and exports (`__all__`):
`SYNCHRONOUS_EVENTS`, `BatchProgress`, `ControlAction`, `Envelope`, `EpochCompleted`,
`EpochRecord`, `Event`, `EventPump`, `LogLevel`, `LogMessage`, `Phase`,
`SessionFinished`, `SessionInfo`, `SessionStarted`, `SessionState`, `Sink`,
`StateChanged`, `SyncChannel`, `TrainingControl`, `TrainingPaused`, `TrainingState`,
`TrainingTarget`, `WaitingPump`, `best_epoch`, `is_synchronous`.

The package is torch-free: besides the standard library it imports only the data modules
`tracking.architecture` and `tracking.weights` (numpy). The window process imports it
and unpickles envelopes without loading PyTorch. `EpochRecord` and `TrainingState` are
also exported by the top-level `hollywood_training` package.

## Example

A minimal frontend. The [console frontend](Console-Frontend) has the same structure and
prints as well.

```python
from hollywood_training.sync import Envelope, SessionState, SyncChannel, TrainingControl
from hollywood_training.training import TrainingManager

channel = SyncChannel()
control = TrainingControl(process.epochs)      # process: a ht.TrainingProcess
session = SessionState()

def sink(envelope: Envelope) -> None:          # called on the training thread
    try:
        session.apply(envelope.event)
        ...                                    # draw or print from `session`
    finally:
        if envelope.synchronous:               # the training thread is waiting
            channel.acknowledge(envelope.sequence)

channel.connect(sink)
result = TrainingManager(process, channel=channel, control=control).run()
```

`run()` wires the same objects for you; see [Runner](Runner) and
[Training Manager](Training-Manager).

# Link Protocol

`hollywood_training/gui/link.py` · part of [GUI](GUI)

The link connects the training process with the window process. `Link` is a thin wrapper
around a `multiprocessing.connection.Connection`, which is an authenticated local socket
on Linux and macOS and a named pipe on Windows. It exchanges small picklable tuples and
serialises sends from several threads. The module is torch-free, and so is everything
that is sent over it.

## Responsibilities

* Define the environment variable that carries the authentication key.
* Send each message as one tuple, from any thread, without interleaving.
* Report every way of losing the other side in one form: `send()` returns `False`, and
  `receive()` raises `EOFError`.

## API

```python
AUTHKEY_ENV = "HOLLYWOOD_TRAINING_AUTHKEY"
```

The environment variable through which the training process passes the hex-encoded key
to the window process.

```python
class Link:
    def __init__(self, connection: Connection) -> None

    @property
    def closed(self) -> bool
    def send(self, *message: object) -> bool
    def receive(self) -> tuple
    def poll(self, timeout: float) -> bool
    def close(self) -> None
```

| Member | Behaviour |
|---|---|
| `send(*message)` | Sends the tuple `message` while holding a lock. Once the link is closed it returns `False` without sending. An `OSError`, `EOFError` or `ValueError` marks the link closed and returns `False`. |
| `receive()` | Blocks until the next message arrives. Raises `EOFError` when the other side is gone (an `OSError` or `ValueError` is converted to it). |
| `poll(timeout)` | `Connection.poll(timeout)`. Returns `False` on `OSError`, `ValueError` or `EOFError`. |
| `close()` | Marks the link closed and closes the connection, ignoring an `OSError`. |
| `closed` | `True` after `close()` or after a failed send. |

### Messages

Every message is a tuple whose first item names its kind. Training process to window
process:

| Message | Sent by (thread) | Meaning and handling |
|---|---|---|
| `("event", Envelope)` | the channel sink `_forward` (training thread) | a session event ([Sync Events](Sync-Events)). Synchronous ones are acknowledged after drawing |
| `("usage", UsageSample)` | the sampler (`hollywood-training-usage`), every 0.5 s | a CPU, memory and GPU reading for the [Usage Panel](GUI-Usage-Panel) |
| `("result", ok, text)` | command execution (`hollywood-training-commands`) | the outcome of a command line, shown in the terminal |
| `("completions", [names])` | start-up, once after `ready` | command names and aliases for Tab completion |
| `("frontend", action, argument)` | command handlers, through `FrontendActions` | `"clear"` or `"close"` (argument `None`), `"screenshot"` (an absolute path), or `"view"` (a [`ViewRequest`](Commands-View), carried out by the window) |
| `("close",)` | `RemoteFrontend.run()` at the end | close the window because the run is over. The window waits 400 ms first |

Window process to training process:

| Message | Sent when | Handling in the training process |
|---|---|---|
| `("ready",)` | the window is shown | start-up is complete |
| `("ack", sequence)` | a synchronous event has been applied and drawn | `SyncChannel.acknowledge(sequence)` releases the waiting `deliver()` |
| `("command", line)` | the user entered a command line | queued for the command thread, which runs `CommandRegistry.execute` and replies `("result", ...)` |
| `("closed",)` | the window was closed | the receiving loop ends and training is stopped |

`("disconnected",)` is not a link message. The window process's receiving thread
creates it locally when `receive()` raises `EOFError` ([Window Process](GUI-Window-Process)).

## How it works

* **Connection.** The [Remote Frontend](GUI-Remote-Frontend) opens a
  `Listener(authkey=secrets.token_bytes(32))` on a random address. It passes the address
  as `--address`, and the key as hex in `HOLLYWOOD_TRAINING_AUTHKEY`. The child connects
  with `Client(address, authkey=key)`. `multiprocessing` authenticates the two ends with
  that key before any message is exchanged, and a new key is made for every window.
* **Serialisation.** `Connection.send` pickles the tuple. Apart from strings, booleans
  and lists, the payloads are an `Envelope` (frozen slot dataclasses with numpy arrays)
  and a `UsageSample`. Neither needs torch, so the window process can unpickle them
  without loading PyTorch ([Architecture](Architecture)).
* **Threads.** Three threads of the training process send. The caller's thread, which
  also runs training, sends events, the completions and `close`. The command thread
  sends results and frontend actions, and the usage thread sends readings. In the window
  process only the GUI thread sends. Each process has exactly one receiving thread, which
  sends nothing. The lock in `send()` keeps messages from interleaving. `receive()` takes
  no lock.
* **Order.** Messages from one process arrive in the order they were sent. The window
  therefore applies events in sequence, and each `ack` follows the drawing of its own
  event.

## Used by / uses

* Used by the [Remote Frontend](GUI-Remote-Frontend) (training side) and the
  [Window Process](GUI-Window-Process) (window side).
* Uses only `multiprocessing.connection`, `threading` and `contextlib`.

## Example

`Link` wraps any `Connection`, for instance one end of a `Pipe`:

```python
from multiprocessing import Pipe

from hollywood_training.gui.link import Link

a, b = Pipe()
training, window = Link(a), Link(b)

window.send("ack", 7)                   # sends the tuple ("ack", 7)
assert training.poll(1.0) and training.receive() == ("ack", 7)

window.close()
assert window.send("closed") is False   # a closed link sends nothing
try:
    training.receive()
except EOFError:
    print("the window process is gone")
```

## Notes

* `poll()` also returns `True` when the other side has closed. The next `receive()` then
  raises `EOFError`. The remote frontend relies on this during start-up to detect a
  child that connected and then failed.
* Messages are pickled. The random per-window key limits the connection to the child
  process that received it.

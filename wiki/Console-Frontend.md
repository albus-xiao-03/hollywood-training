# Console Frontend

`hollywood_training/console.py` · part of [Architecture](Architecture)

`ConsoleFrontend` is the plain-text frontend that `run(process, gui=False)` uses on
machines without a display: servers, CI, containers. It handles every event right away
on the training thread. Synchronous events are therefore acknowledged at once, and
training runs at full speed. It prints a header, one line per selected epoch, state and
log messages, and a footer with the outcome.

## Responsibilities

* Act as the channel's sink: fold every event into a `SessionState` and print it.
* Acknowledge synchronous envelopes immediately, so the hand-off never waits.
* Keep the output short by printing only every n-th epoch.

## API

```python
class ConsoleFrontend:
    def __init__(
        self, channel: SyncChannel, *, stream: TextIO | None = None, log_every: int | None = None
    ) -> None

    session: SessionState

    def handle(self, envelope: Envelope) -> None
```

| Parameter | Meaning |
|---|---|
| `channel` | the channel to listen on. The constructor calls `channel.connect(self.handle)`, replacing any previous sink |
| `stream` | where to write; `None` means `sys.stdout`, looked up on every write |
| `log_every` | print every n-th epoch; by default about 25 lines per run. Values below 1 raise `ValueError("log_every must be at least 1")` |

* `session` is the [`SessionState`](Sync-Session-State) built from every event received.
* `handle(envelope)` applies the event to `session`, prints it, and finally (in a
  `finally` block) acknowledges the envelope if it is synchronous.

## How it works

`SyncChannel` calls the sink on the sending thread, so all output happens on the
training thread, inside `deliver()` / `notify()`. Each line is written with `print(...,
flush=True)`.

| Event | Output |
|---|---|
| `SessionStarted` | header: a rule of 72 `=`, `HOLLYWOOD TRAINING :: <class>[ - <display name>]`, `epochs … \| batch size … \| lr … \| device …`, `loss … \| optimizer …[ (<details>)]`, `normalisation …`, another rule |
| `EpochCompleted` | `formatting.epoch_summary(record, total_epochs)` for epoch 1, every `every`-th epoch, and the last epoch; `every` is `log_every`, or `max(1, total_epochs // 25)` |
| `StateChanged` | `[<state>] <message>`, only when the message is not empty |
| `LogMessage` | the text, prefixed by level: none for info, `ok: `, `warning: `, `error: ` |
| `SessionFinished` | a rule of 72 `-`, `<STATE>: <reason> in <HH:MM:SS.mmm>`, then `best loss <loss> at epoch <n>` if a completed epoch has a finite loss, then `error: <error>` if there was one |
| `BatchProgress`, `TrainingPaused` | nothing; they only update `session` |

The best loss line shows `best.monitored_loss` for the record returned by
`session.best_record()`. That method delegates to `best_epoch()`
([Sync Events](Sync-Events)), which picks the lowest finite validation loss, or the
training loss for epochs without validation. Numbers are formatted by
[Formatting](Formatting). With `run(..., gui=False)` the manager uses the default
`WaitingPump` and does not linger, so the run ends after the last epoch.

Sample output (a 60-epoch run, middle lines elided):

```
========================================================================
HOLLYWOOD TRAINING :: SmallNet
epochs 60 | batch size 16 | lr 0.01 | device cpu (10 threads)
loss CrossEntropyLoss (mean) | optimizer Adam (betas=0.9/0.999)
normalisation data: z-score · layers: none
========================================================================
training SmallNet for 60 epochs on cpu (10 threads)
epoch  1/60 | train 0.6897 | val 0.6827 | accuracy 46.9% | lr 0.01 | 0.009s
epoch  2/60 | train 0.6800 | val 0.6758 | accuracy 40.6% | lr 0.01 | 0.007s
epoch  4/60 | train 0.6679 | val 0.6637 | accuracy 54.2% | lr 0.01 | 0.007s
…
epoch 60/60 | train 0.2713 | val 0.2671 | accuracy 92.7% | lr 0.01 | 0.008s
------------------------------------------------------------------------
FINISHED: completed 60 epochs in 00:00:00.409
best loss 0.2671 at epoch 60
```

When a step raises, the traceback is printed with the `error: ` prefix, and the footer
reads, for example, `FAILED: failed after 0 completed epochs in …` followed by
`error: RuntimeError: bad batch`.

## Used by / uses

* Uses: [Sync Channel](Sync-Channel) (`SyncChannel`, `Envelope`),
  [Sync Events](Sync-Events), [Sync Session State](Sync-Session-State) and
  [Formatting](Formatting).
* Used by: [Runner](Runner), when `gui=False`.

## Example

`run(process, gui=False, log_every=10)` is the usual entry point. The frontend can also
drive a manager directly, for example to capture the output:

```python
import io
from hollywood_training.console import ConsoleFrontend
from hollywood_training.sync import SyncChannel
from hollywood_training.training import TrainingManager

channel = SyncChannel()
buffer = io.StringIO()
ConsoleFrontend(channel, stream=buffer, log_every=5)    # connects itself
result = TrainingManager(process, channel=channel).run()
print(buffer.getvalue())
```

## Notes

* There is no terminal, so no commands. `run()` rejects `start_paused` with
  `gui=False`, and custom commands are not used in this mode.
* Ctrl+C prints `warning: interrupted` and a `STOPPED: …` footer, then the
  `KeyboardInterrupt` propagates out of `run()`.
* Printing happens inside the channel call on the training thread. If writing to the
  stream raises, the exception propagates into the manager; the envelope is still
  acknowledged.

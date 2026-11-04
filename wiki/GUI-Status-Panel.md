# Status Panel

`hollywood_training/gui/widgets/status_panel.py` · part of [GUI](GUI)

`StatusPanel` shows everything about the run (SPEC GUI 2). At the top are the current and
total epoch, the process timer, an epoch progress bar with the ETA, and a batch progress
bar. Below them are key/value rows for batch size, learning rate, loss function,
optimizer, normalization strategy, scheduler, device, losses, metrics, best epoch, epoch
time, parameter count and epoch delay. The panel reads the
[session state](Sync-Session-State) on every paint. The timer therefore ticks smoothly
between events, while the numbers always match the last synchronised epoch.

## Responsibilities

* Show every training-status field the SPEC requires, plus the scheduler, device,
  metrics and the best epoch.
* Keep the process timer moving between events, using `SessionState.elapsed_now()`.
* Fit a variable number of rows (metrics depend on the step functions) into the panel.

## API

```python
Row = tuple[tuple[str, str], ...]  # one or two (label, value) pairs
```

```python
class StatusPanel(HudPanel):
    def __init__(
        self, session: SessionState, clock: AnimationClock, parent: QWidget | None = None
    ) -> None

    def rows(self) -> list[Row]
    def paint_content(self, painter: QPainter, rect: QRectF) -> None
```

* The title is "Training status", the code `STS-03` and the minimum size 360 × 300. The
  panel repaints on every clock tick.
* `rows()` returns the key/value rows shown below the progress bars. It returns an empty
  list before a session has started.
* `paint_content()` draws the panel as described below. Before a session it shows
  `STANDING BY`.

## How it works

**Top section:**

| Element | Content | Source |
|---|---|---|
| `EPOCH` counter | completed and total epochs, such as `077/300`, zero-padded to the width of the total, in glowing 16 pt digits | `session.epoch`, `session.total_epochs` |
| `PROCESS TIMER` | active training time as `HH:MM:SS.mmm`, in glowing digits | `session.elapsed_now()`, which advances between events while `RUNNING` |
| `RUN` bar | the completed fraction of all requested epochs, such as `RUN 25.7%` | `session.epoch_progress()` |
| ETA | `ETA HH:MM:SS` while running, after the first epoch. Otherwise `PAUSED`, `COMPLETE`, `STOPPED`, `FAILED` or `ETA —` | `session.eta()`, `session.state` |
| `BATCH` bar | `BATCH 12/23 · TRAINING` or `· VALIDATION` (`?` for an unknown total), and `BATCH —` between epochs. Striped when the batch count is unknown, while running or inside an epoch. The fill ends in indigo during validation | `session.batch`, `total_batches`, `phase`, `batch_progress()` |

**Rows** (`rows()`), in order:

| Row | Left pair | Right pair |
|---|---|---|
| 1 | `BATCH SIZE` | `LEARNING RATE` |
| 2 | `LOSS FUNCTION`, such as `CrossEntropyLoss (mean)` | |
| 3 | `OPTIMIZER`, name plus details, such as `Adam (betas=0.9/0.999)` | |
| 4 | `NORMALIZATION`, such as `data: z-score · layers: BatchNorm1d` | |
| 5 | `SCHEDULER` (`none` without one) | `DEVICE` |
| 6 | `TRAIN LOSS` | `VAL LOSS` (`n/a` without validation data) |
| 7 and on | training metrics, then `VAL <metric>` validation metrics, two pairs per row | |
| once an epoch has a finite loss | `BEST LOSS`, the best record's `monitored_loss` as `<loss> @ <epoch>` | `EPOCH TIME`, the last epoch's duration in seconds |
| last | `PARAMETERS` | `EPOCH DELAY`, in ms |

The values come from `SessionInfo` ([Training Descriptions](Training-Descriptions)),
from the session's latest record, and from `SessionState.best_record()`, which is the
record with the lowest finite `monitored_loss` (validation loss, or training loss when
an epoch has none). They are formatted by
[Formatting](Formatting): `count`, `learning_rate` and `loss`, plus `metric`, which shows
ratio-like metrics such as `accuracy` as percentages.

**Layout.** Each key column is as wide as its longest label plus 12 px. A row with one
pair spans the full width, and a row with two pairs splits it in half. Values that are
too long are elided in the middle. The row height lies between the font height + 3 px and
22 px. Rows that do not fit are not drawn.

**Cost.** The panel repaints completely on every frame. Its glowing digits use
`draw_glow_mono`, which blits cached per-character sprites, so a timer that changes every
frame stays cheap ([Theme](GUI-Theme)). The bars are drawn with `paint_bar`
([HUD Panel](GUI-HUD-Panel)).

## Used by / uses

* Used by the [Main Window](GUI-Main-Window) as `window.status`. It is repainted by
  `render_now()` during every synchronous hand-off.
* Uses [Session State](Sync-Session-State), [Sync Events](Sync-Events) (`Phase`,
  `TrainingState`, `EpochRecord.monitored_loss`), [HUD Panel](GUI-HUD-Panel),
  [Theme](GUI-Theme) and [Formatting](Formatting).

## Example

```python
from hollywood_training.gui import AnimationClock
from hollywood_training.gui.widgets import StatusPanel

panel = StatusPanel(session, AnimationClock())   # session: a SessionState after SessionStarted
for row in panel.rows():
    print(row)
# (('BATCH SIZE', '16'), ('LEARNING RATE', '0.01'))
# (('LOSS FUNCTION', 'CrossEntropyLoss (mean)'),)
# (('OPTIMIZER', 'Adam (betas=0.9/0.999)'),)
# (('NORMALIZATION', 'layers: none'),)
# (('SCHEDULER', 'none'), ('DEVICE', 'cpu (10 threads)'))
# (('TRAIN LOSS', '—'), ('VAL LOSS', 'n/a'))
# (('PARAMETERS', '183'), ('EPOCH DELAY', '0 ms'))
```

## Notes

* The counters show completed epochs. While epoch 5 is running, the counter reads
  `004/…` and the `BATCH` bar shows the progress inside epoch 5.
* The panel reads only `SessionState` and keeps no copy of it, so it always shows what
  the frontend has applied so far.

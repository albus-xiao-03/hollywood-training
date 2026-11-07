# Network View

`hollywood_training/gui/widgets/network_view.py` · part of [GUI](GUI)

`NetworkView` draws the tracked network (SPEC GUI 1): columns of neurons, the connections
between them drawn from each layer's real connection matrix, the neurons' biases, and the
operations of each layer. The recurrent connections of a recurrent layer are drawn too,
as arcs beside its column. The drawing is rebuilt from the new weight snapshot at every
epoch hand-off, so the lines follow the real training updates. An animation overlay,
with glowing neurons and signal particles that prefer strong weights, runs on top of a
cached pixmap. The `neuron` command highlights one drawn neuron with a pulsing ring and a
box showing its index, label and bias.

## Responsibilities

* Turn an `ArchitectureInfo` and a `WeightSnapshot` into columns and links
  (`build_columns`).
* Lay them out and render the static layer (frame, connections, recurrent connections,
  neurons, labels) into a pixmap. Rebuild it only when the architecture, the weights, the size or the device
  pixel ratio change.
* Animate halos and signals every frame.
* Highlight a selected neuron (`select_neuron`) and describe it (`neuron_lines`).
* Record which epoch's weights were painted last (`last_painted_epoch`).

## API

```python
LEVELS = 12  # strength buckets used to batch connection lines by pen
HOT_LEVELS = 3  # how strongly a connection changed in the last epoch
STATE_SPEED = {
    TrainingState.RUNNING: 1.0,
    TrainingState.PAUSED: 0.12,
    TrainingState.FINISHED: 0.35,
}
```

`STATE_SPEED` scales the signal speed by training state. Other states use 0.5.

```python
@dataclass(slots=True)
class Column:
    indices: np.ndarray  # feature indices shown
    size: int | None  # total features
    title: str
    subtitle: str
    caption: str
    badges: tuple[str, ...] = ()
    labels: tuple[str, ...] | None = None
    bias: np.ndarray | None = None
    kind: str = "hidden"  # "input" | "hidden" | "output"
    x: float = 0.0
    ys: np.ndarray = field(default_factory=lambda: np.zeros(0))
    layer: str | None = None  # the layer whose outputs the column shows
    recurrence: Recurrence | None = None  # that layer's recurrent connections
    loop_source: int | None = None  # the column the recurrence comes from

@dataclass(slots=True)
class Link:
    name: str
    source: int
    target: int
    weights: LayerWeights | None
    bridge: bool = False
```

A `Column` is one drawn column of neurons. The layout sets `x` and `ys` (one y per shown
neuron). A `Link` connects column `source` to column `target`. It holds either a layer's
weights, or, with `bridge=True`, a bridge between one layer's output and the next
layer's differently shaped input: a reshape, or the attention after an attention layer
(the bridge's `name`, `"reshape"` or `"attention"`). This `Link` is unrelated to
`hollywood_training.gui.link.Link` ([Link Protocol](GUI-Link-Protocol)).

```python
def build_columns(
    architecture: ArchitectureInfo, weights: WeightSnapshot | None, limit: int
) -> tuple[list[Column], list[Link]]
```

For each layer in `architecture.layers`:

* The shown indices are the layer's `LayerWeights.in_index` and `out_index` when
  `weights` contains the layer. Otherwise they are `display_indices(features or 1, limit)`
  ([Weight Data](Tracking-Weight-Data)).
* An input column is added when there is no previous column, or when the previous
  column's size or indices differ from the layer's input. The first one is `INPUT`
  ("network input", with the input operations as badges, `input_labels`, kind `input`).
  Later ones are `<LAYER> IN` ("reshaped input"), connected to the previous column by a
  bridge `Link("reshape", ...)`. When the previous layer is an attention layer (its
  `type_name` is in `ATTENTION_TYPES`, i.e. `MultiheadAttention`), the subtitle is
  "attention output" and the bridge is `Link("attention", ...)`: it stands for the
  attention `softmax(QKᵀ)V` from the query, key and value projections to the input of
  `out_proj`, which has no weights of its own.
* The layer's output column has the upper-cased layer name as title, `LayerInfo.summary`
  as subtitle, and the summaries of its operations as badges. It takes `bias` from the
  weights, `layer` is the layer's name, and `recurrence` is the layer's
  `LayerWeights.recurrence` ([Weight Data](Tracking-Weight-Data)). The last layer's
  column also gets `output_labels` and kind `output`.
* Captions count the shown and total features of an input column, or neurons of an
  output column, with the noun pluralised: `1 feature`, `16 neurons`,
  `20 of 128 features`, and `? neurons` when the size is unknown.
* A `Link(layer.name, ...)` carries the layer's `LayerWeights`, or `None`.
* Finally each recurrence gets its `loop_source`: the index of the column whose `layer` is
  `recurrence.source`, i.e. the column itself or the column of its `.proj` layer. A
  recurrence whose source column is missing, or drawn with other neurons than
  `recurrence.source_index`, is dropped (`recurrence = None`). Recurrences are not
  links, so the signals stay on the forward connections.

```python
class Signals:
    def __init__(self, seed: int = 7) -> None
    def __len__(self) -> int
    def configure(self, links: list[Link], count: int) -> None
    def advance(self, dt: float, energy: list[np.ndarray], links: list[Link]) -> None
```

`Signals` holds the particles on the connections in numpy arrays: `link`, `source`,
`target`, `progress` (0-1), `speed` and `positive`.

* `configure(links, count)` takes `abs(W)` of every non-bridge link with weights. A link
  is picked with a probability proportional to the square root of its matrix size, and a
  connection inside it proportionally to `w²`. When `count` or a matrix shape changed,
  all `count` particles respawn at random progress. Otherwise only their signs are
  refreshed, and the particles keep moving.
* `advance(dt, energy, links)` moves each particle by `dt · 0.9 · speed`, where
  `speed = 0.55 + 0.95 · abs(w) / max(abs(W))`. When a particle arrives, the target
  neuron's energy rises by 0.55 (capped at 1). If the following link has weights, the
  particle continues into it with probability 0.75, along an outgoing connection of that
  neuron picked proportionally to `w²`. Otherwise it respawns.

```python
class NetworkView(HudPanel):
    def __init__(
        self, session: SessionState, clock: AnimationClock, parent: QWidget | None = None
    ) -> None
```

```python
    @property
    def selected_neuron(self) -> tuple[str, int] | None
    def select_neuron(self, title: str | None, index: int | None = None) -> None
    def columns(self) -> list[Column]

    @staticmethod
    def neuron_lines(column: Column, k: int) -> list[str]
```

The view draws `session.architecture` with `session.weights`. Its title is "Neural
architecture", and its minimum size is 460 × 320. It has no mouse handling: the `neuron`
command highlights a neuron and shows what the hover tooltip of the first version showed,
and the [Pointer Guard](GUI-Pointer-Guard) keeps pointer input away. The attribute
`last_painted_epoch: int | None` holds `session.weights.epoch` after each paint, or
`None` without weights.

* `select_neuron(title, index)` highlights neuron `index` (a neuron index as drawn, see
  `Column.indices`) of the column titled `title`, and schedules a repaint;
  `select_neuron(None)` removes the highlight. `selected_neuron` is the selection as
  `(column title, neuron index)`, or `None`. The `neuron` command
  ([View Actions](GUI-View-Actions)) uses them.
* `columns()` returns the columns as drawn now (it brings the cache up to date first).
* `neuron_lines(column, k)` describes the `k`-th drawn neuron of `column`:
  `<COLUMN> · neuron <index>`, then its label (when the column has labels), then
  `bias <+0.00000>` (when the column has biases). The highlight box and the `neuron`
  command show these lines.
* `last_selection_readout: list[str] | None` holds the lines of the highlight box at the
  latest paint, or `None` when no highlight was drawn.

## How it works

**Connections**, per link:

| Step | Rule |
|---|---|
| strength | `s = abs(w) / max(abs(W))` within the drawn matrix |
| bucket | `level = min(int(s ** 0.75 * 12), 11)`; the sign (`w >= 0`); `hot = min(int(c ** 0.5 * 3), 2)` with `c = abs(dw) / max(abs(dW))` from `LayerWeights.delta` |
| colour | cyan `accent` (positive) or `indigo` (negative), mixed toward `ice` by `0.28 · hot` (white-hot) |
| opacity | alpha `16 + 205 · share + 18 · hot`, where `share = (level + 0.5) / 12` |
| thickness | always 1 px lines. `share >= 0.5` adds two lines offset by ±0.7 px at half opacity, and `share >= 0.8` two more at ±1.7 px (18 %) |
| order | buckets are sorted ascending, so weak lines come first and strong ones end on top. Each bucket and pass is one `drawLines` call with one pen |

A layer without weights yet gets faint uniform lines between all its neurons. A bridge is
drawn as a translucent, dashed quadrilateral labelled with its upper-cased name:
`RESHAPE`, or `ATTENTION`.

**Recurrent connections**, per column with a `recurrence`, after the connections and
before the neurons. They use the buckets, colours and passes of the table above, from
`Recurrence.matrix` and `Recurrence.delta`, at 80 % of the opacity, so the forward
connections stay readable. Every curve is drawn as 12 straight 1 px segments of a
cubic Bézier curve, the fast path:

* The recurrence comes from the column itself: an arc on the right of the column, from
  source neuron `i` back to neuron `j`. It bulges `2.6·r + 0.35·|y_j − y_i|` px, at most
  up to 45 % of the way to the next column (to the panel's edge for the last column). A
  neuron that feeds itself gets a small loop.
* The recurrence comes from the projection's column (an LSTM with `proj_size`): a curve
  from projection neuron `i` back to neuron `j` of the layer's column, sagging below the
  straight connections between the two columns by 28 % of their distance.

**Neurons and labels.** The ring is `accent`. The core is mixed from `ice` toward cyan or
indigo by `bias / max(abs(bias))` of its column. In a column without bias, or with only
zero biases, input neurons use `text` and the others use `ice`. Each column shows its
title, subtitle and operation badges (upper-cased, separated by ` · `) at the top, and
its caption at the bottom, all elided to the column spacing. Input and output labels
appear beside the first and last column when the labels cover every feature.

**Layout** (`_place`). Columns are spaced evenly, with a wider margin when the first or
last column has labels. The neurons of a column are centred vertically, at most 48 px
apart. The neuron radius follows the row and column spacing, clamped to 2.5-9 px.

**Caching.** Every paint starts with `_ensure_static()`, keyed by
`(architecture_version, weights_version, width, height, dpr)`. On a change it rebuilds
the columns, links, layout, signals (`min(160, 36 + connections // 5)` particles) and the
pixmap. If only the weights changed, every neuron's energy rises by 0.35, a ripple at
each update. The panel code is set here too: `NET-01` without an architecture, otherwise
`N LAYERS · X PARAMS`. A resize clears the key.

**Overlay.** Every frame draws one cached glow sprite per neuron, with opacity
`0.12 + 0.1 · sin(...) + 0.75 · energy`. The signal particles are small sprites, `ice` on
positive weights and `indigo` on negative ones. While `RUNNING`, a faint cyan band sweeps
across every 2.8 s. Without columns, the panel shows a pulsing `AWAITING NETWORK SIGNAL`,
or `NO CONNECTION LAYERS TO DRAW` when the architecture has no layers. Each tick advances
the signals by `dt · STATE_SPEED[state]` and decays the energy by `exp(-3.2 · dt)`.

**Highlight** (`_paint_selection`), drawn last while a neuron is selected and drawn: an
`ice` ring of the neuron's radius + 4 px and a pulsing `accent` ring around it, plus a box
with `neuron_lines(...)` framed in `accent_soft`, to the right of the neuron (to its left
near the right edge) and kept inside the panel. The selection is stored by column title
and neuron index and looked up at every paint, so it stays on the same neuron when the
weights are rebuilt each epoch. A selection that is no longer drawn (a changed
architecture) shows nothing.

## Used by / uses

* Used by the [Main Window](GUI-Main-Window) as `window.network`. It is repainted by
  `render_now()` during every synchronous hand-off.
* Uses [Session State](Sync-Session-State), [Architecture Model](Tracking-Architecture-Model),
  [Weight Data](Tracking-Weight-Data) (`LayerWeights`, `display_indices`,
  `DEFAULT_NEURON_LIMIT`), [HUD Panel](GUI-HUD-Panel), [Theme](GUI-Theme) (`glow_sprite`,
  `mix`) and [Animation](GUI-Animation).

## Example

```python
from torch import nn

from hollywood_training.gui.widgets import build_columns
from hollywood_training.tracking import NetworkTracker

model = nn.Sequential(nn.Linear(2, 30), nn.ReLU(), nn.Linear(30, 3))
tracker = NetworkTracker(model, max_neurons=8)
columns, links = build_columns(tracker.architecture.describe(), tracker.snapshot(0), 8)

print([(c.title, c.caption, c.badges) for c in columns])
# [('INPUT', '2 features', ()), ('0', '8 of 30 neurons', ('ReLU',)), ('2', '3 neurons', ())]
print([link.weights.matrix.shape for link in links])   # [(8, 2), (3, 8)]
```

A model with a reshape, `Conv2d(1, 4, 3) → ReLU → Flatten → Linear(64, 3)` with
`limit=8`, gives the column sizes `[1, 4, 64, 3]`, the links `[layer, bridge, layer]`,
and the caption `8 of 64 features` on the reshaped input (from the tests).

## Notes

* **Performance.** All connections are 1 px lines, which Qt's raster engine draws on a
  fast path. Wider antialiased pens would go through the general stroker. The numbers
  are on the [Performance](Performance) page.
* **Neurons per column** are limited by `SessionInfo.neuron_limit`, which is
  `run(max_neurons=...)` with a default of 20. Statistics still cover the full tensors.

"""``NetworkView``: the tracked network - neurons, their connections and the real weights.

* Each **column** of neurons is a feature vector flowing between connection layers: the
  input, then the output of every connection layer (wider layers are sampled using the
  snapshot's own indices, so adjacent layers always line up).
* Each **connection** is drawn from the layer's real connection matrix: cyan for positive
  weights, indigo for negative ones, width and opacity growing with magnitude. Lines that
  changed most in the last epoch glow white-hot.
* Neuron cores are tinted by their **bias**. Activation / normalisation operations are
  listed under each layer's name.
* Ambient animation: glowing neurons and **signals** travelling along connections, more
  often along stronger weights, flashing the neurons they reach.
* The ``neuron`` command **highlights** one drawn neuron: a pulsing ring and a box with its
  index, label and bias (:meth:`NetworkView.select_neuron`). The view has no mouse
  handling (SPEC R2); the highlight replaces the neuron tooltip of the first version.

The static part (frame, connections, neurons, labels) is rendered into a cached pixmap,
rebuilt only when the weights or the layout change - i.e. during the synchronous epoch
hand-off. Each animation frame only blits that pixmap and draws the glow overlay.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from PySide6.QtCore import QLineF, QPointF, QRectF, Qt
from PySide6.QtGui import (
    QColor,
    QFontMetrics,
    QLinearGradient,
    QPainter,
    QPen,
    QPixmap,
    QPolygonF,
)
from PySide6.QtWidgets import QWidget

from ... import formatting as fmt
from ...sync.events import TrainingState
from ...sync.session import SessionState
from ...tracking.architecture import ArchitectureInfo
from ...tracking.weights import DEFAULT_NEURON_LIMIT, LayerWeights, WeightSnapshot, display_indices
from ..animation import AnimationClock, pulse
from ..theme import PALETTE, color, glow_sprite, mix, mono_font
from .hud import HudPanel

LEVELS = 12  # strength buckets used to batch connection lines by pen
HOT_LEVELS = 3  # how strongly a connection changed in the last epoch
STATE_SPEED = {
    TrainingState.RUNNING: 1.0,
    TrainingState.PAUSED: 0.12,
    TrainingState.FINISHED: 0.35,
}


@dataclass(slots=True)
class Column:
    """One drawn column of neurons."""

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


@dataclass(slots=True)
class Link:
    """Connections between two columns: a layer's weights, or a reshape bridge."""

    name: str
    source: int
    target: int
    weights: LayerWeights | None
    bridge: bool = False


def _amount(indices: np.ndarray, size: int | None, noun: str) -> str:
    """Caption such as ``16 neurons``, ``20 of 128 features`` or ``1 feature``."""
    if size is None:
        return f"? {noun}s"
    shown = len(indices)
    plural = noun if size == 1 else f"{noun}s"
    return f"{shown} of {size} {plural}" if shown < size else f"{size} {plural}"


def build_columns(
    architecture: ArchitectureInfo, weights: WeightSnapshot | None, limit: int
) -> tuple[list[Column], list[Link]]:
    """Turn an architecture (+ current weights) into drawable columns and links."""
    columns: list[Column] = []
    links: list[Link] = []
    count = len(architecture.layers)
    for position, layer in enumerate(architecture.layers):
        layer_weights = weights.layer(layer.name) if weights is not None else None
        if layer_weights is not None:
            in_index, out_index = layer_weights.in_index, layer_weights.out_index
        else:
            in_index = display_indices(layer.in_features or 1, limit)
            out_index = display_indices(layer.out_features or 1, limit)

        previous = columns[-1] if columns else None
        if (
            previous is None
            or previous.size != layer.in_features
            or not np.array_equal(previous.indices, in_index)
        ):
            first = previous is None
            columns.append(
                Column(
                    indices=in_index,
                    size=layer.in_features,
                    title="INPUT" if first else f"{layer.name.upper()} IN",
                    subtitle="network input" if first else "reshaped input",
                    caption=_amount(in_index, layer.in_features, "feature"),
                    badges=tuple(op.summary for op in architecture.input_operations)
                    if first
                    else (),
                    labels=architecture.input_labels if first else None,
                    kind="input" if first else "hidden",
                )
            )
            if not first:
                links.append(Link("reshape", len(columns) - 2, len(columns) - 1, None, bridge=True))

        last = position == count - 1
        columns.append(
            Column(
                indices=out_index,
                size=layer.out_features,
                title=layer.name.upper(),
                subtitle=layer.summary,
                caption=_amount(out_index, layer.out_features, "neuron"),
                badges=tuple(op.summary for op in layer.operations),
                labels=architecture.output_labels if last else None,
                bias=layer_weights.bias if layer_weights is not None else None,
                kind="output" if last else "hidden",
            )
        )
        links.append(Link(layer.name, len(columns) - 2, len(columns) - 1, layer_weights))
    return columns, links


def _segments(x0: float, y0: np.ndarray, x1: float, y1: np.ndarray, offset: float) -> list[QLineF]:
    """Lines from ``(x0, y0[k])`` to ``(x1, y1[k])``, shifted ``offset`` px along their normal."""
    if offset == 0.0:
        return [QLineF(x0, a, x1, b) for a, b in zip(y0.tolist(), y1.tolist(), strict=True)]
    dy = y1 - y0
    length = np.hypot(x1 - x0, dy)
    nx = (-dy / length * offset).tolist()
    ny = ((x1 - x0) / length * offset).tolist()
    return [
        QLineF(x0 + ox, a + oy, x1 + ox, b + oy)
        for ox, oy, a, b in zip(nx, ny, y0.tolist(), y1.tolist(), strict=True)
    ]


class Signals:
    """Particles travelling along connections, preferring strong weights."""

    def __init__(self, seed: int = 7) -> None:
        self._rng = np.random.default_rng(seed)
        self._matrices: list[np.ndarray | None] = []
        self._signs: list[np.ndarray | None] = []
        self._choices: list[np.ndarray | None] = []
        self._next: list[int | None] = []
        self._drawable: list[int] = []
        self._link_probability = np.zeros(0)
        self.link = np.zeros(0, dtype=np.int64)
        self.source = np.zeros(0, dtype=np.int64)
        self.target = np.zeros(0, dtype=np.int64)
        self.progress = np.zeros(0)
        self.speed = np.zeros(0)
        self.positive = np.zeros(0, dtype=bool)

    def __len__(self) -> int:
        return int(self.link.size)

    def configure(self, links: list[Link], count: int) -> None:
        shapes_before = [None if m is None else m.shape for m in self._matrices]
        self._matrices = [
            None if link.bridge or link.weights is None or link.weights.matrix.size == 0
            else np.abs(link.weights.matrix)
            for link in links
        ]  # fmt: skip
        self._signs = [
            None if link.bridge or link.weights is None else link.weights.matrix >= 0
            for link in links
        ]
        self._choices = []
        for magnitude in self._matrices:
            if magnitude is None:
                self._choices.append(None)
                continue
            weight = magnitude.ravel() ** 2
            total = weight.sum()
            self._choices.append(weight / total if total > 0 else None)
        self._next = []
        for link in links:
            follower = next(
                (k for k, other in enumerate(links) if other.source == link.target), None
            )
            self._next.append(
                follower if follower is not None and self._matrices[follower] is not None else None
            )
        self._drawable = [k for k, m in enumerate(self._matrices) if m is not None]
        if not self._drawable:
            self._resize(0)
            return
        sizes = np.array([self._matrices[k].size for k in self._drawable], dtype=float)
        self._link_probability = np.sqrt(sizes) / np.sqrt(sizes).sum()
        shapes_after = [None if m is None else m.shape for m in self._matrices]
        if len(self) != count or shapes_before != shapes_after:
            self._resize(count)
            self._spawn(np.arange(count), stagger=True)
        else:
            self._refresh_signs()

    def _resize(self, count: int) -> None:
        self.link = np.zeros(count, dtype=np.int64)
        self.source = np.zeros(count, dtype=np.int64)
        self.target = np.zeros(count, dtype=np.int64)
        self.progress = np.zeros(count)
        self.speed = np.zeros(count)
        self.positive = np.zeros(count, dtype=bool)

    def _strength(self, k: int, j: int, i: int) -> float:
        magnitude = self._matrices[k]
        peak = magnitude.max()
        return float(magnitude[j, i] / peak) if peak > 0 else 0.0

    def _assign(self, p: int, k: int, j: int, i: int, start: float) -> None:
        self.link[p], self.target[p], self.source[p] = k, j, i
        self.progress[p] = start
        self.speed[p] = 0.55 + 0.95 * self._strength(k, j, i)
        self.positive[p] = bool(self._signs[k][j, i])

    def _spawn(self, which: np.ndarray, *, stagger: bool = False) -> None:
        for p in which:
            k = int(self._rng.choice(self._drawable, p=self._link_probability))
            matrix = self._matrices[k]
            choices = self._choices[k]
            flat = (
                int(self._rng.choice(matrix.size, p=choices))
                if choices is not None
                else int(self._rng.integers(matrix.size))
            )
            j, i = divmod(flat, matrix.shape[1])
            self._assign(int(p), k, j, i, float(self._rng.random()) if stagger else 0.0)

    def _refresh_signs(self) -> None:
        for p in range(len(self)):
            k = int(self.link[p])
            if self._signs[k] is not None:
                self.positive[p] = bool(self._signs[k][self.target[p], self.source[p]])

    def advance(self, dt: float, energy: list[np.ndarray], links: list[Link]) -> None:
        if not len(self):
            return
        self.progress += dt * 0.9 * self.speed
        for p in np.nonzero(self.progress >= 1.0)[0]:
            k = int(self.link[p])
            j = int(self.target[p])
            column_energy = energy[links[k].target]
            if j < column_energy.size:
                column_energy[j] = min(1.0, column_energy[j] + 0.55)
            follower = self._next[k]
            if follower is not None and self._rng.random() < 0.75:
                outgoing = self._matrices[follower][:, j] ** 2
                total = outgoing.sum()
                if total > 0:
                    row = int(self._rng.choice(outgoing.size, p=outgoing / total))
                    self._assign(int(p), follower, row, j, 0.0)
                    continue
            self._spawn(np.array([p]))


class NetworkView(HudPanel):
    """Draws ``session.architecture`` with ``session.weights``."""

    def __init__(
        self, session: SessionState, clock: AnimationClock, parent: QWidget | None = None
    ) -> None:
        super().__init__("Neural architecture", "NET-01", parent)
        self._session = session
        self._columns: list[Column] = []
        self._links: list[Link] = []
        self._energy: list[np.ndarray] = []
        self._signals = Signals()
        self._static: QPixmap | None = None
        self._static_key: tuple | None = None
        self._layout_key: tuple | None = None
        self._weights_version = -1
        self._radius = 6.0
        self._time = 0.0
        self._selected: tuple[str, int] | None = None  # (column title, neuron index)
        self.last_painted_epoch: int | None = None
        self.last_selection_readout: list[str] | None = None  # the box at the latest paint
        self.setMinimumSize(460, 320)
        clock.tick.connect(self._advance)

    # --------------------------------------------------------------- selection
    @property
    def selected_neuron(self) -> tuple[str, int] | None:
        """The highlighted neuron as ``(column title, neuron index)``, or None."""
        return self._selected

    def select_neuron(self, title: str | None, index: int | None = None) -> None:
        """Highlight neuron ``index`` of the column titled ``title``; None removes it."""
        self._selected = None if title is None or index is None else (title, int(index))
        self.update()

    def columns(self) -> list[Column]:
        """The columns as drawn now, for the current architecture and weights."""
        self._ensure_static()
        return list(self._columns)

    @staticmethod
    def neuron_lines(column: Column, k: int) -> list[str]:
        """What is shown about the ``k``-th drawn neuron of ``column``."""
        index = int(column.indices[k])
        lines = [f"{column.title} · neuron {index}"]
        if column.labels and index < len(column.labels):
            lines.append(column.labels[index])
        if column.bias is not None:
            lines.append(f"bias {float(column.bias[k]):+.5f}")
        return lines

    def _find(self, title: str, index: int) -> tuple[Column, int] | None:
        for column in self._columns:
            if column.title == title:
                hits = np.flatnonzero(column.indices == index)
                return (column, int(hits[0])) if hits.size else None
        return None

    # ----------------------------------------------------------------- caching
    def _ensure_static(self) -> None:
        session = self._session
        dpr = round(self.devicePixelRatioF(), 2)
        key = (
            session.architecture_version,
            session.weights_version,
            self.width(),
            self.height(),
            dpr,
        )
        if key == self._static_key:
            return
        self._static_key = key
        architecture = session.architecture
        if architecture is None or not architecture.layers:
            self._columns, self._links, self._energy = [], [], []
            self._signals.configure([], 0)
        else:
            limit = session.info.neuron_limit if session.info else DEFAULT_NEURON_LIMIT
            self._columns, self._links = build_columns(architecture, session.weights, limit)
            self._place(self.content_rect())
            sizes = tuple(len(column.indices) for column in self._columns)
            layout_key = (session.architecture_version, sizes)
            if layout_key != self._layout_key:
                self._layout_key = layout_key
                self._energy = [np.zeros(size) for size in sizes]
            elif session.weights_version != self._weights_version:
                for energy in self._energy:  # the update moment: a ripple through the net
                    np.add(energy, 0.35, out=energy)
            connections = sum(
                0 if link.weights is None else link.weights.matrix.size for link in self._links
            )
            self._signals.configure(self._links, int(min(160, 36 + connections // 5)))
        self._weights_version = session.weights_version
        code = "NET-01"
        if architecture is not None:
            parameters = fmt.count(architecture.total_parameters)
            code = f"{len(architecture.layers)} LAYERS · {parameters} PARAMS"
        self._code = code.upper()  # set directly: the cache below is drawn right away
        self._static = self._render_static(dpr)

    def _place(self, rect: QRectF) -> None:
        columns = self._columns
        n = len(columns)
        left = rect.left() + (70 if columns[0].labels else 34)
        right = rect.right() - (84 if columns[-1].labels else 34)
        top, bottom = rect.top() + 58, rect.bottom() - 24
        gap = (right - left) / max(1, n - 1)
        most = max(len(column.indices) for column in columns)
        row = (bottom - top) / max(1, most - 1) if most > 1 else 48.0
        for k, column in enumerate(columns):
            column.x = (left + right) / 2 if n == 1 else left + gap * k
            count = len(column.indices)
            spacing = min((bottom - top) / max(1, count - 1), 48.0) if count > 1 else 0.0
            column.ys = (top + bottom) / 2 - spacing * (count - 1) / 2 + spacing * np.arange(count)
        self._radius = float(np.clip(min(row * 0.3, gap * 0.06 if n > 1 else 9.0), 2.5, 9.0))

    # ---------------------------------------------------------------- painting
    def _render_static(self, dpr: float) -> QPixmap:
        pixmap = QPixmap(max(1, int(self.width() * dpr)), max(1, int(self.height() * dpr)))
        pixmap.setDevicePixelRatio(dpr)
        pixmap.fill(Qt.GlobalColor.transparent)
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.paint_frame(painter)
        if self._columns:
            self._paint_guides(painter)
            for link in self._links:
                if link.bridge:
                    self._paint_bridge(painter, link)
                else:
                    self._paint_connections(painter, link)
            self._paint_neurons(painter)
            self._paint_labels(painter)
        painter.end()
        return pixmap

    def _paint_guides(self, painter: QPainter) -> None:
        rect = self.content_rect()
        pen = QPen(color(PALETTE.grid, 90), 1.0, Qt.PenStyle.DotLine)
        painter.setPen(pen)
        for column in self._columns:
            painter.drawLine(
                QPointF(column.x, rect.top() + 50), QPointF(column.x, rect.bottom() - 18)
            )

    def _paint_connections(self, painter: QPainter, link: Link) -> None:
        # Every stroke here is a 1 px line: Qt's raster engine draws those on a fast path,
        # while wider antialiased pens go through the general stroker (~10x slower).
        # Thickness and glow of strong weights come from parallel offset lines instead.
        source, target = self._columns[link.source], self._columns[link.target]
        if link.weights is None:  # not initialised yet: faint uniform wiring
            painter.setPen(QPen(color(PALETTE.text_dim, 40), 1.0))
            painter.drawLines(
                [QLineF(source.x, ys, target.x, yt) for ys in source.ys for yt in target.ys]
            )
            return
        matrix = link.weights.matrix
        peak = float(np.abs(matrix).max()) if matrix.size else 0.0
        if peak <= 0:
            return
        strength = np.abs(matrix) / peak
        level = np.minimum((strength**0.75 * LEVELS).astype(int), LEVELS - 1)
        hot = np.zeros(matrix.shape, dtype=int)
        delta = link.weights.delta
        if delta is not None and delta.size and np.abs(delta).max() > 0:
            change = np.abs(delta) / np.abs(delta).max()
            hot = np.minimum((change**0.5 * HOT_LEVELS).astype(int), HOT_LEVELS - 1)
        positive = (matrix >= 0).astype(int)
        code = ((level * 2 + positive) * HOT_LEVELS + hot).ravel()
        order = np.argsort(code, kind="stable")  # weak buckets first, strong on top
        bounds = np.flatnonzero(np.diff(code[order])) + 1
        ice = color(PALETTE.ice)
        for group in np.split(order, bounds):
            bucket = int(code[group[0]])
            lev, sign, heat = (
                bucket // (2 * HOT_LEVELS),
                (bucket // HOT_LEVELS) % 2,
                bucket % HOT_LEVELS,
            )
            share = (lev + 0.5) / LEVELS
            rows, cols = np.unravel_index(group, matrix.shape)
            y0, y1 = source.ys[cols], target.ys[rows]
            tint = mix(color(PALETTE.accent if sign else PALETTE.indigo), ice, heat * 0.28)
            alpha = 16 + 205 * share + 18 * heat
            passes = [(0.0, 1.0)]
            if share >= 0.5:
                passes += [(-0.7, 0.5), (0.7, 0.5)]
            if share >= 0.8:
                passes += [(-1.7, 0.18), (1.7, 0.18)]
            for offset, opacity in passes:
                ink = QColor(tint)
                ink.setAlpha(min(255, int(alpha * opacity)))
                painter.setPen(QPen(ink, 1.0))
                painter.drawLines(_segments(source.x, y0, target.x, y1, offset))

    def _paint_bridge(self, painter: QPainter, link: Link) -> None:
        a, b = self._columns[link.source], self._columns[link.target]
        r = self._radius
        polygon = QPolygonF(
            [
                QPointF(a.x, float(a.ys.min()) - r),
                QPointF(b.x, float(b.ys.min()) - r),
                QPointF(b.x, float(b.ys.max()) + r),
                QPointF(a.x, float(a.ys.max()) + r),
            ]
        )
        fill = QLinearGradient(QPointF(a.x, 0), QPointF(b.x, 0))
        fill.setColorAt(0.0, color(PALETTE.electric, 55))
        fill.setColorAt(1.0, color(PALETTE.accent, 25))
        painter.setBrush(fill)
        painter.setPen(QPen(color(PALETTE.accent, 110), 1.0, Qt.PenStyle.DashLine))
        painter.drawPolygon(polygon)
        painter.setFont(mono_font(7.5, bold=True, spacing=2.0))
        painter.setPen(color(PALETTE.accent_soft))
        center = polygon.boundingRect().center()
        painter.drawText(
            QRectF(center.x() - 60, center.y() - 8, 120, 16),
            Qt.AlignmentFlag.AlignCenter,
            "RESHAPE",
        )

    def _paint_neurons(self, painter: QPainter) -> None:
        r = self._radius
        ring = QPen(color(PALETTE.accent, 215), 1.2)
        ice = color(PALETTE.ice)
        for column in self._columns:
            bias = column.bias
            scale = float(np.abs(bias).max()) if bias is not None and bias.size else 0.0
            for k, y in enumerate(column.ys):
                if bias is not None and scale > 0:
                    value = float(bias[k]) / scale
                    target = color(PALETTE.accent if value >= 0 else PALETTE.indigo)
                    fill = mix(ice, target, min(1.0, abs(value)))
                elif column.kind == "input":
                    fill = color(PALETTE.text, 235)
                else:
                    fill = color(PALETTE.ice, 235)
                painter.setPen(ring)
                painter.setBrush(fill)
                painter.drawEllipse(QPointF(column.x, float(y)), r, r)

    def _paint_labels(self, painter: QPainter) -> None:
        rect = self.content_rect()
        gap = (self._columns[-1].x - self._columns[0].x) / max(1, len(self._columns) - 1)
        width = max(60.0, gap - 12) if len(self._columns) > 1 else rect.width() - 20
        title_font = mono_font(8.5, bold=True, spacing=1.5)
        small = mono_font(7.5)
        badge_font = mono_font(7.0, bold=True, spacing=1.0)

        def centered(x: float, font, text: str, y: float, ink) -> None:
            metrics = QFontMetrics(font)
            text = metrics.elidedText(text, Qt.TextElideMode.ElideRight, int(width))
            painter.setFont(font)
            painter.setPen(ink)
            box = QRectF(x - width / 2, y, width, metrics.height() + 2)
            painter.drawText(box, Qt.AlignmentFlag.AlignHCenter, text)

        for column in self._columns:
            x = column.x
            centered(x, title_font, column.title, rect.top() + 4, color(PALETTE.accent))
            centered(x, small, column.subtitle, rect.top() + 20, color(PALETTE.text_dim))
            if column.badges:
                badge = " · ".join(b.upper() for b in column.badges)
                centered(x, badge_font, badge, rect.top() + 34, color(PALETTE.accent_soft))
            centered(x, small, column.caption, rect.bottom() - 16, color(PALETTE.text_dim))

        label_font = mono_font(8.0)
        metrics = QFontMetrics(label_font)
        painter.setFont(label_font)
        painter.setPen(color(PALETTE.ice, 220))
        for column, side in ((self._columns[0], -1), (self._columns[-1], 1)):
            labels = column.labels
            if not labels or column.size is None or len(labels) < column.size:
                continue
            for k, y in enumerate(column.ys):
                text = metrics.elidedText(
                    labels[int(column.indices[k])], Qt.TextElideMode.ElideRight, 72
                )
                if side < 0:
                    box = QRectF(column.x - self._radius - 80, y - 8, 72, 16)
                    painter.drawText(
                        box, Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter, text
                    )
                else:
                    box = QRectF(column.x + self._radius + 8, y - 8, 76, 16)
                    painter.drawText(
                        box, Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, text
                    )

    def paintEvent(self, _event) -> None:
        self._ensure_static()
        painter = QPainter(self)
        if self._static is not None:
            painter.drawPixmap(0, 0, self._static)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        readout = None
        if self._columns:
            self._paint_dynamic(painter)
            readout = self._paint_selection(painter)
        else:
            self._paint_waiting(painter)
        painter.end()
        weights = self._session.weights
        self.last_painted_epoch = weights.epoch if weights is not None else None
        self.last_selection_readout = readout

    def _paint_selection(self, painter: QPainter) -> list[str] | None:
        """The highlighted neuron: a pulsing ring and a box with what is known about it."""
        found = None if self._selected is None else self._find(*self._selected)
        if found is None:
            return None
        column, k = found
        center = QPointF(column.x, float(column.ys[k]))
        radius = self._radius
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.setPen(QPen(color(PALETTE.ice), 1.6))
        painter.drawEllipse(center, radius + 4, radius + 4)
        glow = pulse(self._time, 1.2)
        painter.setPen(QPen(color(PALETTE.accent, int(70 + 130 * (1 - glow))), 1.2))
        painter.drawEllipse(center, radius + 7 + 5 * glow, radius + 7 + 5 * glow)
        lines = self.neuron_lines(column, k)
        font = mono_font(8.0)
        metrics = QFontMetrics(font)
        width = max(metrics.horizontalAdvance(line) for line in lines) + 14
        height = metrics.height() * len(lines) + 8
        rect = self.content_rect()
        left = center.x() + radius + 14
        if left + width > rect.right():
            left = center.x() - radius - 14 - width
        top = min(max(center.y() - height / 2, rect.top()), rect.bottom() - height)
        box = QRectF(left, top, width, height)
        painter.setPen(QPen(color(PALETTE.accent_soft), 1.0))
        painter.setBrush(color(PALETTE.void, 225))
        painter.drawRect(box)
        painter.setFont(font)
        painter.setPen(color(PALETTE.ice))
        for row, line in enumerate(lines):
            baseline = box.top() + 4 + metrics.ascent() + row * metrics.height()
            painter.drawText(QPointF(box.left() + 7, baseline), line)
        return lines

    def _paint_dynamic(self, painter: QPainter) -> None:
        dpr = self.devicePixelRatioF()
        t = self._time
        halo_radius = max(6, int(self._radius * 2.3))
        halo = glow_sprite(halo_radius, color(PALETTE.accent, 230), dpr=dpr)
        for ci, column in enumerate(self._columns):
            energy = self._energy[ci] if ci < len(self._energy) else np.zeros(len(column.ys))
            for k, y in enumerate(column.ys):
                level = 0.12 + 0.1 * math.sin(t * 2.3 + k * 0.9 + ci * 1.7)
                if k < energy.size:
                    level += 0.75 * float(energy[k])
                painter.setOpacity(max(0.0, min(1.0, level)))
                painter.drawPixmap(QPointF(column.x - halo_radius, float(y) - halo_radius), halo)
        painter.setOpacity(1.0)

        signals = self._signals
        if len(signals):
            bright = glow_sprite(5, color(PALETTE.ice, 255), core=0.35, dpr=dpr)
            dark = glow_sprite(5, color(PALETTE.indigo, 255), core=0.35, dpr=dpr)
            for p in range(len(signals)):
                link = self._links[int(signals.link[p])]
                a, b = self._columns[link.source], self._columns[link.target]
                i, j = int(signals.source[p]), int(signals.target[p])
                if i >= a.ys.size or j >= b.ys.size:
                    continue
                s = float(signals.progress[p])
                x = a.x + (b.x - a.x) * s
                y = float(a.ys[i]) + (float(b.ys[j]) - float(a.ys[i])) * s
                painter.drawPixmap(QPointF(x - 5, y - 5), bright if signals.positive[p] else dark)

        if self._session.state is TrainingState.RUNNING:  # scanning band
            rect = self.content_rect()
            x = rect.left() + (t % 2.8) / 2.8 * rect.width()
            band = QLinearGradient(QPointF(x - 50, 0), QPointF(x + 50, 0))
            band.setColorAt(0.0, color(PALETTE.accent, 0))
            band.setColorAt(0.5, color(PALETTE.accent, 20))
            band.setColorAt(1.0, color(PALETTE.accent, 0))
            painter.fillRect(QRectF(x - 50, rect.top() + 50, 100, rect.height() - 66), band)

    def _paint_waiting(self, painter: QPainter) -> None:
        rect = self.content_rect()
        architecture = self._session.architecture
        text = "AWAITING NETWORK SIGNAL"
        if architecture is not None and not architecture.layers:
            text = "NO CONNECTION LAYERS TO DRAW"
        painter.setFont(mono_font(11, bold=True, spacing=3.0))
        painter.setPen(color(PALETTE.accent, int(90 + 140 * pulse(self._time, 1.6))))
        painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, text)

    # --------------------------------------------------------------- animation
    def _advance(self, dt: float) -> None:
        self._time += dt
        if self._columns:
            factor = STATE_SPEED.get(self._session.state, 0.5)
            self._signals.advance(dt * factor, self._energy, self._links)
            decay = math.exp(-dt * 3.2)
            for energy in self._energy:
                energy *= decay
        self.update()

    def resizeEvent(self, event) -> None:
        self._static_key = None
        super().resizeEvent(event)

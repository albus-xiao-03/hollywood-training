"""``LossChart``: training and validation loss, extended after every epoch.

The x axis spans all requested epochs, so the curves visibly grow toward the end of the
run. Click the chart to switch between a linear and a logarithmic loss axis; hover it to
read the values of an epoch. Axes, grid and curves are cached and rebuilt when a new
epoch arrives (during the synchronous hand-off); the pulsing markers at the newest points
and the hover read-out are drawn on top every frame.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import (
    QColor,
    QFontMetrics,
    QLinearGradient,
    QPainter,
    QPainterPath,
    QPen,
    QPixmap,
    QPolygonF,
)
from PySide6.QtWidgets import QWidget

from ... import formatting as fmt
from ...sync.session import SessionState
from ..animation import AnimationClock, pulse
from ..theme import PALETTE, color, mono_font
from .hud import HudPanel


def nice_ticks(low: float, high: float, count: int = 5) -> list[float]:
    """Round tick values (1-2-2.5-5 steps) covering ``[low, high]``, about ``count`` intervals."""
    span = high - low
    if not math.isfinite(span) or span <= 0:
        return [low]
    magnitude = 10 ** math.floor(math.log10(span / max(1, count)))
    candidates = [m * magnitude for m in (1, 2, 2.5, 5, 10)]
    step = min(candidates, key=lambda c: abs(span / c - count))
    first = math.ceil(low / step - 1e-9) * step
    ticks = []
    value = first
    while value <= high + step * 1e-6:
        ticks.append(round(value, 12))
        value += step
    return ticks


def decade_ticks(low: float, high: float) -> list[float]:
    """Powers of ten between ``low`` and ``high`` (inclusive)."""
    return [10.0**k for k in range(round(math.log10(low)), round(math.log10(high)) + 1)]


@dataclass(slots=True)
class Axes:
    """Maps epochs / loss values to widget coordinates."""

    plot: QRectF
    x0: float
    x1: float
    y0: float
    y1: float
    log: bool

    def x(self, epoch: float) -> float:
        return self.plot.left() + (epoch - self.x0) / (self.x1 - self.x0) * self.plot.width()

    def y(self, value: float) -> float:
        if self.log:
            low, high = math.log10(self.y0), math.log10(self.y1)
            value = math.log10(max(value, self.y0))
        else:
            low, high = self.y0, self.y1
        return self.plot.bottom() - (value - low) / (high - low) * self.plot.height()

    def visible(self, value: float) -> bool:
        return math.isfinite(value) and (value > 0 or not self.log)


def _curve(axes: Axes, epochs: np.ndarray, values: np.ndarray) -> QPainterPath:
    path = QPainterPath()
    drawing = False
    for epoch, value in zip(epochs, values, strict=True):
        if not axes.visible(float(value)):
            drawing = False
            continue
        point = QPointF(axes.x(float(epoch)), axes.y(float(value)))
        if drawing:
            path.lineTo(point)
        else:
            path.moveTo(point)
            drawing = True
    return path


class LossChart(HudPanel):
    """Draws ``session.history`` as training / validation loss curves."""

    def __init__(
        self, session: SessionState, clock: AnimationClock, parent: QWidget | None = None
    ) -> None:
        super().__init__("Loss", "LSS-02", parent)
        self._session = session
        self._log = False
        self._static: QPixmap | None = None
        self._static_key: tuple | None = None
        self._axes: Axes | None = None
        self._time = 0.0
        self._hover: QPointF | None = None
        self.last_painted_epoch: int | None = None
        self.setMinimumSize(320, 200)
        self.setMouseTracking(True)
        self.setCursor(Qt.CursorShape.CrossCursor)
        self.setToolTip("Click: switch between linear and logarithmic scale")
        clock.tick.connect(self._advance)

    @property
    def log_scale(self) -> bool:
        return self._log

    def set_log_scale(self, enabled: bool) -> None:
        self._log = bool(enabled)
        self._static_key = None
        self.update()

    def has_positive_loss(self) -> bool:
        """Whether a logarithmic axis can be drawn: some loss is finite and positive."""
        _epochs, train, val = self._series()
        values = np.concatenate([train, val])
        return bool(np.any(np.isfinite(values) & (values > 0)))

    # ---------------------------------------------------------------- data
    def _series(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        history = self._session.history
        epochs = np.array([r.epoch for r in history], dtype=float)
        train = np.array([r.train_loss for r in history], dtype=float)
        val = np.array(
            [math.nan if r.val_loss is None else r.val_loss for r in history], dtype=float
        )
        return epochs, train, val

    def _make_axes(self, plot: QRectF, epochs: np.ndarray, values: np.ndarray) -> Axes | None:
        finite = values[np.isfinite(values)]
        if epochs.size == 0 or finite.size == 0:
            return None
        x1 = float(max(self._session.total_epochs, epochs.max(), 2))
        positive = finite[finite > 0]
        if self._log and positive.size:
            y0 = 10 ** math.floor(math.log10(positive.min()))
            y1 = 10 ** math.ceil(math.log10(positive.max()))
            if y1 <= y0:
                y1 = y0 * 10
            return Axes(plot, 1.0, x1, y0, y1, True)
        low = min(0.0, float(finite.min()))
        high = float(finite.max())
        if high <= low:
            high = low + 1.0
        return Axes(plot, 1.0, x1, low, high + (high - low) * 0.08, False)

    # ------------------------------------------------------------ painting
    def _ensure_static(self) -> None:
        dpr = round(self.devicePixelRatioF(), 2)
        session = self._session
        key = (
            session.history_version,
            session.total_epochs,
            self._log,
            self.width(),
            self.height(),
            dpr,
        )
        if key == self._static_key:
            return
        self._static_key = key
        pixmap = QPixmap(max(1, int(self.width() * dpr)), max(1, int(self.height() * dpr)))
        pixmap.setDevicePixelRatio(dpr)
        pixmap.fill(Qt.GlobalColor.transparent)
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.paint_frame(painter)
        self._render_plot(painter)
        painter.end()
        self._static = pixmap

    def _render_plot(self, painter: QPainter) -> None:
        rect = self.content_rect()
        plot = QRectF(rect.left() + 56, rect.top() + 24, rect.width() - 66, rect.height() - 48)
        epochs, train, val = self._series()
        axes = self._make_axes(plot, epochs, np.concatenate([train, val]))
        self._axes = axes
        small = mono_font(7.5)
        painter.setFont(small)
        if axes is None:
            painter.setPen(QPen(color(PALETTE.grid, 120), 1.0))
            painter.drawRect(plot)
            painter.setFont(mono_font(10, bold=True, spacing=3.0))
            painter.setPen(color(PALETTE.accent, 150))
            painter.drawText(plot, Qt.AlignmentFlag.AlignCenter, "AWAITING FIRST EPOCH")
            return

        # grid + tick labels
        y_ticks = decade_ticks(axes.y0, axes.y1) if axes.log else nice_ticks(axes.y0, axes.y1, 5)
        x_ticks = [t for t in nice_ticks(axes.x0, axes.x1, 6) if t >= 1 and float(t).is_integer()]
        metrics = QFontMetrics(small)
        for value in y_ticks:
            y = axes.y(value)
            if y < plot.top() - 1 or y > plot.bottom() + 1:
                continue
            painter.setPen(QPen(color(PALETTE.grid, 110), 1.0))
            painter.drawLine(QPointF(plot.left(), y), QPointF(plot.right(), y))
            painter.setPen(color(PALETTE.text_dim))
            label = f"{value:.0e}" if axes.log else f"{value:.4g}"
            painter.drawText(
                QRectF(rect.left(), y - 7, 50, 14),
                Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
                label,
            )
        for value in x_ticks:
            x = axes.x(value)
            painter.setPen(QPen(color(PALETTE.grid, 70), 1.0, Qt.PenStyle.DotLine))
            painter.drawLine(QPointF(x, plot.top()), QPointF(x, plot.bottom()))
            painter.setPen(color(PALETTE.text_dim))
            text = f"{int(value)}"
            painter.drawText(
                QPointF(x - metrics.horizontalAdvance(text) / 2, plot.bottom() + 14), text
            )
        painter.setPen(QPen(color(PALETTE.panel_edge, 170), 1.0))
        painter.drawLine(plot.bottomLeft(), plot.bottomRight())
        painter.drawLine(plot.bottomLeft(), plot.topLeft())
        painter.setPen(color(PALETTE.text_dim))
        painter.drawText(
            QPointF(plot.right() - metrics.horizontalAdvance("EPOCH"), plot.bottom() + 26), "EPOCH"
        )

        # training loss: area, glow, line
        painter.save()
        painter.setClipRect(plot.adjusted(-2, -2, 2, 2))
        train_path = _curve(axes, epochs, train)
        visible = [k for k in range(epochs.size) if axes.visible(float(train[k]))]
        if visible:
            first, last = visible[0], visible[-1]
            area = QPainterPath(train_path)
            area.lineTo(QPointF(axes.x(float(epochs[last])), plot.bottom()))
            area.lineTo(QPointF(axes.x(float(epochs[first])), plot.bottom()))
            area.closeSubpath()
            fill = QLinearGradient(plot.topLeft(), plot.bottomLeft())
            fill.setColorAt(0.0, color(PALETTE.accent, 70))
            fill.setColorAt(1.0, color(PALETTE.accent, 0))
            painter.fillPath(area, fill)
        self._stroke(painter, train_path, color(PALETTE.accent), dashed=False)
        if np.isfinite(val).any():
            self._stroke(painter, _curve(axes, epochs, val), color(PALETTE.indigo), dashed=True)
        painter.restore()

        # best epoch marker
        best = self._session.best_record()
        if best is not None:
            value = best.monitored_loss
            if axes.visible(value):
                center = QPointF(axes.x(best.epoch), axes.y(value))
                diamond = QPolygonF(
                    [
                        center + QPointF(0, -5),
                        center + QPointF(5, 0),
                        center + QPointF(0, 5),
                        center + QPointF(-5, 0),
                    ]
                )
                painter.setPen(QPen(color(PALETTE.ice), 1.2))
                painter.setBrush(color(PALETTE.success, 200))
                painter.drawPolygon(diamond)

        self._legend(painter, rect, train, val, best)

    @staticmethod
    def _stroke(painter: QPainter, path: QPainterPath, ink: QColor, *, dashed: bool) -> None:
        glow = QColor(ink)
        glow.setAlpha(45)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.setPen(
            QPen(
                glow, 6.0, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin
            )
        )
        painter.drawPath(path)
        pen = QPen(
            ink, 2.0, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin
        )
        if dashed:
            pen.setStyle(Qt.PenStyle.CustomDashLine)
            pen.setDashPattern([4.0, 2.5])
        painter.setPen(pen)
        painter.drawPath(path)

    def _legend(self, painter: QPainter, rect: QRectF, train, val, best) -> None:
        font = mono_font(8.0, bold=True, spacing=1.0)
        metrics = QFontMetrics(font)
        painter.setFont(font)
        x = rect.left() + 56
        y = rect.top() + 10
        entries = [(PALETTE.accent, False, f"TRAIN {fmt.loss(float(train[-1]))}")]
        if np.isfinite(val).any():
            entries.append((PALETTE.indigo, True, f"VAL {fmt.loss(float(val[-1]))}"))
        if best is not None:
            value = best.monitored_loss
            entries.append((PALETTE.success, None, f"BEST {fmt.loss(value)} @ {best.epoch}"))
        for ink, dashed, text in entries:
            if dashed is None:
                painter.setPen(QPen(color(PALETTE.ice), 1.0))
                painter.setBrush(color(ink, 200))
                center = QPointF(x + 6, y)
                painter.drawPolygon(
                    QPolygonF(
                        [
                            center + QPointF(0, -4),
                            center + QPointF(4, 0),
                            center + QPointF(0, 4),
                            center + QPointF(-4, 0),
                        ]
                    )
                )
            else:
                pen = QPen(color(ink), 2.0)
                if dashed:
                    pen.setStyle(Qt.PenStyle.CustomDashLine)
                    pen.setDashPattern([3.0, 2.0])
                painter.setPen(pen)
                painter.drawLine(QPointF(x, y), QPointF(x + 14, y))
            painter.setPen(color(PALETTE.ice))
            painter.drawText(QPointF(x + 19, y + 4), text)
            x += 19 + metrics.horizontalAdvance(text) + 18
        # the axes fall back to linear when no loss is positive: report what is drawn
        scale = "LOG" if self._axes is not None and self._axes.log else "LIN"
        painter.setPen(color(PALETTE.text_dim))
        painter.drawText(QPointF(rect.right() - metrics.horizontalAdvance(scale) - 4, y + 4), scale)

    def paintEvent(self, _event) -> None:
        self._ensure_static()
        painter = QPainter(self)
        if self._static is not None:
            painter.drawPixmap(0, 0, self._static)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        axes = self._axes
        history = self._session.history
        if axes is not None and history:
            last = history[-1]
            glow = pulse(self._time, 1.3)
            for value, ink in ((last.train_loss, PALETTE.accent), (last.val_loss, PALETTE.indigo)):
                if value is None or not axes.visible(value):
                    continue
                center = QPointF(axes.x(last.epoch), axes.y(value))
                painter.setPen(QPen(color(ink, int(60 + 120 * (1 - glow))), 1.5))
                painter.setBrush(Qt.BrushStyle.NoBrush)
                painter.drawEllipse(center, 4 + 6 * glow, 4 + 6 * glow)
                painter.setPen(Qt.PenStyle.NoPen)
                painter.setBrush(color(PALETTE.ice))
                painter.drawEllipse(center, 2.6, 2.6)
            self._paint_hover(painter, axes)
        painter.end()
        self.last_painted_epoch = history[-1].epoch if history else None

    def _paint_hover(self, painter: QPainter, axes: Axes) -> None:
        if self._hover is None or not axes.plot.contains(self._hover):
            return
        history = self._session.history
        epoch = axes.x0 + (self._hover.x() - axes.plot.left()) / axes.plot.width() * (
            axes.x1 - axes.x0
        )
        record = min(history, key=lambda r: abs(r.epoch - epoch))
        x = axes.x(record.epoch)
        painter.setPen(QPen(color(PALETTE.accent_soft, 140), 1.0, Qt.PenStyle.DashLine))
        painter.drawLine(QPointF(x, axes.plot.top()), QPointF(x, axes.plot.bottom()))
        lines = [f"EPOCH {record.epoch}", f"TRAIN {fmt.loss(record.train_loss)}"]
        if record.val_loss is not None:
            lines.append(f"VAL   {fmt.loss(record.val_loss)}")
        font = mono_font(8.0)
        metrics = QFontMetrics(font)
        width = max(metrics.horizontalAdvance(line) for line in lines) + 14
        height = metrics.height() * len(lines) + 8
        left = x + 10 if x + 10 + width < axes.plot.right() else x - 10 - width
        box = QRectF(left, axes.plot.top() + 6, width, height)
        painter.setPen(QPen(color(PALETTE.panel_edge), 1.0))
        painter.setBrush(color(PALETTE.void, 220))
        painter.drawRect(box)
        painter.setFont(font)
        painter.setPen(color(PALETTE.ice))
        for k, line in enumerate(lines):
            painter.drawText(
                QPointF(box.left() + 7, box.top() + 4 + metrics.ascent() + k * metrics.height()),
                line,
            )

    # ------------------------------------------------------------ interaction
    def _advance(self, dt: float) -> None:
        self._time += dt
        if self._session.history:
            self.update()

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self.set_log_scale(not self._log)
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:
        self._hover = event.position()
        super().mouseMoveEvent(event)

    def leaveEvent(self, event) -> None:
        self._hover = None
        super().leaveEvent(event)

    def resizeEvent(self, event) -> None:
        self._static_key = None
        super().resizeEvent(event)

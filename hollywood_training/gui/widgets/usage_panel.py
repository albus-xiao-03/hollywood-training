"""``UsagePanel``: CPU, memory and - when the model trains on a GPU - GPU usage.

Readings (:class:`~hollywood_training.monitor.usage.UsageSample`) either arrive from the
training process (:meth:`UsagePanel.set_sample`, used by the isolated GUI process) or
are pulled twice a second from a sampler function (:meth:`UsagePanel.set_sampler`, used
in-process). Displayed levels ease toward each new reading. The GPU rows only appear when
the session trains on a GPU that can be read (CUDA or Apple MPS).
"""

from __future__ import annotations

from collections import deque
from collections.abc import Callable

from PySide6.QtCore import QPointF, QRectF, Qt, QTimer
from PySide6.QtGui import QFontMetrics, QLinearGradient, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import QWidget

from ... import formatting as fmt
from ...monitor.usage import UsageSample
from ..animation import AnimationClock, approach
from ..theme import PALETTE, color, mono_font
from .hud import HudPanel, paint_bar

SAMPLE_INTERVAL_MS = 500
HISTORY = 120  # one minute of CPU readings


class UsagePanel(HudPanel):
    def __init__(self, clock: AnimationClock, parent: QWidget | None = None) -> None:
        super().__init__("System usage", "SYS-04", parent)
        self._sampler: Callable[[], UsageSample] | None = None
        self._sample: UsageSample | None = None
        self._cpu_history: deque[float] = deque(maxlen=HISTORY)
        self._shown: dict[str, float] = {"cpu": 0.0, "memory": 0.0, "gpu": 0.0, "vram": 0.0}
        self._time = 0.0
        self._timer = QTimer(self)
        self._timer.setInterval(SAMPLE_INTERVAL_MS)
        self._timer.timeout.connect(self.sample_now)
        self.setMinimumSize(320, 140)
        clock.tick.connect(self._advance)

    @property
    def sample(self) -> UsageSample | None:
        return self._sample

    def set_sampler(self, sampler: Callable[[], UsageSample] | None) -> None:
        """Pull readings from ``sampler`` twice a second (None stops pulling)."""
        self._sampler = sampler
        if sampler is None:
            self._timer.stop()
            return
        self.sample_now()
        self._timer.start()

    def set_sample(self, sample: UsageSample) -> None:
        """Show a reading taken elsewhere (e.g. by the training process)."""
        self._sample = sample
        self._cpu_history.append(sample.cpu_percent)

    def stop(self) -> None:
        self._timer.stop()

    def sample_now(self) -> None:
        if self._sampler is None:
            return
        try:
            sample = self._sampler()
        except Exception:  # never let a monitoring hiccup disturb the GUI
            return
        self.set_sample(sample)

    def _advance(self, dt: float) -> None:
        self._time += dt
        sample = self._sample
        if sample is not None:
            targets = {"cpu": sample.cpu_percent / 100.0, "memory": sample.memory_fraction}
            gpu = sample.gpu
            if gpu is not None:
                targets["gpu"] = (gpu.utilization or 0.0) / 100.0
                if gpu.memory_used is not None and gpu.memory_total:
                    targets["vram"] = gpu.memory_used / gpu.memory_total
            for key, target in targets.items():
                self._shown[key] = approach(self._shown[key], target, dt, 0.12)
        self.update()

    # ---------------------------------------------------------------- painting
    def paint_content(self, painter: QPainter, rect: QRectF) -> None:
        sample = self._sample
        if sample is None:
            painter.setFont(mono_font(9, bold=True, spacing=2.0))
            painter.setPen(color(PALETTE.accent, 150))
            painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, "SENSORS OFFLINE")
            return
        rows = [
            (
                "CPU",
                self._shown["cpu"],
                f"{sample.cpu_percent:5.1f}%",
                f"proc {sample.process_cpu_percent:.0f}% · {sample.cpu_count} cores",
            ),
            (
                "MEM",
                self._shown["memory"],
                f"{fmt.data_size(sample.memory_used)} / {fmt.data_size(sample.memory_total)}",
                f"proc {fmt.data_size(sample.process_memory)}",
            ),
        ]
        gpu = sample.gpu
        if gpu is not None:
            if gpu.utilization is not None:
                rows.append(("GPU", self._shown["gpu"], f"{gpu.utilization:5.1f}%", gpu.name))
            if gpu.memory_used is not None:
                total = fmt.data_size(gpu.memory_total) if gpu.memory_total else "?"
                fraction = self._shown["vram"] if gpu.memory_total else None
                rows.append(
                    ("VRAM", fraction, f"{fmt.data_size(gpu.memory_used)} / {total}", gpu.name)
                )

        label_font = mono_font(8.0, bold=True, spacing=1.5)
        value_font = mono_font(8.0)
        detail_font = mono_font(7.0)
        metrics = QFontMetrics(value_font)
        row_height = 34.0
        spark_height = max(18.0, min(80.0, rect.height() - row_height * len(rows) - 8))
        y = rect.top()
        for index, (label, fraction, value, detail) in enumerate(rows):
            painter.setFont(label_font)
            painter.setPen(color(PALETTE.accent))
            painter.drawText(QRectF(rect.left(), y, 44, 14), Qt.AlignmentFlag.AlignVCenter, label)
            value_width = metrics.horizontalAdvance(value) + 6
            bar = QRectF(rect.left() + 46, y + 2, max(40.0, rect.width() - 52 - value_width), 10)
            paint_bar(painter, bar, fraction, self._time + index * 0.3, ticks=10)
            painter.setFont(value_font)
            painter.setPen(color(PALETTE.ice))
            painter.drawText(
                QRectF(bar.right() + 6, y, value_width, 14),
                Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
                value,
            )
            painter.setFont(detail_font)
            painter.setPen(color(PALETTE.text_dim))
            painter.drawText(QPointF(rect.left() + 46, y + 23), detail)
            y += row_height
        self._paint_sparkline(
            painter,
            QRectF(rect.left() + 46, rect.bottom() - spark_height, rect.width() - 46, spark_height),
        )

    def _paint_sparkline(self, painter: QPainter, area: QRectF) -> None:
        history = list(self._cpu_history)
        painter.setFont(mono_font(6.5, bold=True, spacing=1.0))
        painter.setPen(color(PALETTE.text_dim))
        painter.drawText(QPointF(area.left() - 46, area.bottom() - 4), "CPU 60s")
        painter.setPen(QPen(color(PALETTE.grid, 120), 1.0))
        painter.drawLine(area.bottomLeft(), area.bottomRight())
        if len(history) < 2:
            return
        step = area.width() / (HISTORY - 1)
        start = area.right() - step * (len(history) - 1)
        path = QPainterPath()
        for k, value in enumerate(history):
            point = QPointF(
                start + step * k, area.bottom() - area.height() * min(1.0, value / 100.0)
            )
            if k == 0:
                path.moveTo(point)
            else:
                path.lineTo(point)
        area_path = QPainterPath(path)
        area_path.lineTo(QPointF(area.right(), area.bottom()))
        area_path.lineTo(QPointF(start, area.bottom()))
        area_path.closeSubpath()
        fill = QLinearGradient(area.topLeft(), area.bottomLeft())
        fill.setColorAt(0.0, color(PALETTE.accent, 80))
        fill.setColorAt(1.0, color(PALETTE.accent, 0))
        painter.fillPath(area_path, fill)
        painter.setPen(QPen(color(PALETTE.accent, 200), 1.2))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawPath(path)

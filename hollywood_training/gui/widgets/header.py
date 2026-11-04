"""``HeaderBar``: product title, the training network's class name, state and device.

The title and the network name (rendered with a neon glow) are cached; the rotating
emblem and the state badge are drawn every frame.
"""

from __future__ import annotations

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QFontMetrics, QLinearGradient, QPainter, QPen, QPixmap
from PySide6.QtWidgets import QWidget

from ... import formatting as fmt
from ..._version import __version__
from ...sync.events import TrainingState
from ...sync.session import SessionState
from ..animation import AnimationClock, pulse
from ..theme import PALETTE, color, draw_glow_text, mono_font

STATE_BADGES: dict[TrainingState, tuple[str, str, bool]] = {
    TrainingState.PREPARING: ("BOOTING", PALETTE.accent_soft, True),
    TrainingState.RUNNING: ("TRAINING", PALETTE.accent, True),
    TrainingState.PAUSED: ("PAUSED", PALETTE.electric, True),
    TrainingState.FINISHED: ("COMPLETE", PALETTE.success, False),
    TrainingState.STOPPED: ("STOPPED", PALETTE.text, False),
    TrainingState.FAILED: ("FAILURE", PALETTE.error, True),
}


class HeaderBar(QWidget):
    HEIGHT = 80

    def __init__(
        self, session: SessionState, clock: AnimationClock, parent: QWidget | None = None
    ) -> None:
        super().__init__(parent)
        self._session = session
        self._time = 0.0
        self._cache: QPixmap | None = None
        self._cache_key: tuple | None = None
        self.setFixedHeight(self.HEIGHT)
        clock.tick.connect(self._advance)

    def _advance(self, dt: float) -> None:
        self._time += dt
        self.update()

    def _ensure_cache(self) -> None:
        info = self._session.info
        name = self._session.network_name or "—"
        alias = info.display_name if info else None
        params = info.architecture.total_parameters if info else None
        dpr = round(self.devicePixelRatioF(), 2)
        key = (name, alias, params, self.width(), self.height(), dpr)
        if key == self._cache_key:
            return
        self._cache_key = key
        pixmap = QPixmap(max(1, int(self.width() * dpr)), max(1, int(self.height() * dpr)))
        pixmap.setDevicePixelRatio(dpr)
        pixmap.fill(Qt.GlobalColor.transparent)
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        width, height = self.width(), self.height()

        strip = QLinearGradient(QPointF(0, 0), QPointF(0, height))
        strip.setColorAt(0.0, color(PALETTE.void, 150))
        strip.setColorAt(1.0, color(PALETTE.void, 30))
        painter.fillRect(QRectF(0, 0, width, height - 2), strip)
        line = QLinearGradient(QPointF(0, 0), QPointF(width, 0))
        line.setColorAt(0.0, color(PALETTE.accent, 0))
        line.setColorAt(0.5, color(PALETTE.accent, 190))
        line.setColorAt(1.0, color(PALETTE.accent, 0))
        painter.setPen(QPen(line, 1.5))
        painter.drawLine(QPointF(0, height - 1.5), QPointF(width, height - 1.5))

        title_font = mono_font(15, bold=True, spacing=3.0)
        draw_glow_text(
            painter,
            QPointF(86, 38),
            "HOLLYWOOD TRAINING",
            title_font,
            color(PALETTE.ice),
            color(PALETTE.accent),
        )
        painter.setFont(mono_font(7.5, spacing=2.0))
        painter.setPen(color(PALETTE.text_dim))
        painter.drawText(QPointF(88, 58), f"NEURAL TRAINING VISUALIZER · v{__version__}")

        center = width / 2
        label_font = mono_font(7.5, bold=True, spacing=3.0)
        painter.setFont(label_font)
        painter.setPen(color(PALETTE.text_dim))
        label = "TRAINING NETWORK"
        painter.drawText(
            QPointF(center - QFontMetrics(label_font).horizontalAdvance(label) / 2, 18), label
        )
        name_font = mono_font(19, bold=True, spacing=2.0)
        name_width = QFontMetrics(name_font).horizontalAdvance(name)
        draw_glow_text(
            painter,
            QPointF(center - name_width / 2, 46),
            name,
            name_font,
            color(PALETTE.ice),
            color(PALETTE.accent),
            1.2,
        )
        details = [
            part for part in (alias, f"{fmt.count(params)} parameters" if params else None) if part
        ]
        if details:
            detail_font = mono_font(8.0, spacing=1.0)
            text = "  ·  ".join(details)
            painter.setFont(detail_font)
            painter.setPen(color(PALETTE.accent_soft))
            painter.drawText(
                QPointF(center - QFontMetrics(detail_font).horizontalAdvance(text) / 2, 64), text
            )
        painter.end()
        self._cache = pixmap

    def paintEvent(self, _event) -> None:
        self._ensure_cache()
        painter = QPainter(self)
        if self._cache is not None:
            painter.drawPixmap(0, 0, self._cache)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        self._paint_emblem(painter, QPointF(42, self.height() / 2 - 2))
        self._paint_badge(painter)
        painter.end()

    def _paint_emblem(self, painter: QPainter, center: QPointF) -> None:
        t = self._time
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.setPen(QPen(color(PALETTE.accent, 55), 1.0))
        painter.drawEllipse(center, 24, 24)
        arc = QPen(color(PALETTE.accent), 2.4)
        arc.setCapStyle(Qt.PenCapStyle.RoundCap)
        painter.setPen(arc)
        outer = QRectF(center.x() - 24, center.y() - 24, 48, 48)
        start = int(-t * 140 * 16) % 5760
        painter.drawArc(outer, start, 100 * 16)
        painter.drawArc(outer, (start + 180 * 16) % 5760, 60 * 16)
        painter.setPen(QPen(color(PALETTE.electric), 2.0))
        inner = QRectF(center.x() - 15, center.y() - 15, 30, 30)
        painter.drawArc(inner, int(t * 220 * 16) % 5760, 140 * 16)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(color(PALETTE.ice))
        core = 3.5 + 1.8 * pulse(t, 1.1)
        painter.drawEllipse(center, core, core)

    def _paint_badge(self, painter: QPainter) -> None:
        state = self._session.state
        text, ink, blink = STATE_BADGES.get(state, ("—", PALETTE.text, False))
        font = mono_font(9.0, bold=True, spacing=2.5)
        metrics = QFontMetrics(font)
        width = metrics.horizontalAdvance(text) + 40
        rect = QRectF(self.width() - width - 24, 14, width, 26)
        glow = pulse(self._time, 1.2) if blink else 1.0
        painter.setPen(QPen(color(ink, int(120 + 120 * glow)), 1.4))
        painter.setBrush(color(ink, int(18 + 22 * glow)))
        painter.drawRoundedRect(rect, 3, 3)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(color(ink, int(80 + 175 * glow)))
        painter.drawEllipse(QPointF(rect.left() + 14, rect.center().y()), 4, 4)
        painter.setFont(font)
        painter.setPen(color(PALETTE.ice))
        painter.drawText(
            QRectF(rect.left() + 24, rect.top(), width - 28, rect.height()),
            Qt.AlignmentFlag.AlignVCenter,
            text,
        )

        info = self._session.info
        if info is not None:
            small = mono_font(7.5, spacing=1.0)
            device = f"DEVICE {info.device.upper()}"
            painter.setFont(small)
            painter.setPen(color(PALETTE.text_dim))
            right = self.width() - 24
            painter.drawText(
                QPointF(right - QFontMetrics(small).horizontalAdvance(device), 58), device
            )

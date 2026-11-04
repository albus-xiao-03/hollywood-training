"""``HudPanel``: the framed, titled panel every GUI section is built on.

The frame has chamfered corners, a translucent blue fill, bright corner brackets and an
uppercase title with a small panel code. Subclasses either paint their content in
:meth:`HudPanel.paint_content` or place child widgets inside :meth:`contentsRect`.
"""

from __future__ import annotations

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QFontMetrics, QLinearGradient, QPainter, QPen
from PySide6.QtWidgets import QWidget

from ..theme import PALETTE, chamfer_path, color, mono_font

TITLE_HEIGHT = 26
CUT = 10.0


def paint_bar(
    painter: QPainter,
    rect: QRectF,
    fraction: float | None,
    time: float,
    *,
    start: str = PALETTE.electric,
    end: str = PALETTE.accent,
    ticks: int = 10,
) -> None:
    """A HUD progress / level bar. ``fraction`` None draws moving "unknown" stripes."""
    painter.save()
    painter.setPen(QPen(color(PALETTE.panel_edge, 150), 1.0))
    painter.setBrush(color(PALETTE.void, 160))
    painter.drawRect(rect)
    inner = rect.adjusted(2, 2, -2, -2)
    painter.setClipRect(inner)
    painter.setPen(Qt.PenStyle.NoPen)
    if fraction is None:
        stripe = 12.0
        offset = (time * 40.0) % (stripe * 2)
        x = inner.left() - stripe * 2 + offset
        painter.setBrush(color(start, 120))
        while x < inner.right():
            painter.drawRect(QRectF(x, inner.top(), stripe, inner.height()))
            x += stripe * 2
    else:
        filled = QRectF(
            inner.left(), inner.top(), inner.width() * max(0.0, min(1.0, fraction)), inner.height()
        )
        gradient = QLinearGradient(inner.topLeft(), inner.topRight())
        gradient.setColorAt(0.0, color(start))
        gradient.setColorAt(1.0, color(end))
        painter.fillRect(filled, gradient)
        if filled.width() > 6:  # a highlight sweeping along the filled part
            x = filled.left() + (time * 0.55 % 1.0) * filled.width()
            shine = QLinearGradient(QPointF(x - 18, 0), QPointF(x + 18, 0))
            shine.setColorAt(0.0, color(PALETTE.ice, 0))
            shine.setColorAt(0.5, color(PALETTE.ice, 110))
            shine.setColorAt(1.0, color(PALETTE.ice, 0))
            painter.fillRect(
                QRectF(x - 18, filled.top(), 36, filled.height()).intersected(filled), shine
            )
    painter.setPen(QPen(color(PALETTE.void, 170), 1.0))
    for k in range(1, ticks):
        x = inner.left() + inner.width() * k / ticks
        painter.drawLine(QPointF(x, inner.top()), QPointF(x, inner.bottom()))
    painter.restore()


class HudPanel(QWidget):
    def __init__(self, title: str, code: str = "", parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._title = title.upper()
        self._code = code.upper()
        self.setContentsMargins(12, TITLE_HEIGHT + 6, 12, 10)

    @property
    def title(self) -> str:
        return self._title

    @property
    def code(self) -> str:
        return self._code

    def set_code(self, code: str) -> None:
        """Change the small text at the right of the title strip."""
        self._code = code.upper()
        self.update()

    def content_rect(self) -> QRectF:
        return QRectF(self.contentsRect())

    def paintEvent(self, _event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.paint_frame(painter)
        self.paint_content(painter, self.content_rect())
        painter.end()

    def paint_content(self, painter: QPainter, rect: QRectF) -> None:
        """Override to draw the panel's content."""

    def paint_frame(self, painter: QPainter) -> None:
        rect = QRectF(self.rect()).adjusted(1.5, 1.5, -1.5, -1.5)
        path = chamfer_path(rect, CUT)

        fill = QLinearGradient(rect.topLeft(), rect.bottomLeft())
        fill.setColorAt(0.0, color(PALETTE.panel, 200))
        fill.setColorAt(1.0, color(PALETTE.panel, 135))
        painter.fillPath(path, fill)
        painter.setPen(QPen(color(PALETTE.panel_edge, 150), 1.0))
        painter.drawPath(path)

        # bright brackets on the square corners, a highlight on the cut ones
        bracket = QPen(color(PALETTE.accent, 230), 2.0)
        bracket.setCapStyle(Qt.PenCapStyle.SquareCap)
        painter.setPen(bracket)
        arm = 14.0
        top_right, bottom_left = rect.topRight(), rect.bottomLeft()
        painter.drawLine(QPointF(top_right.x() - arm, top_right.y()), top_right)
        painter.drawLine(top_right, QPointF(top_right.x(), top_right.y() + arm))
        painter.drawLine(QPointF(bottom_left.x(), bottom_left.y() - arm), bottom_left)
        painter.drawLine(bottom_left, QPointF(bottom_left.x() + arm, bottom_left.y()))
        painter.setPen(QPen(color(PALETTE.accent, 160), 1.5))
        painter.drawLine(
            QPointF(rect.left(), rect.top() + CUT), QPointF(rect.left() + CUT, rect.top())
        )
        painter.drawLine(
            QPointF(rect.right() - CUT, rect.bottom()), QPointF(rect.right(), rect.bottom() - CUT)
        )

        # title strip
        font = mono_font(8.5, bold=True, spacing=2.0)
        painter.setFont(font)
        painter.setPen(color(PALETTE.accent))
        baseline = rect.top() + 18
        painter.drawText(QPointF(rect.left() + CUT + 6, baseline), f"▸ {self._title}")
        title_width = QFontMetrics(font).horizontalAdvance(f"▸ {self._title}")
        code_width = 0
        if self._code:
            code_font = mono_font(7.5, spacing=1.5)
            painter.setFont(code_font)
            painter.setPen(color(PALETTE.text_dim))
            code_width = QFontMetrics(code_font).horizontalAdvance(self._code)
            painter.drawText(QPointF(rect.right() - code_width - 18, baseline), self._code)
        line_start = rect.left() + CUT + 14 + title_width
        line_end = rect.right() - 18 - (code_width + 12 if self._code else 0)
        if line_end > line_start:
            underline = QLinearGradient(QPointF(line_start, 0), QPointF(line_end, 0))
            underline.setColorAt(0.0, color(PALETTE.accent, 120))
            underline.setColorAt(1.0, color(PALETTE.accent, 0))
            painter.setPen(QPen(underline, 1.0))
            painter.drawLine(QPointF(line_start, baseline - 4), QPointF(line_end, baseline - 4))

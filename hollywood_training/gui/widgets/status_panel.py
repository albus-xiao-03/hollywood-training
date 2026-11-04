"""``StatusPanel``: everything about the run - counters, progress, timer and settings.

Shows the total / current epoch, batch progress, batch size, learning rate, loss
function, optimizer, normalisation strategy, scheduler, device, losses and metrics, the
best epoch, the process timer and the ETA. Values are read from the
:class:`~hollywood_training.sync.session.SessionState` on every paint, so the timer ticks
smoothly between events while the numbers always match the last synchronised epoch.
"""

from __future__ import annotations

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QFontMetrics, QPainter, QPen
from PySide6.QtWidgets import QWidget

from ... import formatting as fmt
from ...sync.events import Phase, TrainingState
from ...sync.session import SessionState
from ..animation import AnimationClock
from ..theme import PALETTE, color, draw_glow_mono, glow_text_width, mono_font
from .hud import HudPanel, paint_bar

Row = tuple[tuple[str, str], ...]  # one or two (label, value) pairs


class StatusPanel(HudPanel):
    def __init__(
        self, session: SessionState, clock: AnimationClock, parent: QWidget | None = None
    ) -> None:
        super().__init__("Training status", "STS-03", parent)
        self._session = session
        self._time = 0.0
        self.setMinimumSize(360, 300)
        clock.tick.connect(self._advance)

    def _advance(self, dt: float) -> None:
        self._time += dt
        self.update()

    # ----------------------------------------------------------------- content
    def rows(self) -> list[Row]:
        """The key/value rows shown below the progress bars."""
        s = self._session
        info = s.info
        if info is None:
            return []
        last = s.last_record
        best = s.best_record()
        optimizer = info.optimizer + (
            f" ({info.optimizer_details})" if info.optimizer_details else ""
        )
        val_loss = fmt.loss(last.val_loss) if last else fmt.PLACEHOLDER
        rows: list[Row] = [
            (
                ("BATCH SIZE", fmt.count(s.batch_size)),
                ("LEARNING RATE", fmt.learning_rate(s.learning_rate)),
            ),
            (("LOSS FUNCTION", info.loss_function),),
            (("OPTIMIZER", optimizer),),
            (("NORMALIZATION", info.normalization),),
            (("SCHEDULER", info.scheduler or "none"), ("DEVICE", info.device)),
            (
                ("TRAIN LOSS", fmt.loss(last.train_loss) if last else fmt.PLACEHOLDER),
                ("VAL LOSS", val_loss if info.has_validation else "n/a"),
            ),
        ]
        if last is not None:
            metrics = [
                (name.upper(), fmt.metric(name, v)) for name, v in last.train_metrics.items()
            ]
            metrics += [
                (f"VAL {name.upper()}", fmt.metric(name, v)) for name, v in last.val_metrics.items()
            ]
            rows += [tuple(metrics[k : k + 2]) for k in range(0, len(metrics), 2)]
        if best is not None:
            value = best.monitored_loss
            rows.append(
                (
                    ("BEST LOSS", f"{fmt.loss(value)} @ {best.epoch}"),
                    ("EPOCH TIME", f"{last.duration:.3f} s" if last else fmt.PLACEHOLDER),
                )
            )
        rows.append(
            (
                ("PARAMETERS", fmt.count(info.architecture.total_parameters)),
                ("EPOCH DELAY", f"{s.delay * 1000:g} ms"),
            )
        )
        return rows

    def paint_content(self, painter: QPainter, rect: QRectF) -> None:
        s = self._session
        if s.info is None:
            painter.setFont(mono_font(10, bold=True, spacing=3.0))
            painter.setPen(color(PALETTE.accent, 160))
            painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, "STANDING BY")
            return
        left, right = rect.left(), rect.right()
        label_font = mono_font(7.5, bold=True, spacing=1.5)
        big = mono_font(16, bold=True, spacing=1.0)

        # counters: epoch (left), process timer (right)
        painter.setFont(label_font)
        painter.setPen(color(PALETTE.text_dim))
        painter.drawText(QPointF(left, rect.top() + 9), "EPOCH")
        timer_label = "PROCESS TIMER"
        label_width = QFontMetrics(label_font).horizontalAdvance(timer_label)
        painter.drawText(QPointF(right - label_width, rect.top() + 9), timer_label)
        width = len(str(max(1, s.total_epochs)))
        epoch_text = f"{s.epoch:0{width}d}/{s.total_epochs:0{width}d}"
        baseline = rect.top() + 32
        ice, accent = color(PALETTE.ice), color(PALETTE.accent)
        draw_glow_mono(painter, QPointF(left, baseline), epoch_text, big, ice, accent, 0.8)
        timer = fmt.duration(s.elapsed_now())
        timer_left = right - glow_text_width(timer, big)
        draw_glow_mono(painter, QPointF(timer_left, baseline), timer, big, accent, accent, 0.7)

        # epoch progress
        y = rect.top() + 42
        small = mono_font(7.5)
        painter.setFont(small)
        painter.setPen(color(PALETTE.text))
        painter.drawText(QPointF(left, y + 9), f"RUN {fmt.percent(s.epoch_progress())}")
        eta = s.eta()
        eta_text = (
            f"ETA {fmt.duration(eta, millis=False)}" if eta is not None else _state_hint(s.state)
        )
        eta_width = QFontMetrics(small).horizontalAdvance(eta_text)
        painter.drawText(QPointF(right - eta_width, y + 9), eta_text)
        paint_bar(painter, QRectF(left, y + 13, rect.width(), 11), s.epoch_progress(), self._time)

        # batch progress
        y += 32
        phase = "VALIDATION" if s.phase is Phase.VALIDATION else "TRAINING"
        total = s.total_batches if s.total_batches else "?"
        batch_text = f"BATCH {s.batch}/{total} · {phase}" if s.batch else "BATCH —"
        painter.setPen(color(PALETTE.text))
        painter.drawText(QPointF(left, y + 9), batch_text)
        batch_fraction = s.batch_progress() if s.state is TrainingState.RUNNING or s.batch else 0.0
        end = PALETTE.indigo if s.phase is Phase.VALIDATION else PALETTE.accent
        paint_bar(
            painter,
            QRectF(left, y + 13, rect.width(), 8),
            batch_fraction,
            self._time * 1.4,
            end=end,
            ticks=20,
        )

        # settings and results
        y += 32
        painter.setPen(QPen(color(PALETTE.grid, 160), 1.0))
        painter.drawLine(QPointF(left, y), QPointF(right, y))
        y += 6
        key_font = mono_font(7.5, bold=True, spacing=1.0)
        value_font = mono_font(8.5)
        key_metrics, value_metrics = QFontMetrics(key_font), QFontMetrics(value_font)
        rows = self.rows()
        # key column widths: measured from the labels in each column
        left_keys = [row[0][0] for row in rows]
        right_keys = [row[1][0] for row in rows if len(row) > 1]
        left_key = max((key_metrics.horizontalAdvance(k) for k in left_keys), default=0) + 12
        right_key = max((key_metrics.horizontalAdvance(k) for k in right_keys), default=0) + 12
        minimum = value_metrics.height() + 3
        line = max(float(minimum), min(22.0, (rect.bottom() - y) / max(1, len(rows))))
        half = rect.width() / 2
        for row in rows:
            if y + minimum > rect.bottom() + 2:
                break
            if len(row) == 1:
                cells = [(left, rect.width(), left_key)]
            else:
                cells = [(left, half - 10, left_key), (left + half, half, right_key)]
            for (label, value), (x, cell, key_width) in zip(row, cells, strict=False):
                painter.setFont(key_font)
                painter.setPen(color(PALETTE.text_dim))
                painter.drawText(
                    QRectF(x, y, key_width, line), Qt.AlignmentFlag.AlignVCenter, label
                )
                painter.setFont(value_font)
                painter.setPen(color(PALETTE.ice))
                room = int(cell - key_width - 4)
                shown = value_metrics.elidedText(value, Qt.TextElideMode.ElideMiddle, room)
                box = QRectF(x + key_width, y, cell - key_width, line)
                painter.drawText(box, Qt.AlignmentFlag.AlignVCenter, shown)
            y += line


def _state_hint(state: TrainingState) -> str:
    return {
        TrainingState.PAUSED: "PAUSED",
        TrainingState.FINISHED: "COMPLETE",
        TrainingState.STOPPED: "STOPPED",
        TrainingState.FAILED: "FAILED",
    }.get(state, "ETA —")

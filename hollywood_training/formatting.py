"""Text formatting shared by the terminal commands, the console frontend and the GUI."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .sync.events import EpochRecord

PLACEHOLDER = "—"


def _missing(value: float | None) -> bool:
    return value is None or (isinstance(value, float) and math.isnan(value))


def duration(seconds: float | None, *, millis: bool = True) -> str:
    """``HH:MM:SS.mmm`` (or ``HH:MM:SS``)."""
    if _missing(seconds) or seconds is None or seconds < 0:
        return PLACEHOLDER
    whole = int(seconds)
    hours, rest = divmod(whole, 3600)
    minutes, secs = divmod(rest, 60)
    text = f"{hours:02d}:{minutes:02d}:{secs:02d}"
    if millis:
        text += f".{int((seconds - whole) * 1000):03d}"
    return text


def loss(value: float | None) -> str:
    """Loss values: fixed point for ordinary magnitudes, scientific otherwise."""
    if _missing(value) or value is None:
        return PLACEHOLDER
    if math.isinf(value):
        return "inf" if value > 0 else "-inf"
    if value == 0 or 1e-3 <= abs(value) < 1e4:
        return f"{value:.4f}"
    return f"{value:.3e}"


def learning_rate(value: float | None) -> str:
    if _missing(value) or value is None:
        return PLACEHOLDER
    return f"{value:.4g}"


def metric(name: str, value: float) -> str:
    """Metrics whose name suggests a ratio are shown as percentages."""
    lowered = name.lower()
    if any(word in lowered for word in ("acc", "precision", "recall", "f1")) and 0 <= value <= 1:
        return f"{value * 100:.1f}%"
    return loss(value)


def percent(fraction: float | None, digits: int = 1) -> str:
    if _missing(fraction) or fraction is None:
        return PLACEHOLDER
    return f"{fraction * 100:.{digits}f}%"


def data_size(num_bytes: float | None) -> str:
    if _missing(num_bytes) or num_bytes is None:
        return PLACEHOLDER
    value = float(num_bytes)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if abs(value) < 1024 or unit == "TiB":
            return f"{value:.0f} {unit}" if unit == "B" else f"{value:.1f} {unit}"
        value /= 1024
    return f"{value:.1f} TiB"


def count(value: int | None) -> str:
    return PLACEHOLDER if value is None else f"{value:,}"


def epoch_summary(record: EpochRecord, total_epochs: int) -> str:
    """One line per epoch, e.g. ``epoch 12/200 | train 0.4123 | val 0.4555 | ...``."""
    width = len(str(total_epochs))
    parts = [f"epoch {record.epoch:>{width}}/{total_epochs}", f"train {loss(record.train_loss)}"]
    if record.val_loss is not None:
        parts.append(f"val {loss(record.val_loss)}")
    parts += [f"{name} {metric(name, value)}" for name, value in record.train_metrics.items()]
    parts += [f"val {name} {metric(name, value)}" for name, value in record.val_metrics.items()]
    parts += [f"lr {learning_rate(record.learning_rate)}", f"{record.duration:.3f}s"]
    return " | ".join(parts)

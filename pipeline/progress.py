"""Progress reporting shared by the pipeline stages and the UI.

Stages write to a :class:`RunLog` and yield :class:`Update` objects; the UI
turns each update plus the log's telemetry into one Gradio output tuple.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

# Redraw the live loss curve every N epochs (and on the last one).
LIVE_PLOT_UPDATE_EVERY = 2

# ~40 KB of visible text; enough for 200+ epochs.
_LOG_CHAR_BUDGET = 40_000


def fmt_eta(seconds: float) -> str:
    if seconds < 60:   return f"ETA: {int(seconds)}s"
    if seconds < 3600: return f"ETA: {int(seconds/60)}m {int(seconds%60)}s"
    return f"ETA: {seconds/3600:.1f}h"


def phase_eta_text(phase: str) -> str:
    phase_name = (phase or "working").replace("_", " ")
    return f"ETA: {phase_name}"


@dataclass
class Update:
    """One progress update. The status text is always the run log so far."""

    loss_history: list[dict] | None = None  # redraw the loss curve from these epochs
    eta: str | None = None                  # defaults to the log's current ETA text
    tsne_fig: Any = None                    # show this cluster plot
    train_summary: str = ""


class RunLog:
    """The run's status log plus the telemetry shown in the live metrics card."""

    def __init__(self) -> None:
        self.lines: list[str] = []
        self.telemetry: dict = {
            "phase": "Waiting",
            "status_log": "Initialising…",
            "eta_text": "ETA: estimating…",
            "epoch_text": "—",
            "latest_metrics": {},
            "progress_label": "Preparing run",
        }

    def log(self, msg: str) -> str:
        self.lines.append(msg)
        # Cap by character budget rather than line count — some messages (e.g.
        # cleaning reports, tracebacks) contain many embedded newlines and would
        # blow through a per-call cap without ever being trimmed.
        combined = "\n".join(self.lines)
        if len(combined) > _LOG_CHAR_BUDGET:
            # Drop the oldest half and rebuild
            self.lines = self.lines[len(self.lines) // 2 :]
            combined = "\n".join(self.lines)
        self.telemetry["status_log"] = combined
        return combined

    def set_phase(self, phase: str, *, eta_text: str | None = None, status_line: str | None = None) -> None:
        self.telemetry["phase"] = phase
        self.telemetry["progress_label"] = phase
        self.telemetry["eta_text"] = eta_text or phase_eta_text(phase)
        if status_line:
            self.telemetry["status_log"] = status_line

    def set_eta(self, eta_text: str) -> None:
        self.telemetry["eta_text"] = eta_text

    def update_epoch_metrics(self, *, epoch_text=None, train_loss=None, val_loss=None, metric=None, lr_value=None):
        if epoch_text is not None:
            self.telemetry["epoch_text"] = epoch_text
        metrics = self.telemetry["latest_metrics"]
        if train_loss is not None:
            metrics["train_loss"] = str(train_loss)
        if val_loss is not None:
            metrics["val_loss"] = str(val_loss)
        if metric is not None:
            metrics["metric"] = str(metric)
        if lr_value is not None:
            metrics["lr"] = str(lr_value)

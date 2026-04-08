"""
Utilities for comparing multiple training runs side by side.
Stores run metadata in outputs/run_history.json.
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path

logger = logging.getLogger(__name__)

# Resolve relative to this file so the path is correct regardless of CWD.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
HISTORY_FILE  = _PROJECT_ROOT / "outputs" / "run_history.json"


def save_run(
    model_name: str,
    modality: str,
    task: str,
    training_mode: str,
    hyperparams: dict,
    history: list[dict],
    bundle_path: str,
    metrics: dict | None = None,
) -> None:
    HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
    runs = _load_raw()
    runs.append({
        "id":            len(runs),
        "timestamp":     time.strftime("%Y-%m-%d %H:%M"),
        "model_name":    model_name,
        "modality":      modality,
        "task":          task,
        "training_mode": training_mode,
        "hyperparams":   hyperparams,
        "history":       history,
        "bundle_path":   bundle_path,
        "metrics":       metrics or {},
        "final_val_acc":  history[-1].get("val_acc",  0) if history else 0,
        "final_val_loss": history[-1].get("val_loss", 0) if history else 0,
    })
    HISTORY_FILE.write_text(json.dumps(runs, indent=2))


def load_history() -> list[dict]:
    return _load_raw()


def _load_raw() -> list[dict]:
    if not HISTORY_FILE.exists():
        return []
    try:
        return json.loads(HISTORY_FILE.read_text())
    except Exception as e:
        logger.warning("Could not read run history from %s: %s", HISTORY_FILE, e)
        return []


def compare_runs_plot(run_ids: list[int]) -> "matplotlib.figure.Figure":
    """Plot val_acc and val_loss curves for multiple runs."""
    import matplotlib.pyplot as plt
    runs = {r["id"]: r for r in load_history()}
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    for rid in run_ids:
        if rid not in runs:
            continue
        run  = runs[rid]
        hist = run["history"]
        label = f"[{rid}] {run['model_name']} ({run['modality']})"
        epochs = [h["epoch"] for h in hist]
        ax1.plot(epochs, [h.get("val_loss", 0) for h in hist], label=label)
        ax2.plot(epochs, [h.get("val_acc",  0) for h in hist], label=label)
    ax1.set(xlabel="Epoch", ylabel="Val Loss",     title="Validation Loss")
    ax2.set(xlabel="Epoch", ylabel="Val Acc / MAE", title="Validation Metric")
    for ax in (ax1, ax2):
        ax.legend(fontsize=7)
    fig.tight_layout()
    return fig


def history_dataframe() -> "pd.DataFrame":
    import pandas as pd
    runs = load_history()
    if not runs:
        return pd.DataFrame(columns=["id", "timestamp", "model_name", "modality",
                                      "task", "final_val_acc", "bundle_path"])
    return pd.DataFrame([{
        "id":       r["id"],
        "timestamp": r["timestamp"],
        "model":    r["model_name"],
        "modality": r["modality"],
        "task":     r["task"],
        "mode":     r["training_mode"],
        "val_acc":  r["final_val_acc"],
        "val_loss": r["final_val_loss"],
        "bundle":   r["bundle_path"],
    } for r in runs])

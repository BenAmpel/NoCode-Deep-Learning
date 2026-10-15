"""Run the pipeline stages in order.

Stages raise on failure; they never turn an exception into status text. The
UI's ``run_pipeline`` generator formats errors for display, and
``run_model_sweep`` records them per model.
"""
from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Callable, Iterator

from config import DEFAULTS, DEVICE, set_seed
from pipeline.config import RunConfig
from pipeline.context import RunContext, RunResult
from pipeline.data import load_data
from pipeline.evaluate import evaluate
from pipeline.export import export
from pipeline.model import build_model
from pipeline.progress import RunLog, Update
from pipeline.train import train
from training import cancel as _cancel
from ui.project_state import load_project_state, save_project_state

OUTPUTS_ROOT = Path(__file__).resolve().parent.parent / "outputs"


class PreflightError(ValueError):
    """The settings cannot start a run; ``errors`` holds one message per problem."""

    def __init__(self, errors: list[str]):
        super().__init__("\n".join(errors))
        self.errors = errors


def preflight_errors(cfg: RunConfig) -> list[str]:
    errors: list[str] = []
    if not cfg.data_path or not str(cfg.data_path).strip():
        errors.append("❌ **No data path provided.** Upload a dataset or enter a path.")
    if not cfg.model_name:
        errors.append("❌ **No model selected.** Choose an architecture in the Model tab.")
    if cfg.task == "anomaly" and cfg.model_name != "Autoencoder":
        errors.append(
            "❌ **Anomaly detection requires the Autoencoder model.** "
            f"Selected model '{cfg.model_name}' does not support anomaly detection. "
            "Switch to Autoencoder or change the task."
        )
    return errors


def safe_bundle_name(cfg: RunConfig) -> str:
    """The bundle name with anything but letters, digits, _ and - replaced."""
    return re.sub(r"[^a-zA-Z0-9_\-]", "_", str(cfg.bundle_name).strip()) or "my_model"


def _no_progress(*_args, **_kwargs) -> None:
    pass


def prepare_run(cfg: RunConfig, log: RunLog, progress: Callable | None = None) -> RunContext:
    """Remember the preprocessing choices and reset per-run global state."""
    save_project_state({
        **load_project_state(),
        "preprocessing": {
            "augmentation": cfg.augmentation,
            "val_split": cfg.val_split,
            "tabular_missing_strategy": cfg.tabular_missing_strategy,
            "tabular_clip_outliers": bool(cfg.tabular_clip_outliers),
            "tabular_scaling": cfg.tabular_scaling,
            "text_lowercase": bool(cfg.text_lowercase),
            "text_strip_urls": bool(cfg.text_strip_urls),
            "text_strip_punctuation": bool(cfg.text_strip_punctuation),
            "text_remove_stopwords": bool(cfg.text_remove_stopwords),
            "text_deduplicate": bool(cfg.text_deduplicate),
            "text_apply_stemming": bool(cfg.text_apply_stemming),
            "text_apply_lemmatization": bool(cfg.text_apply_lemmatization),
            "text_use_ngrams": bool(cfg.text_use_ngrams),
            "image_size": int(cfg.image_size),
            "image_normalization": cfg.image_normalization,
            "image_force_grayscale": bool(cfg.image_force_grayscale),
        },
    })

    ctx = RunContext(
        cfg=cfg,
        log=log,
        progress=progress or _no_progress,
        bundle_name=safe_bundle_name(cfg),
        device=DEVICE,
        runtime_device="cpu" if (cfg.modality == "video" and DEVICE == "mps") else DEVICE,
        data_path=cfg.data_path,
    )
    set_seed(DEFAULTS["seed"])
    _cancel.reset()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.close("all")  # figures from the previous run are already rendered
    return ctx


def execute(ctx: RunContext) -> Iterator[Update | RunResult]:
    """Run every stage, yielding progress updates and finally the RunResult."""
    yield from load_data(ctx)
    yield from build_model(ctx)
    yield from train(ctx)
    yield from evaluate(ctx)
    yield from export(ctx, str(OUTPUTS_ROOT))
    yield ctx.result


def cleanup(ctx: RunContext) -> None:
    for path in ctx.temp_files:
        try:
            os.unlink(path)
        except OSError:
            pass


def run(cfg: RunConfig, log: RunLog | None = None, progress: Callable | None = None) -> Iterator[Update | RunResult]:
    """The whole pipeline for callers that want exceptions, not status text.

    Raises PreflightError for unusable settings, ``training.cancel.TrainingStopped``
    when the user pressed Stop, and whatever a stage raised otherwise.
    """
    errors = preflight_errors(cfg)
    if errors:
        raise PreflightError(errors)
    log = log or RunLog()
    log.log("Initialising…")
    log.set_phase("initialising", eta_text="ETA: estimating…")
    ctx = prepare_run(cfg, log, progress)
    try:
        yield from execute(ctx)
    finally:
        cleanup(ctx)

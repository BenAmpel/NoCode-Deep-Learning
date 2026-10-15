"""State passed from one pipeline stage to the next."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from pipeline.config import RunConfig
from pipeline.progress import RunLog


@dataclass
class RunResult:
    """What a finished run produced; the UI renders its final screen from this."""

    bundle_path: str = ""
    history: list = field(default_factory=list)
    metrics: dict = field(default_factory=dict)
    eval_summary: str = ""
    training_summary: str = ""
    arch_fig: Any = None
    tsne_fig: Any = None
    cm_fig: Any = None
    roc_fig: Any = None
    gcam_fig: Any = None
    shap_fig: Any = None
    mis_img_fig: Any = None
    mis_df: Any = None
    reg_fig: Any = None
    anomaly_fig: Any = None
    text_shap_html: Any = None
    sklearn_model: bool = False


@dataclass
class RunContext:
    cfg: RunConfig
    log: RunLog
    progress: Callable
    bundle_name: str          # sanitised cfg.bundle_name
    device: str               # the detected accelerator
    runtime_device: str       # what this run trains on (video falls back from MPS)
    data_path: str            # cfg.data_path, or a cleaned temporary copy of it
    temp_files: list[str] = field(default_factory=list)  # removed when the run ends

    # Filled in by the stages, in order.
    extra_data: dict = field(default_factory=dict)
    train_loader: Any = None
    val_loader: Any = None
    classes: list = field(default_factory=list)
    prep: dict = field(default_factory=dict)
    num_classes: int = 0
    model: Any = None
    sklearn_model: bool = False
    graph_node2vec: bool = False
    final_model: Any = None
    stopped_early: bool = False
    clustering_result: Any = None
    evaluation_artifacts: dict = field(default_factory=dict)
    result: RunResult = field(default_factory=RunResult)

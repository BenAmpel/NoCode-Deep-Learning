"""Stage 5: export the model bundle and record the run in the history."""
from __future__ import annotations

import logging
from typing import Iterator

from pipeline.context import RunContext
from pipeline.progress import Update

logger = logging.getLogger("nocode_dl")


def export(ctx: RunContext, output_dir: str) -> Iterator[Update]:
    cfg, log = ctx.cfg, ctx.log
    train_loader, val_loader = ctx.train_loader, ctx.val_loader
    classes, prep = ctx.classes, ctx.prep
    sklearn_model, graph_node2vec = ctx.sklearn_model, ctx.graph_node2vec
    final_model = ctx.final_model
    safe_bundle_name = ctx.bundle_name
    history = ctx.result.history
    metrics_payload = ctx.result.metrics
    clustering_result = ctx.clustering_result
    evaluation_artifacts = ctx.evaluation_artifacts

    log.set_phase("exporting", eta_text="ETA: exporting")
    ctx.progress(0.95, desc="Exporting bundle…")
    log.log("Exporting ONNX bundle…")
    yield Update()

    sample_batch = None
    if not sklearn_model:
        # Any real batch works for tracing the ONNX graph; fall back to the
        # training loader when validation is empty (very small datasets).
        for loader in (val_loader, train_loader):
            if loader is None:
                continue
            try:
                sample_inputs, _ = next(iter(loader))
            except StopIteration:
                continue
            sample_batch = ({k: v[:1] for k, v in sample_inputs.items()}
                            if isinstance(sample_inputs, dict)
                            else sample_inputs[:1])
            break

    from export.exporter import export_bundle
    prep["model_name"] = cfg.model_name
    bundle_path, export_warnings = export_bundle(
        model=final_model,
        preprocessing_config=prep,
        classes=classes,
        bundle_name=safe_bundle_name,
        output_dir=output_dir,
        sample_batch=sample_batch,
        is_sklearn=(sklearn_model or graph_node2vec),
        clustering_result=clustering_result,
        evaluation_artifacts=evaluation_artifacts,
    )
    for w in export_warnings:
        log.log(w)
        yield Update()

    # ── 8. Save run history ───────────────────────────────────────────────
    try:
        from training.run_comparison import save_run
        save_run(
            model_name=cfg.model_name, modality=cfg.modality,
            task=cfg.task, training_mode=cfg.training_mode,
            hyperparams={"lr": cfg.lr, "batch_size": cfg.batch_size, "epochs": cfg.epochs,
                         "optimizer": cfg.optimizer, "scheduler": cfg.scheduler_name},
            history=history, bundle_path=bundle_path,
            metrics=metrics_payload,
        )
    except Exception as e:
        logger.warning("Could not save run history: %s", e)

    log.set_phase("complete", eta_text="ETA: Done")
    ctx.progress(1.0, desc="Done!")
    log.log(f"\n✅ Done!  Bundle saved to:\n  {bundle_path}")

    ctx.result.bundle_path = bundle_path

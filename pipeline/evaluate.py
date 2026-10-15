"""Stage 4: architecture view, clustering and evaluation metrics and plots."""
from __future__ import annotations

import logging
from typing import Iterator

from pipeline.context import RunContext
from pipeline.progress import Update

logger = logging.getLogger("nocode_dl")


def evaluate(ctx: RunContext) -> Iterator[Update]:
    cfg, log = ctx.cfg, ctx.log
    extra_data = ctx.extra_data
    train_loader, val_loader = ctx.train_loader, ctx.val_loader
    classes, prep, num_classes = ctx.classes, ctx.prep, ctx.num_classes
    sklearn_model, graph_node2vec = ctx.sklearn_model, ctx.graph_node2vec
    final_model = ctx.final_model
    runtime_device = ctx.runtime_device
    clustering_result = None
    eval_summary_text = ""
    metrics_payload: dict = {}
    evaluation_artifacts: dict = {}

    arch_fig = None
    if (
        final_model is not None
        and not sklearn_model
        and cfg.model_name != "Autoencoder"
        and hasattr(final_model, "named_modules")
    ):
        try:
            from ui.architecture_viz import summarise_model
            _, arch_fig = summarise_model(final_model)
        except Exception as e:
            logger.warning("Architecture visualisation failed: %s", e)

    # ── 5. Clustering ─────────────────────────────────────────────────────
    tsne_fig = None
    if cfg.task in ("clustering",) or cfg.model_name == "Autoencoder":
        log.set_phase("clustering", eta_text="ETA: clustering")
        ctx.progress(0.84, desc="Clustering embeddings…")
        if sklearn_model:
            log.log("ℹ️  sklearn models don't support embedding-based clustering.\n"
                         "   Use an MLP or Autoencoder for full clustering support.")
            yield Update()
        else:
            log.log("Extracting embeddings → KMeans…")
            yield Update()
            try:
                from training.clustering import run_clustering, tsne_plot
                k = int(cfg.n_clusters) if int(cfg.n_clusters) > 1 else num_classes
                clustering_result = run_clustering(final_model, train_loader, n_clusters=k)
                tsne_fig = tsne_plot(clustering_result["features"],
                                    clustering_result["cluster_labels"])
                log.log(f"  Silhouette score: {clustering_result['silhouette_score']}")
                yield Update(tsne_fig=tsne_fig)
            except Exception as e:
                log.log(f"  Clustering skipped: {e}")
                yield Update()

    # ── 6. Evaluation ─────────────────────────────────────────────────────
    cm_fig = roc_fig = gcam_fig = shap_fig = None
    mis_img_fig = None
    mis_df_val  = None
    reg_fig     = None
    anomaly_fig = None
    text_shap_html = None

    # Standard classification / multi-label evaluation
    if cfg.task in ("classification", "multi-label") and cfg.model_name != "Autoencoder":
        log.set_phase("evaluating", eta_text="ETA: evaluating")
        ctx.progress(0.88, desc="Evaluating…")
        log.log("Running evaluation…")
        yield Update()

        try:
            from eval.metrics import classification_metric_summary
            if cfg.modality == "graph":
                from eval.metrics import compute_confusion_matrix_from_arrays, compute_roc_curves_from_arrays
                from training.graph_trainer import predict_graph_classification
                import torch as _torch

                if graph_node2vec:
                    y_true, y_pred, y_prob = predict_graph_classification(
                        final_model,
                        extra_data,
                        sklearn_model=True,
                        embeddings=extra_data.get("graph_embeddings"),
                    )
                else:
                    y_true, y_pred, y_prob = predict_graph_classification(final_model, extra_data)
                cm_fig, eval_summary_text = compute_confusion_matrix_from_arrays(y_true, y_pred, classes)
                roc_fig = compute_roc_curves_from_arrays(y_true, y_prob, classes)
                metrics_payload = classification_metric_summary(
                    y_true, y_pred, y_prob, n_classes=len(classes)
                )
                feature_names = extra_data.get("feature_names") or []
                x_all = extra_data.get("x")
                if isinstance(x_all, _torch.Tensor):
                    x_all = x_all.cpu().numpy()
                val_idx_export = extra_data.get("val_idx")
                if isinstance(val_idx_export, _torch.Tensor):
                    val_idx_export = val_idx_export.cpu().numpy()
                feature_rows = []
                if x_all is not None and val_idx_export is not None and feature_names:
                    for row_position, node_idx in enumerate(val_idx_export[:2000]):
                        row = x_all[int(node_idx)]
                        feature_rows.append({
                            **{
                                feature_names[col_idx]: float(row[col_idx])
                                for col_idx in range(min(len(feature_names), len(row)))
                            },
                            "node_id": extra_data.get("node_ids", [])[int(node_idx)] if extra_data.get("node_ids") else str(int(node_idx)),
                            "y_true": int(y_true[row_position]),
                            "y_pred": int(y_pred[row_position]),
                            "is_error": bool(y_true[row_position] != y_pred[row_position]),
                        })
                evaluation_artifacts = {
                    "task": cfg.task,
                    "report_text": eval_summary_text,
                    "y_true": y_true.tolist(),
                    "y_pred": y_pred.tolist(),
                    "y_prob": y_prob.tolist() if y_prob is not None else None,
                    "feature_rows": feature_rows,
                }
            else:
                from eval.metrics import (
                    compute_confusion_matrix,
                    compute_roc_curves,
                    _get_preds_and_probs,
                )
                cm_fig, eval_summary_text = compute_confusion_matrix(
                    final_model, val_loader, classes, cfg.modality, cfg.task,
                    is_sklearn=sklearn_model,
                    X_val=extra_data.get("X_val"), y_val=extra_data.get("y_val"),
                    device=runtime_device)
                if cfg.task in ("classification", "multi-label"):
                    roc_fig = compute_roc_curves(
                        final_model, val_loader, classes, cfg.modality, cfg.task,
                        is_sklearn=sklearn_model,
                        X_val=extra_data.get("X_val"), y_val=extra_data.get("y_val"),
                        device=runtime_device)
                if cfg.task == "classification":
                    y_true, y_pred, y_prob = _get_preds_and_probs(
                        final_model, val_loader, classes, sklearn_model,
                        extra_data.get("X_val"), extra_data.get("y_val"), runtime_device
                    )
                    metrics_payload = classification_metric_summary(
                        y_true, y_pred, y_prob, n_classes=len(classes)
                    )
                    evaluation_artifacts = {
                        "task": cfg.task,
                        "report_text": eval_summary_text,
                        "y_true": y_true.tolist(),
                        "y_pred": y_pred.tolist(),
                        "y_prob": y_prob.tolist() if y_prob is not None else None,
                    }
                    if cfg.modality == "tabular":
                        feature_order = prep.get("feature_order") or prep.get("feature_columns") or []
                        X_val_export = extra_data.get("X_val")
                        if X_val_export is not None and feature_order:
                            row_limit = min(len(X_val_export), 2000)
                            evaluation_artifacts["feature_rows"] = [
                                {
                                    **{
                                        feature_order[idx]: float(row[idx])
                                        for idx in range(min(len(feature_order), len(row)))
                                    },
                                    "y_true": int(y_true[row_idx]),
                                    "y_pred": int(y_pred[row_idx]),
                                    "is_error": bool(y_true[row_idx] != y_pred[row_idx]),
                                }
                                for row_idx, row in enumerate(X_val_export[:row_limit])
                            ]
            log.log("  ✓ Metrics computed")
        except Exception as e:
            log.log(f"  Metrics error: {e}")
        yield Update()

        if cfg.modality in ("image", "audio", "video") and not sklearn_model:
            try:
                from eval.gradcam import compute_gradcam
                gcam_fig = compute_gradcam(final_model, val_loader, cfg.modality,
                                           cfg.model_name, prep, n_samples=4)
                if gcam_fig:
                    log.log("  ✓ Grad-CAM generated")
            except Exception as e:
                log.log(f"  Grad-CAM skipped: {e}")
            yield Update()

        if cfg.modality == "tabular":
            try:
                from eval.shap_explain import compute_shap
                shap_fig = compute_shap(final_model, extra_data, cfg.modality, prep)
                if shap_fig:
                    log.log("  ✓ SHAP computed")
            except Exception as e:
                log.log(f"  SHAP skipped: {e}")
            yield Update()

        if cfg.modality == "text" and not sklearn_model:
            try:
                from eval.shap_explain import compute_text_shap
                text_shap_html = compute_text_shap(
                    final_model, val_loader, classes, prep, n_samples=3)
                if text_shap_html:
                    log.log("  ✓ Token importance computed")
            except Exception as e:
                log.log(f"  Token importance skipped: {e}")
            yield Update()

        try:
            if cfg.modality == "graph":
                raise ValueError("Graph misclassification review is not yet available in the generic sample-review panel.")
            from eval.misclassified import find_misclassified
            mis_img_fig, mis_df_val = find_misclassified(
                final_model, val_loader, classes, cfg.modality, prep,
                task=cfg.task,
                val_samples=extra_data.get("val_samples"),
                val_texts=extra_data.get("val_texts"),
                X_val=extra_data.get("X_val"), y_val=extra_data.get("y_val"),
                is_sklearn=sklearn_model,
            )
            log.log("  ✓ Misclassified samples collected")
        except Exception as e:
            log.log(f"  Misclassified skipped: {e}")
        yield Update()

    # Regression evaluation
    if cfg.task == "regression" and cfg.model_name != "Autoencoder":
        log.set_phase("evaluating", eta_text="ETA: evaluating")
        ctx.progress(0.88, desc="Evaluating…")
        log.log("Running regression evaluation…")
        yield Update()

        try:
            import numpy as np
            import torch
            from eval.regression_metrics import (
                compute_regression_metrics,
                residual_plot,
                format_regression_report,
            )

            if sklearn_model:
                X_val = extra_data.get("X_val")
                y_val = extra_data.get("y_val")
                if X_val is None or y_val is None:
                    raise ValueError("Validation data is unavailable for regression evaluation.")
                all_true = np.asarray(y_val, dtype=float).ravel().tolist()
                all_pred = np.asarray(final_model.predict(X_val), dtype=float).ravel().tolist()
            else:
                all_true, all_pred = [], []
                final_model.eval()
                with torch.no_grad():
                    for batch in val_loader:
                        inputs, labels = batch
                        logits = final_model(inputs.to(runtime_device))
                        all_true.extend(np.asarray(labels.cpu().numpy(), dtype=float).ravel().tolist())
                        all_pred.extend(np.asarray(logits.detach().cpu().numpy(), dtype=float).ravel().tolist())

            reg_metrics = compute_regression_metrics(all_true, all_pred)
            metrics_payload = reg_metrics
            eval_summary_text = format_regression_report(reg_metrics)
            reg_fig = residual_plot(all_true, all_pred)
            evaluation_artifacts = {
                "task": cfg.task,
                "report_text": eval_summary_text,
                "y_true": all_true,
                "y_pred": all_pred,
            }
            if cfg.modality in ("tabular", "timeseries"):
                feature_rows = []
                feature_names = prep.get("feature_order") or prep.get("feature_columns") or []
                if cfg.modality == "tabular":
                    X_val_export = extra_data.get("X_val")
                    if X_val_export is not None and feature_names:
                        row_limit = min(len(X_val_export), 2000)
                        for idx, row in enumerate(X_val_export[:row_limit]):
                            feature_rows.append({
                                **{
                                    feature_names[col_idx]: float(row[col_idx])
                                    for col_idx in range(min(len(feature_names), len(row)))
                                },
                                "y_true": float(all_true[idx]),
                                "y_pred": float(all_pred[idx]),
                                "abs_error": float(abs(all_pred[idx] - all_true[idx])),
                            })
                elif cfg.modality == "timeseries":
                    X_val_export = extra_data.get("X_val")
                    if X_val_export is not None:
                        row_limit = min(len(X_val_export), 1000)
                        for idx, row in enumerate(X_val_export[:row_limit]):
                            flat = np.asarray(row).reshape(-1)
                            row_dict = {
                                f"window_feature_{col_idx}": float(flat[col_idx])
                                for col_idx in range(min(len(flat), 24))
                            }
                            row_dict.update({
                                "y_true": float(all_true[idx]),
                                "y_pred": float(all_pred[idx]),
                                "abs_error": float(abs(all_pred[idx] - all_true[idx])),
                            })
                            feature_rows.append(row_dict)
                if feature_rows:
                    evaluation_artifacts["feature_rows"] = feature_rows
            log.log("  ✓ Regression metrics (R², RMSE, MAE)")
        except Exception as e:
            log.log(f"  Regression metrics skipped: {e}")
        yield Update()

    # Anomaly detection evaluation (Autoencoder)
    if cfg.model_name == "Autoencoder" or cfg.task == "anomaly":
        log.set_phase("computing anomaly scores", eta_text="ETA: computing anomaly scores")
        ctx.progress(0.88, desc="Computing anomaly scores…")
        try:
            from eval.anomaly_detection import (
                compute_reconstruction_errors, find_anomalies,
                reconstruction_error_plot)
            errors       = compute_reconstruction_errors(final_model, val_loader, runtime_device)
            anom_result  = find_anomalies(errors)
            anomaly_fig  = reconstruction_error_plot(errors)
            n_anom       = anom_result["n_anomalies"]
            ratio        = anom_result["anomaly_ratio"] * 100
            thr          = anom_result["threshold"]
            eval_summary_text += (
                f"\n\n**Anomaly Detection**\n"
                f"Threshold: {thr:.4f}  |  "
                f"Anomalies: {n_anom} ({ratio:.1f}%)"
            )
            log.log(f"  ✓ Anomaly detection: {n_anom} anomalies ({ratio:.1f}%)")
        except Exception as e:
            log.log(f"  Anomaly detection skipped: {e}")
        yield Update()

    ctx.clustering_result = clustering_result
    ctx.evaluation_artifacts = evaluation_artifacts
    r = ctx.result
    r.metrics, r.eval_summary = metrics_payload, eval_summary_text
    r.arch_fig, r.tsne_fig = arch_fig, tsne_fig
    r.cm_fig, r.roc_fig, r.gcam_fig, r.shap_fig = cm_fig, roc_fig, gcam_fig, shap_fig
    r.mis_img_fig, r.mis_df = mis_img_fig, mis_df_val
    r.reg_fig, r.anomaly_fig, r.text_shap_html = reg_fig, anomaly_fig, text_shap_html
    r.sklearn_model = sklearn_model

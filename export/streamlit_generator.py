"""
Generate a standalone Streamlit dashboard from a trained model bundle.

The generated ``dashboard.py`` is self-contained and reads metadata written into
the bundle so it can render richer task-aware KPI pages, precision-recall and
calibration views, error slices, and comparison summaries without requiring the
NoCode-DL app to be running.
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any


def _safe_float(value: Any) -> float | None:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except Exception:
        return None


def _load_json_if_exists(path: Path) -> dict | list | None:
    if not path.is_file():
        return None
    try:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:
        return None


def _parse_classification_report_rows(summary: str) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    if not summary:
        return rows
    for raw_line in str(summary).splitlines():
        line = raw_line.strip()
        if (
            not line
            or "precision" in line
            or line.startswith("accuracy")
            or line.startswith("macro avg")
            or line.startswith("weighted avg")
        ):
            continue
        match = re.match(r"(.+?)\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s+(\d+)$", line)
        if not match:
            continue
        label, precision, recall, f1_score, support = match.groups()
        rows.append({
            "label": label.strip(),
            "precision": precision,
            "recall": recall,
            "f1_score": f1_score,
            "support": support,
        })
    return rows


def _benchmark_summary(
    bundle_path: str,
    modality: str,
    task: str,
    model_name: str,
    metrics: dict | None,
) -> dict[str, Any]:
    metrics = metrics or {}
    metric_key = "accuracy" if task in ("classification", "multi-label") else "r2" if task == "regression" else None
    lower_is_better = task in ("regression", "anomaly")
    current_value = _safe_float((metrics or {}).get(metric_key)) if metric_key else None
    peer_rows: list[dict[str, Any]] = []

    try:
        from training.run_comparison import load_history
        all_runs = load_history()
    except Exception:
        all_runs = []

    for run in all_runs:
        if run.get("bundle_path") == bundle_path:
            continue
        if run.get("modality") != modality or run.get("task") != task:
            continue
        peer_value = _safe_float((run.get("metrics") or {}).get(metric_key)) if metric_key else None
        if peer_value is None:
            continue
        peer_rows.append({
            "model_name": run.get("model_name", "Unknown"),
            "training_mode": run.get("training_mode", "unknown"),
            "timestamp": run.get("timestamp", ""),
            "metric": peer_value,
        })

    if metric_key and current_value is not None:
        peer_rows.append({
            "model_name": model_name,
            "training_mode": "current",
            "timestamp": "Current bundle",
            "metric": current_value,
            "is_current": True,
        })

    if metric_key and peer_rows:
        peer_rows.sort(key=lambda row: row["metric"], reverse=not lower_is_better)
        current_rank = None
        for idx, row in enumerate(peer_rows, start=1):
            if row.get("is_current"):
                current_rank = idx
                break
        winner = peer_rows[0]
    else:
        current_rank = None
        winner = None

    return {
        "metric_key": metric_key,
        "lower_is_better": lower_is_better,
        "current_value": current_value,
        "current_rank": current_rank,
        "peer_rows": peer_rows[:8],
        "winner": winner,
        "note": (
            "Ranking is based on saved run history for the same modality and task. "
            "Statistical significance requires exported fold-level CV scores."
            if peer_rows else
            "No comparable runs were found in local history for this modality and task."
        ),
    }


def generate_streamlit_dashboard(
    bundle_path: str,
    modality: str,
    model_name: str,
    task: str,
    training_mode: str,
    history: list[dict] | None = None,
    metrics: dict | None = None,
    classes: list[str] | None = None,
    eval_summary: str | None = None,
    evaluation_artifacts: dict | None = None,
) -> str:
    """Generate a Streamlit dashboard script inside the model bundle."""
    bundle_path = os.path.abspath(bundle_path)
    history = history or []
    metrics = metrics or {}
    classes = classes or []

    artifacts = evaluation_artifacts or _load_json_if_exists(Path(bundle_path) / "evaluation_artifacts.json") or {}
    if eval_summary and not artifacts.get("report_text"):
        artifacts["report_text"] = eval_summary
    if "report_rows" not in artifacts:
        artifacts["report_rows"] = _parse_classification_report_rows(
            artifacts.get("report_text") or eval_summary or ""
        )

    meta = {
        "modality": modality,
        "model_name": model_name,
        "task": task,
        "training_mode": training_mode,
        "history": history,
        "metrics": metrics,
        "classes": classes,
        "eval_summary": eval_summary or artifacts.get("report_text", ""),
        "evaluation_artifacts": artifacts,
        "benchmark_summary": _benchmark_summary(bundle_path, modality, task, model_name, metrics),
    }

    meta_path = os.path.join(bundle_path, "dashboard_meta.json")
    with open(meta_path, "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2, default=str)

    dashboard_code = _build_dashboard_code()
    out_path = os.path.join(bundle_path, "dashboard.py")
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write(dashboard_code)

    reqs_path = os.path.join(bundle_path, "requirements_dashboard.txt")
    with open(reqs_path, "w", encoding="utf-8") as fh:
        fh.write("streamlit>=1.30\npandas\nmatplotlib\nnumpy\n")

    return out_path


def _build_dashboard_code() -> str:
    return '''#!/usr/bin/env python3
"""
NoCode-DL — Streamlit Performance Dashboard
===========================================
Auto-generated. Run with:  streamlit run dashboard.py
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st


BUNDLE_DIR = Path(__file__).resolve().parent


def _load_json(name):
    path = BUNDLE_DIR / name
    if path.is_file():
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    return {}


meta = _load_json("dashboard_meta.json")
labels = _load_json("labels.json")
prep = _load_json("preprocessing.json")

modality = meta.get("modality", "unknown")
model_name = meta.get("model_name", "unknown")
task = meta.get("task", "unknown")
training_mode = meta.get("training_mode", "unknown")
history = meta.get("history", [])
metrics = meta.get("metrics", {})
classes = meta.get("classes", list(labels.values()) if labels else [])
evaluation = meta.get("evaluation_artifacts", {}) or {}
benchmark = meta.get("benchmark_summary", {}) or {}
eval_summary = meta.get("eval_summary", "") or evaluation.get("report_text", "")


def _fmt_metric(value, pct=False):
    try:
        numeric = float(value)
    except Exception:
        return "—"
    if pct:
        return f"{numeric * 100:.1f}%"
    return f"{numeric:.4f}"


def _metric_lookup(task_name):
    if task_name in ("classification", "multi-label"):
        return [
            ("Accuracy", metrics.get("accuracy"), True),
            ("Macro F1", metrics.get("f1_macro"), True),
            ("AUC", metrics.get("auc_ovr"), False),
            ("MCC", metrics.get("mcc"), False),
        ]
    if task_name == "regression":
        return [
            ("R²", metrics.get("r2"), False),
            ("RMSE", metrics.get("rmse"), False),
            ("MAE", metrics.get("mae"), False),
            ("Epochs", len(history) if history else None, False),
        ]
    if task_name == "clustering":
        return [
            ("Silhouette", metrics.get("silhouette"), False),
            ("Clusters", metrics.get("n_clusters"), False),
            ("Epochs", len(history) if history else None, False),
            ("Mode", training_mode, False),
        ]
    return [
        ("Primary metric", metrics.get("mae") or metrics.get("accuracy"), False),
        ("Epochs", len(history) if history else None, False),
        ("Modality", modality, False),
        ("Mode", training_mode, False),
    ]


def _normalise_rows(rows):
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


def _plot_training_curves():
    if not history:
        st.info("No training history was exported with this bundle.")
        return
    df = pd.DataFrame(history)
    cols = st.columns(2)
    with cols[0]:
        fig, ax = plt.subplots(figsize=(6, 3.5), dpi=130)
        if "train_loss" in df:
            ax.plot(df["epoch"], df["train_loss"], label="Train", marker="o", markersize=3)
        if "val_loss" in df:
            ax.plot(df["epoch"], df["val_loss"], label="Validation", marker="o", markersize=3)
        ax.set_title("Loss")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.legend(fontsize=8)
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
    with cols[1]:
        fig, ax = plt.subplots(figsize=(6, 3.5), dpi=130)
        if "val_acc" in df:
            ax.plot(df["epoch"], df["val_acc"], color="#0f766e", marker="o", markersize=3)
        ax.set_title("Validation metric")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Metric")
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)


def _binary_pr_curve(y_true_binary, scores):
    order = np.argsort(-scores)
    y = y_true_binary[order]
    tp = np.cumsum(y == 1)
    fp = np.cumsum(y == 0)
    precision = tp / np.maximum(tp + fp, 1)
    recall = tp / np.maximum(tp[-1], 1)
    return recall, precision


def _calibration_bins(y_true_binary, scores, bins=10):
    edges = np.linspace(0.0, 1.0, bins + 1)
    bin_ids = np.digitize(scores, edges[1:-1], right=True)
    records = []
    for idx in range(bins):
        mask = bin_ids == idx
        if not np.any(mask):
            continue
        records.append({
            "confidence": float(np.mean(scores[mask])),
            "accuracy": float(np.mean(y_true_binary[mask])),
            "count": int(mask.sum()),
        })
    return pd.DataFrame(records)


def _classification_dfs():
    y_true = np.array(evaluation.get("y_true") or [])
    y_pred = np.array(evaluation.get("y_pred") or [])
    y_prob = evaluation.get("y_prob")
    y_prob = np.array(y_prob) if y_prob is not None else None
    report_rows = _normalise_rows(evaluation.get("report_rows") or [])
    feature_rows = _normalise_rows(evaluation.get("feature_rows") or [])
    return y_true, y_pred, y_prob, report_rows, feature_rows


def _plot_confusion_from_arrays(y_true, y_pred):
    if y_true.size == 0 or y_pred.size == 0:
        st.info("Confusion analysis is not available for this bundle.")
        return
    n_classes = int(max(y_true.max(initial=0), y_pred.max(initial=0)) + 1)
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for truth, pred in zip(y_true, y_pred):
        cm[int(truth), int(pred)] += 1
    display_labels = classes[:n_classes] if classes else [str(i) for i in range(n_classes)]
    fig, ax = plt.subplots(figsize=(6.2, 5.4), dpi=130)
    im = ax.imshow(cm, cmap="Blues")
    fig.colorbar(im, ax=ax)
    ax.set_xticks(range(n_classes))
    ax.set_yticks(range(n_classes))
    ax.set_xticklabels(display_labels, rotation=45, ha="right")
    ax.set_yticklabels(display_labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion matrix")
    thresh = cm.max() / 2 if cm.size else 0
    for i in range(n_classes):
        for j in range(n_classes):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black", fontsize=8)
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


def _plot_precision_recall(y_true, y_prob):
    if y_true.size == 0 or y_prob is None or y_prob.size == 0:
        st.info("Precision-recall curves need exported prediction confidences. This bundle only includes aggregate metrics.")
        return
    n_classes = y_prob.shape[1] if y_prob.ndim > 1 else 2
    fig, ax = plt.subplots(figsize=(6.4, 4.6), dpi=130)
    if n_classes == 2 and y_prob.ndim > 1:
        recall, precision = _binary_pr_curve((y_true == 1).astype(int), y_prob[:, 1])
        ax.plot(recall, precision, label=classes[1] if len(classes) > 1 else "Positive", linewidth=2)
    else:
        for idx in range(min(n_classes, len(classes) or n_classes)):
            binary = (y_true == idx).astype(int)
            recall, precision = _binary_pr_curve(binary, y_prob[:, idx])
            label = classes[idx] if idx < len(classes) else f"Class {idx}"
            ax.plot(recall, precision, linewidth=1.8, label=label)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-recall")
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


def _plot_calibration(y_true, y_prob):
    if y_true.size == 0 or y_prob is None or y_prob.size == 0:
        st.info("Calibration needs exported prediction confidences. This bundle only includes aggregate metrics.")
        return
    fig, ax = plt.subplots(figsize=(6.0, 4.4), dpi=130)
    if y_prob.ndim > 1 and y_prob.shape[1] > 1:
        scores = y_prob[:, 1]
        binary = (y_true == 1).astype(int)
        bins = _calibration_bins(binary, scores)
        label = classes[1] if len(classes) > 1 else "Positive"
    else:
        scores = y_prob.reshape(-1)
        binary = y_true.astype(int)
        bins = _calibration_bins(binary, scores)
        label = "Positive"
    if bins.empty:
        st.info("Calibration plot could not be computed from the exported scores.")
        return
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1)
    ax.plot(bins["confidence"], bins["accuracy"], marker="o", linewidth=2, label=label)
    ax.set_xlabel("Predicted confidence")
    ax.set_ylabel("Observed accuracy")
    ax.set_title("Calibration")
    ax.legend(fontsize=8)
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


def _classification_error_slices(y_true, y_pred, feature_rows):
    st.subheader("Error slices")
    if y_true.size == 0 or y_pred.size == 0:
        st.info("Per-class error slices are unavailable for this bundle.")
        return
    labels_local = classes if classes else [str(i) for i in sorted(set(y_true.tolist()))]
    error_rows = []
    for idx, label in enumerate(labels_local):
        mask = y_true == idx
        if not np.any(mask):
            continue
        error_rows.append({
            "Class": label,
            "Support": int(mask.sum()),
            "Error rate": round(float(np.mean(y_pred[mask] != y_true[mask])) * 100, 2),
        })
    if error_rows:
        st.dataframe(pd.DataFrame(error_rows), use_container_width=True, hide_index=True)

    if feature_rows.empty:
        st.caption("Feature-level slices were not exported for this run.")
        return
    feature_df = feature_rows.copy()
    if "is_error" not in feature_df.columns:
        feature_df["is_error"] = feature_df["y_true"] != feature_df["y_pred"]
    numeric_cols = [
        col for col in feature_df.columns
        if col not in {"y_true", "y_pred", "is_error"} and pd.api.types.is_numeric_dtype(feature_df[col])
    ][:3]
    if not numeric_cols:
        st.caption("No numeric feature columns were available for slice analysis.")
        return
    slice_rows = []
    for col in numeric_cols:
        median = float(feature_df[col].median())
        for bucket_name, mask in (
            ("Low", feature_df[col] <= median),
            ("High", feature_df[col] > median),
        ):
            if mask.sum() == 0:
                continue
            slice_rows.append({
                "Feature": col,
                "Slice": bucket_name,
                "Rows": int(mask.sum()),
                "Error rate": round(float(feature_df.loc[mask, "is_error"].mean()) * 100, 2),
            })
    if slice_rows:
        st.dataframe(pd.DataFrame(slice_rows), use_container_width=True, hide_index=True)


def _regression_views():
    y_true = np.array(evaluation.get("y_true") or [], dtype=float)
    y_pred = np.array(evaluation.get("y_pred") or [], dtype=float)
    feature_rows = _normalise_rows(evaluation.get("feature_rows") or [])
    if y_true.size == 0 or y_pred.size == 0:
        st.info("Residual analysis is unavailable for this bundle.")
        return
    residuals = y_pred - y_true
    cols = st.columns(2)
    with cols[0]:
        fig, ax = plt.subplots(figsize=(6, 4), dpi=130)
        ax.scatter(y_true, residuals, alpha=0.6, s=18, color="#0f766e")
        ax.axhline(0.0, linestyle="--", color="gray", linewidth=1)
        ax.set_xlabel("True value")
        ax.set_ylabel("Residual")
        ax.set_title("Residual scatter")
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
    with cols[1]:
        fig, ax = plt.subplots(figsize=(6, 4), dpi=130)
        ax.hist(np.abs(residuals), bins=20, color="#1d4ed8", alpha=0.85)
        ax.set_xlabel("Absolute error")
        ax.set_ylabel("Count")
        ax.set_title("Absolute error distribution")
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    if feature_rows.empty:
        st.caption("Feature-level regression slices were not exported for this run.")
        return
    feature_df = feature_rows.copy()
    if "abs_error" not in feature_df.columns:
        feature_df["abs_error"] = np.abs(feature_df["y_pred"] - feature_df["y_true"])
    numeric_cols = [
        col for col in feature_df.columns
        if col not in {"y_true", "y_pred", "abs_error"} and pd.api.types.is_numeric_dtype(feature_df[col])
    ][:3]
    rows = []
    for col in numeric_cols:
        q1 = float(feature_df[col].quantile(0.25))
        q3 = float(feature_df[col].quantile(0.75))
        for label, mask in (
            ("Low quartile", feature_df[col] <= q1),
            ("High quartile", feature_df[col] >= q3),
        ):
            if mask.sum() == 0:
                continue
            rows.append({
                "Feature": col,
                "Slice": label,
                "Rows": int(mask.sum()),
                "Mean abs error": round(float(feature_df.loc[mask, "abs_error"].mean()), 4),
            })
    if rows:
        st.subheader("Error slices by feature")
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def _significance_section():
    st.subheader("Benchmark & significance summary")
    peer_rows = benchmark.get("peer_rows") or []
    if not peer_rows:
        st.info(benchmark.get("note", "No benchmark data is available."))
        return
    metric_key = benchmark.get("metric_key") or "metric"
    current = benchmark.get("current_value")
    winner = benchmark.get("winner") or {}
    rank = benchmark.get("current_rank")
    top = st.columns(3)
    with top[0]:
        st.metric("Ranking metric", metric_key.replace("_", " ").title())
    with top[1]:
        st.metric("Current bundle rank", f"#{rank}" if rank else "—")
    with top[2]:
        st.metric("Best comparable run", winner.get("model_name", "—"))
    peer_df = pd.DataFrame(peer_rows)
    if not peer_df.empty:
        peer_df["metric_display"] = peer_df["metric"].map(lambda v: f"{v * 100:.2f}%" if metric_key in {"accuracy", "f1_macro"} else f"{v:.4f}")
        st.dataframe(peer_df.rename(columns={
            "model_name": "Model",
            "training_mode": "Mode",
            "timestamp": "Timestamp",
            "metric_display": "Metric",
        })[["Model", "Mode", "Timestamp", "Metric"]], use_container_width=True, hide_index=True)
        fig, ax = plt.subplots(figsize=(7, 4), dpi=130)
        ax.bar(peer_df["model_name"], peer_df["metric"], color=["#0f766e" if row.get("is_current") else "#93c5fd" for _, row in peer_df.iterrows()])
        ax.set_title("Comparable runs")
        ax.set_ylabel(metric_key.replace("_", " ").title())
        ax.tick_params(axis="x", rotation=35)
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
    st.caption(benchmark.get("note", ""))


st.set_page_config(page_title=f"Dashboard — {model_name}", page_icon="📊", layout="wide")

st.markdown("""
<style>
    .report-shell {
        padding: 0.5rem 0 1rem 0;
    }
    .hero {
        background: linear-gradient(135deg, #0f172a 0%, #1f4d4d 100%);
        border-radius: 24px;
        padding: 1.4rem 1.6rem;
        color: white;
        margin-bottom: 1rem;
        box-shadow: 0 18px 40px rgba(15, 23, 42, 0.15);
    }
    .hero h1 {
        margin: 0;
        font-size: 2rem;
    }
    .hero p {
        margin: 0.4rem 0 0 0;
        color: rgba(255,255,255,0.82);
    }
    .kpi-card {
        background: linear-gradient(180deg, #ffffff 0%, #f8fafc 100%);
        border: 1px solid #dbe4ea;
        border-radius: 18px;
        padding: 1rem 1rem 0.9rem 1rem;
        min-height: 112px;
        box-shadow: 0 10px 24px rgba(15, 23, 42, 0.06);
    }
    .kpi-label {
        font-size: 0.78rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #64748b;
        margin-bottom: 0.35rem;
        font-weight: 700;
    }
    .kpi-value {
        font-size: 1.65rem;
        font-weight: 800;
        color: #0f172a;
    }
    .kpi-caption {
        color: #64748b;
        font-size: 0.86rem;
        margin-top: 0.25rem;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="report-shell">', unsafe_allow_html=True)
st.markdown(
    f"""
    <div class="hero">
        <h1>{model_name} performance dashboard</h1>
        <p>{modality.title()} · {task.replace('_', ' ').title()} · {training_mode.replace('_', ' ').title()}</p>
    </div>
    """,
    unsafe_allow_html=True,
)

kpi_cols = st.columns(4)
for idx, (label, value, pct) in enumerate(_metric_lookup(task)):
    with kpi_cols[idx]:
        display = value if isinstance(value, str) else _fmt_metric(value, pct=pct)
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-label">{label}</div>
                <div class="kpi-value">{display}</div>
                <div class="kpi-caption">Task-aware headline KPI</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

overview, curves, diagnostics, comparisons = st.tabs([
    "Overview",
    "Training",
    "Diagnostics",
    "Comparisons",
])

with overview:
    left, right = st.columns([1.25, 1.0])
    with left:
        st.subheader("Evaluation snapshot")
        if metrics:
            metrics_df = pd.DataFrame(
                [{"Metric": key.replace("_", " ").title(), "Value": _fmt_metric(value, pct=key in {"accuracy", "precision_macro", "recall_macro", "f1_macro", "balanced_accuracy"})}
                 for key, value in metrics.items() if value is not None]
            )
            st.dataframe(metrics_df, use_container_width=True, hide_index=True)
        else:
            st.info("No aggregate metrics were exported with this bundle.")
        if eval_summary:
            with st.expander("Narrative evaluation summary", expanded=False):
                st.code(eval_summary, language="text")
    with right:
        st.subheader("Bundle details")
        details = pd.DataFrame([
            {"Field": "Model", "Value": model_name},
            {"Field": "Task", "Value": task},
            {"Field": "Modality", "Value": modality},
            {"Field": "Training mode", "Value": training_mode},
            {"Field": "Classes", "Value": ", ".join(classes) if classes else "—"},
        ])
        st.dataframe(details, use_container_width=True, hide_index=True)
        if prep:
            st.subheader("Preprocessing")
            prep_rows = [{"Setting": key, "Value": str(value)} for key, value in prep.items() if key in {
                "image_size", "sample_rate", "window_size", "lag_steps", "rolling_window",
                "forecast_horizon", "subset_percent", "scaling_strategy", "normalization",
            }]
            if prep_rows:
                st.dataframe(pd.DataFrame(prep_rows), use_container_width=True, hide_index=True)

with curves:
    _plot_training_curves()

with diagnostics:
    if task in ("classification", "multi-label"):
        y_true, y_pred, y_prob, report_rows, feature_rows = _classification_dfs()
        top = st.columns(2)
        with top[0]:
            st.subheader("Per-class quality")
            if not report_rows.empty:
                report_chart = report_rows.copy()
                for col in ("precision", "recall", "f1_score"):
                    report_chart[col] = report_chart[col].astype(float)
                chart_df = report_chart.melt(
                    id_vars=["label"],
                    value_vars=["precision", "recall", "f1_score"],
                    var_name="Metric",
                    value_name="Score",
                )
                fig, ax = plt.subplots(figsize=(6.4, 4.6), dpi=130)
                for metric_name, group in chart_df.groupby("Metric"):
                    ax.plot(group["label"], group["Score"], marker="o", label=metric_name.replace("_", " ").title())
                ax.set_ylim(0, 1.02)
                ax.set_title("Per-class precision / recall / F1")
                ax.tick_params(axis="x", rotation=35)
                ax.legend(fontsize=8)
                fig.tight_layout()
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)
                st.dataframe(report_rows.rename(columns={
                    "label": "Class",
                    "f1_score": "F1",
                    "support": "Support",
                }), use_container_width=True, hide_index=True)
            else:
                st.info("Per-class classification rows were not exported for this bundle.")
        with top[1]:
            _plot_confusion_from_arrays(y_true, y_pred)

        second = st.columns(2)
        with second[0]:
            st.subheader("Precision-recall")
            _plot_precision_recall(y_true, y_prob)
        with second[1]:
            st.subheader("Calibration")
            _plot_calibration(y_true, y_prob)

        _classification_error_slices(y_true, y_pred, feature_rows)

    elif task == "regression":
        _regression_views()
    elif task == "clustering":
        st.info("Clustering bundles currently export aggregate metrics and cluster artifacts, but not richer diagnostic arrays.")
    else:
        st.info("Task-specific diagnostics are limited for this task in the current export format.")

with comparisons:
    _significance_section()

st.markdown("</div>", unsafe_allow_html=True)
'''

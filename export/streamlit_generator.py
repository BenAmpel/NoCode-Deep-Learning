"""
Generate a standalone Streamlit dashboard from a trained model bundle.

The generated ``dashboard.py`` is self-contained: it reads the bundle's
``labels.json``, ``preprocessing.json``, and ``run_history.json`` to
reconstruct KPI cards, training curves, metrics tables, and model info
without requiring the NoCode-DL app to be running.

Usage (after generation)::

    pip install streamlit pandas matplotlib
    streamlit run <bundle_path>/dashboard.py
"""
from __future__ import annotations

import json
import os
from pathlib import Path


def generate_streamlit_dashboard(
    bundle_path: str,
    modality: str,
    model_name: str,
    task: str,
    training_mode: str,
    history: list[dict] | None = None,
    metrics: dict | None = None,
    classes: list[str] | None = None,
) -> str:
    """Generate a Streamlit dashboard script inside the model bundle.

    Parameters
    ----------
    bundle_path : str
        Path to the saved model bundle directory.
    modality : str
        Data modality (image, text, tabular, timeseries, audio, video).
    model_name : str
        Name of the model architecture.
    task : str
        Task type (classification, regression, clustering, anomaly).
    training_mode : str
        Training mode (fine-tune, from_scratch).
    history : list[dict], optional
        Training epoch history (epoch, train_loss, val_loss, val_acc).
    metrics : dict, optional
        Evaluation metrics payload (accuracy, f1_macro, mae, etc.).
    classes : list[str], optional
        Class/label names.

    Returns
    -------
    str
        Absolute path to the generated ``dashboard.py`` file.
    """
    bundle_path = os.path.abspath(bundle_path)
    history = history or []
    metrics = metrics or {}
    classes = classes or []

    # Persist history and metrics into the bundle so the dashboard can read them
    meta_path = os.path.join(bundle_path, "dashboard_meta.json")
    with open(meta_path, "w", encoding="utf-8") as fh:
        json.dump({
            "modality": modality,
            "model_name": model_name,
            "task": task,
            "training_mode": training_mode,
            "history": history,
            "metrics": metrics,
            "classes": classes,
        }, fh, indent=2, default=str)

    dashboard_code = _build_dashboard_code()
    out_path = os.path.join(bundle_path, "dashboard.py")
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write(dashboard_code)

    reqs_path = os.path.join(bundle_path, "requirements_dashboard.txt")
    with open(reqs_path, "w", encoding="utf-8") as fh:
        fh.write("streamlit>=1.30\npandas\nmatplotlib\n")

    return out_path


def _build_dashboard_code() -> str:
    return '''#!/usr/bin/env python3
"""
NoCode-DL — Streamlit Executive Dashboard
==========================================
Auto-generated. Run with:  streamlit run dashboard.py
"""
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

# ── Load bundle data ─────────────────────────────────────────────────────────

BUNDLE_DIR = Path(__file__).resolve().parent

def _load_json(name):
    p = BUNDLE_DIR / name
    if p.is_file():
        with open(p, encoding="utf-8") as f:
            return json.load(f)
    return {}

meta    = _load_json("dashboard_meta.json")
labels  = _load_json("labels.json")
prep    = _load_json("preprocessing.json")

modality      = meta.get("modality", "unknown")
model_name    = meta.get("model_name", "unknown")
task          = meta.get("task", "unknown")
training_mode = meta.get("training_mode", "unknown")
history       = meta.get("history", [])
metrics       = meta.get("metrics", {})
classes       = meta.get("classes", list(labels.values()) if labels else [])

# ── Page config ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title=f"Dashboard — {model_name}",
    page_icon="📈",
    layout="wide",
)

st.markdown("""
<style>
    .kpi-card {
        background: linear-gradient(135deg, #f8faf9 0%, #eef4f0 100%);
        border: 1px solid #d7e5db;
        border-radius: 16px;
        padding: 20px;
        text-align: center;
    }
    .kpi-card .label {
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #6a857c;
        margin-bottom: 4px;
    }
    .kpi-card .value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #1f3b35;
    }
    .kpi-card .value.green { color: #16856f; }
    .kpi-card .value.yellow { color: #c4820e; }
    .kpi-card .value.red { color: #c42b1c; }
</style>
""", unsafe_allow_html=True)

# ── Header ───────────────────────────────────────────────────────────────────

st.title("📈 Executive Dashboard")
st.caption(f"Model: **{model_name}** · Task: **{task}** · Modality: **{modality}** · Mode: **{training_mode}**")
st.divider()

# ── KPI strip ────────────────────────────────────────────────────────────────

def _kpi_card(label, value, css_class=""):
    return f"""
    <div class="kpi-card">
        <div class="label">{label}</div>
        <div class="value {css_class}">{value}</div>
    </div>
    """

cols = st.columns(4)

# Card 1: Task
with cols[0]:
    st.markdown(_kpi_card("Task", f"{task.title()} · {modality}"), unsafe_allow_html=True)

# Card 2: Primary metric
with cols[1]:
    if task in ("classification", "multi-label"):
        acc = metrics.get("accuracy")
        if acc is not None:
            pct = float(acc) * 100
            css = "green" if pct >= 80 else "yellow" if pct >= 60 else "red"
            st.markdown(_kpi_card("Accuracy", f"{pct:.1f}%", css), unsafe_allow_html=True)
        elif history:
            st.markdown(_kpi_card("Accuracy", f"{history[-1].get('val_acc', 0):.1f}%"), unsafe_allow_html=True)
        else:
            st.markdown(_kpi_card("Accuracy", "—"), unsafe_allow_html=True)
    elif task == "regression":
        mae = metrics.get("mae")
        st.markdown(_kpi_card("MAE", f"{float(mae):.4f}" if mae else "—"), unsafe_allow_html=True)
    else:
        st.markdown(_kpi_card("Metric", "—"), unsafe_allow_html=True)

# Card 3: Val loss
with cols[2]:
    val_loss = f"{history[-1]['val_loss']:.4f}" if history else "—"
    st.markdown(_kpi_card("Val Loss", val_loss), unsafe_allow_html=True)

# Card 4: Epochs
with cols[3]:
    st.markdown(_kpi_card("Epochs", str(len(history)) if history else "—"), unsafe_allow_html=True)

st.divider()

# ── Main content ─────────────────────────────────────────────────────────────

left, right = st.columns([3, 2])

# ── Training curves ──────────────────────────────────────────────────────────

with right:
    st.subheader("Training Curves")
    if history:
        df_hist = pd.DataFrame(history)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3.5), dpi=120)
        ax1.plot(df_hist["epoch"], df_hist["train_loss"], label="Train", marker="o", markersize=3)
        ax1.plot(df_hist["epoch"], df_hist["val_loss"], label="Val", marker="o", markersize=3)
        ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss"); ax1.set_title("Loss")
        ax1.legend(fontsize=8)
        ax2.plot(df_hist["epoch"], df_hist["val_acc"], color="steelblue", marker="o", markersize=3)
        ax2.set_xlabel("Epoch"); ax2.set_ylabel("Metric"); ax2.set_title("Validation Metric")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
    else:
        st.info("No training history available.")

# ── Metrics detail ───────────────────────────────────────────────────────────

with left:
    st.subheader("Evaluation Metrics")
    if metrics:
        display_metrics = {}
        for k, v in metrics.items():
            if v is None:
                continue
            if k in ("accuracy", "precision_macro", "recall_macro", "f1_macro",
                     "precision_weighted", "recall_weighted", "f1_weighted",
                     "balanced_accuracy"):
                display_metrics[k.replace("_", " ").title()] = f"{float(v)*100:.2f}%"
            else:
                display_metrics[k.replace("_", " ").title()] = f"{float(v):.4f}"
        df_metrics = pd.DataFrame(
            list(display_metrics.items()), columns=["Metric", "Value"]
        )
        st.dataframe(df_metrics, use_container_width=True, hide_index=True)
    else:
        st.info("No evaluation metrics available.")

st.divider()

# ── Class info ───────────────────────────────────────────────────────────────

if classes:
    st.subheader("Classes")
    st.write(", ".join(str(c) for c in classes))

# ── Recommendations ──────────────────────────────────────────────────────────

st.subheader("Recommendations")

items = []
if history and len(history) >= 3:
    last = history[-1]
    if last["val_loss"] > last["train_loss"] * 1.3:
        gap = (last["val_loss"] - last["train_loss"]) / max(last["train_loss"], 1e-6) * 100
        items.append(
            f"**Possible overfitting** — validation loss is {gap:.0f}% higher than training loss. "
            "Consider more data, stronger regularization, or data augmentation."
        )

if task in ("classification", "multi-label"):
    acc = metrics.get("accuracy")
    if acc and float(acc) < 0.7:
        items.append(
            f"**Low accuracy ({float(acc)*100:.1f}%)** — try a larger model, more epochs, or higher learning rate."
        )
    f1 = metrics.get("f1_macro")
    if acc and f1 and abs(float(acc) - float(f1)) > 0.15:
        items.append(
            "**Accuracy/F1 gap** — class imbalance may be affecting results. Check per-class metrics."
        )
    if acc and float(acc) >= 0.9:
        items.append(
            f"**Strong performance ({float(acc)*100:.1f}%)** — verify on held-out data before deploying."
        )

if not items:
    items.append("No specific issues detected. Review the metrics above for details.")

for item in items:
    st.markdown(f"- {item}")

# ── Bundle info ──────────────────────────────────────────────────────────────

with st.expander("Bundle Details"):
    st.text(f"Bundle path: {BUNDLE_DIR}")
    st.text(f"Model: {model_name} ({training_mode})")
    has_onnx = (BUNDLE_DIR / "model.onnx").exists()
    has_pt = (BUNDLE_DIR / "model.pt").exists() or (BUNDLE_DIR / "model.pth").exists()
    has_joblib = (BUNDLE_DIR / "model.joblib").exists()
    onnx_str = "yes" if has_onnx else "no"
    pt_str = "yes" if has_pt else "no"
    sk_str = "yes" if has_joblib else "no"
    st.text(f"ONNX: {onnx_str} · PyTorch: {pt_str} · Sklearn: {sk_str}")
    if prep:
        st.json(prep)
'''



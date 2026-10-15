"""
Helpers for the "Try Your Model" inference panel.

Supports image, text, tabular, time-series and audio modalities. All
preprocessing and model execution goes through export/bundle_runtime.py,
the same file shipped inside every exported bundle. All ``predict_*``
functions return a safe error payload instead of raising, so the UI
can always display something meaningful.

Usage
-----
    from ui.inference_helpers import (
        load_bundle, predict_image, predict_text,
        predict_tabular, predict_audio,
        predictions_to_markdown, predictions_to_chart,
    )

    bundle = load_bundle("outputs/my_model_bundle")
    results = predict_image(bundle_path, image_array, top_k=3)
    print(predictions_to_markdown(results))
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level inference cache
# ---------------------------------------------------------------------------
# Models are expensive to load (disk I/O + weight initialisation).  Cache the
# loaded model/session keyed by bundle path so repeated "Try Your Model" calls
# reuse the already-loaded weights instead of hitting disk every time.
# The cache is capped at 4 entries to avoid unbounded memory growth.

_MAX_CACHE = 4
_bundle_cache: dict[str, dict] = {}  # bundle_path → load_bundle() result


def _cached_bundle(bundle_path: str) -> dict:
    """Return cached bundle metadata, loading from disk only on first access."""
    if bundle_path not in _bundle_cache:
        if len(_bundle_cache) >= _MAX_CACHE:
            _bundle_cache.pop(next(iter(_bundle_cache)))
        _bundle_cache[bundle_path] = load_bundle(bundle_path)
    return _bundle_cache[bundle_path]


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Confidence bar settings
_BAR_FULL  = "█"
_BAR_EMPTY = "░"
_BAR_WIDTH = 10

# Rank medals for the markdown table
_RANK_MEDALS = {1: "🥇", 2: "🥈", 3: "🥉"}

# Bar chart colours for top-3
_RANK_COLOURS = {1: "#FFD700", 2: "#C0C0C0", 3: "#CD7F32"}  # gold, silver, bronze
_DEFAULT_BAR_COLOUR = "#4C9BE8"


# ---------------------------------------------------------------------------
# load_bundle
# ---------------------------------------------------------------------------

def load_bundle(bundle_path: str) -> dict:
    """
    Load metadata files from a saved model bundle directory.

    Expects the bundle to contain ``labels.json`` and ``preprocessing.json``.
    Optionally contains a ``model.pt`` or ``model.onnx`` file.

    Parameters
    ----------
    bundle_path:
        Path to the model bundle directory produced by the export pipeline.

    Returns
    -------
    dict
        Keys:

        - ``labels``        (dict) — ``{str_index: label_name, ...}``
        - ``preprocessing`` (dict) — preprocessing config used at training time
        - ``model_path``    (str | None) — absolute path to ``.pt`` file if present
        - ``onnx_path``     (str | None) — absolute path to ``.onnx`` file if present
    """
    # Security: ensure the resolved path stays within the outputs/ directory
    # to prevent path traversal attacks when bundle_path comes from user input.
    _project_root = Path(__file__).resolve().parent.parent
    _outputs_root = (_project_root / "outputs").resolve()
    _resolved     = Path(bundle_path).resolve()
    if not str(_resolved).startswith(str(_outputs_root)):
        # Soft warning — allow absolute paths that exist (local desktop use)
        # but log the anomaly for visibility.
        logger.warning(
            "Bundle path '%s' is outside the expected outputs/ directory. "
            "Proceeding, but verify this path is trusted.",
            bundle_path,
        )

    bundle_path = os.path.abspath(bundle_path)

    labels_file = os.path.join(bundle_path, "labels.json")
    prep_file   = os.path.join(bundle_path, "preprocessing.json")

    with open(labels_file, "r", encoding="utf-8") as fh:
        labels: dict = json.load(fh)

    with open(prep_file, "r", encoding="utf-8") as fh:
        preprocessing: dict = json.load(fh)

    # Optional model files
    def _find(name: str) -> str | None:
        p = os.path.join(bundle_path, name)
        return p if os.path.isfile(p) else None

    model_path = _find("model.pt") or _find("model.pth") or _find("model.joblib") or _find("model.pkl")
    onnx_path  = _find("model.onnx")

    return {
        "labels":        labels,
        "preprocessing": preprocessing,
        "model_path":    model_path,
        "onnx_path":     onnx_path,
    }


# ---------------------------------------------------------------------------
# Prediction (all preprocessing lives in export/bundle_runtime.py)
# ---------------------------------------------------------------------------
# The runtime is the same file the exporter ships inside every bundle, so
# "Try Your Model" and an exported server cannot drift apart.

_runtime_cache: dict[str, Any] = {}  # abs bundle path -> bundle_runtime.Bundle


def _runtime_bundle(bundle_path: str):
    from export.bundle_runtime import Bundle

    key = os.path.abspath(bundle_path)
    if key not in _runtime_cache:
        if len(_runtime_cache) >= _MAX_CACHE:
            _runtime_cache.pop(next(iter(_runtime_cache)))
        outputs_root = (Path(__file__).resolve().parent.parent / "outputs").resolve()
        # Bundles this app exported live under outputs/; a bundle from anywhere
        # else may be someone else's, so its pickle (model.joblib) isn't loaded.
        _runtime_cache[key] = Bundle(key, trust_pickle=Path(key).resolve().is_relative_to(outputs_root))
    return _runtime_cache[key]


def _predict(bundle_path: str, raw: Any, top_k: int) -> list[dict]:
    try:
        result = _runtime_bundle(bundle_path).predict(raw, top_k=top_k)
    except Exception as exc:
        return _error_payload(exc)
    if result["task"] == "regression":
        return _scalar_prediction_payload(result["value"], label="Predicted value")
    return result["predictions"]


def predict_image(bundle_path: str, image_input: Any, top_k: int = 3) -> list[dict]:
    """Classify an image (file path, numpy array or PIL image).

    Returns up to ``top_k`` dicts sorted by confidence,
    ``[{"label": str, "confidence": float, "rank": int}, ...]``, or
    ``[{"label": "Error", "confidence": 0.0, "rank": 1, "error": str}]``.
    """
    return _predict(bundle_path, image_input, top_k)


def predict_text(bundle_path: str, text: str, top_k: int = 3) -> list[dict]:
    """Classify raw text (cleaned and tokenized as in training)."""
    return _predict(bundle_path, text, top_k)


def predict_tabular(bundle_path: str, feature_values: dict[str, Any], top_k: int = 3) -> list[dict]:
    """Predict from raw column values. Regression bundles return one ``value`` entry."""
    return _predict(bundle_path, feature_values, top_k)


def predict_timeseries(bundle_path: str, window_values: Any, top_k: int = 3) -> list[dict]:
    """Predict from recent rows (JSON string or list), oldest first."""
    return _predict(bundle_path, window_values, top_k)


def predict_audio(bundle_path: str, audio_path: str, top_k: int = 3) -> list[dict]:
    """Classify an audio file via the same mel-spectrogram pipeline as training."""
    return _predict(bundle_path, audio_path, top_k)


def explain_text_prediction(bundle_path: str, text: str) -> str:
    """Run occlusion-based token importance for a single text input.

    Returns an HTML string with tokens highlighted by importance
    (red = supports prediction, blue = opposes it).
    """
    import html as html_mod

    from export import bundle_runtime as rt

    try:
        bundle = _runtime_bundle(bundle_path)
        prep = bundle.prep
        encoded = bundle._text_inputs(text)
        full_ids, full_mask = encoded["input_ids"], encoded["attention_mask"]
        real_len = int(full_mask[0].sum())
        if real_len == 0:
            return ""

        if prep.get("vocab"):
            cleaned = rt._clean_text_like_training(text, prep.get("cleaning", {}))
            tokens = rt.simple_tokenize(cleaned)[:real_len]
            positions = list(range(real_len))
        else:
            tok = bundle._tokenizer
            special = set(tok.all_special_ids)
            positions = [i for i in range(real_len) if int(full_ids[0, i]) not in special]
            tokens = [tok.decode([int(full_ids[0, i])]).strip() for i in positions]
        if not positions:
            return ""

        def class_probs(ids, mask):
            out = bundle._run({"input_ids": ids, "attention_mask": mask})[: len(bundle.labels)]
            return rt._softmax(out)

        base_probs = class_probs(full_ids, full_mask)
        pred_cls = int(np.argmax(base_probs))
        pred_conf = float(base_probs[pred_cls])
        pred_label = bundle.labels.get(str(pred_cls), f"class_{pred_cls}")

        # Occlusion: drop each token in turn (pad id 0, masked out).
        importance = np.zeros(len(positions), dtype=np.float32)
        for j, pos in enumerate(positions):
            occ_ids, occ_mask = full_ids.copy(), full_mask.copy()
            occ_ids[0, pos] = 0
            occ_mask[0, pos] = 0
            importance[j] = pred_conf - float(class_probs(occ_ids, occ_mask)[pred_cls])

        abs_imp = np.abs(importance)
        max_imp = abs_imp.max() if abs_imp.max() > 0 else 1.0
        spans = []
        for j, token in enumerate(tokens):
            opacity = float(abs_imp[j] / max_imp) * 0.7
            color = "220,38,38" if importance[j] >= 0 else "37,99,235"
            spans.append(
                f'<span style="background:rgba({color},{opacity:.2f});'
                f'padding:2px 4px;border-radius:4px;margin:1px;" '
                f'title="importance: {importance[j]:.4f}">{html_mod.escape(token)}</span> '
            )

        return (
            f'<div style="line-height:2.2;margin-top:8px;">'
            f'<div style="font-size:0.8rem;margin-bottom:6px;color:var(--color-text-muted,#6a857c);">'
            f'Predicted: <b>{html_mod.escape(str(pred_label))}</b> ({pred_conf:.0%}) · '
            f'<span style="color:rgba(220,38,38,0.8);">Red</span> = supports · '
            f'<span style="color:rgba(37,99,235,0.8);">Blue</span> = opposes</div>'
            f'<div>{"".join(spans)}</div></div>'
        )
    except Exception:
        return ""


# ---------------------------------------------------------------------------
# predictions_to_markdown
# ---------------------------------------------------------------------------

def predictions_to_markdown(predictions: list[dict]) -> str:
    """
    Render a top-k prediction list as a markdown table with ASCII confidence bars.

    Parameters
    ----------
    predictions:
        List of dicts from any ``predict_*`` function.

    Returns
    -------
    str
        Markdown table, e.g.::

            | Rank | Label | Confidence | Bar        |
            |------|-------|------------|------------|
            | 🥇   | cat   | 94.3%      | █████████░ |

    On error payload, returns a single-row table showing the error message.
    """
    # Error payload shortcut
    if predictions and predictions[0].get("label") == "Error":
        err = predictions[0].get("error", "Unknown error")
        return f"❌ **Prediction failed:** {err}"

    if predictions and "value" in predictions[0]:
        label = predictions[0].get("label", "Prediction")
        value = predictions[0].get("value")
        return f"### {label}\n\n`{value:.6f}`"

    lines = [
        "| Rank | Label | Confidence | " + ("—" * _BAR_WIDTH) + " |",
        "|:----:|:------|----------:|:" + ("-" * _BAR_WIDTH) + ":|",
    ]
    for pred in predictions:
        rank       = pred.get("rank", 1)
        label      = pred.get("label", "?")
        confidence = pred.get("confidence", 0.0)

        medal = _RANK_MEDALS.get(rank, f"#{rank}")
        pct   = confidence * 100.0
        filled = round(pct / 100.0 * _BAR_WIDTH)
        bar   = _BAR_FULL * filled + _BAR_EMPTY * (_BAR_WIDTH - filled)
        lines.append(f"| {medal} | {label} | {pct:.1f}% | {bar} |")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# predictions_to_chart
# ---------------------------------------------------------------------------

def predictions_to_chart(predictions: list[dict]):
    """
    Create a horizontal bar chart of top-k predictions.

    Bars are coloured gold / silver / bronze for the top 3 ranks, and a
    default blue for any additional predictions.

    Parameters
    ----------
    predictions:
        List of dicts from any ``predict_*`` function.

    Returns
    -------
    matplotlib.figure.Figure
        The figure object (not shown).  Pass to ``gr.Plot`` in Gradio.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Filter out error entries
    valid = [p for p in predictions if p.get("label") != "Error"]
    if not valid:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No predictions available", ha="center", va="center",
                transform=ax.transAxes, fontsize=12, color="grey")
        ax.axis("off")
        plt.close(fig)  # rendered by Gradio after return; frees pyplot's reference
        return fig

    if "value" in valid[0]:
        value = float(valid[0]["value"])
        label = valid[0].get("label", "Prediction")
        fig, ax = plt.subplots(figsize=(5.5, 2.6))
        ax.barh([label], [value], color="#115d52")
        ax.axvline(0, color="#94a3b8", linewidth=1, linestyle="--")
        ax.text(value, 0, f" {value:.4f}", va="center", ha="left" if value >= 0 else "right")
        ax.set_title("Predicted value", fontsize=12, fontweight="bold")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        fig.tight_layout()
        plt.close(fig)  # rendered by Gradio after return; frees pyplot's reference
        return fig

    labels      = [p["label"]      for p in valid]
    confidences = [p["confidence"] * 100.0 for p in valid]
    ranks       = [p.get("rank", i + 1) for i, p in enumerate(valid)]
    colours     = [_RANK_COLOURS.get(r, _DEFAULT_BAR_COLOUR) for r in ranks]

    # Plot in reverse order so rank 1 appears at the top
    labels      = labels[::-1]
    confidences = confidences[::-1]
    colours     = colours[::-1]

    fig, ax = plt.subplots(figsize=(7, max(2.5, len(valid) * 0.8)))

    bars = ax.barh(labels, confidences, color=colours, edgecolor="white", height=0.55)

    # Value annotations inside bars
    for bar, conf in zip(bars, confidences):
        x_pos = bar.get_width() - 2.5
        ha    = "right"
        colour = "white"
        if conf < 10:
            x_pos = bar.get_width() + 1.0
            ha    = "left"
            colour = "#333333"
        ax.text(
            x_pos,
            bar.get_y() + bar.get_height() / 2,
            f"{conf:.1f}%",
            va="center",
            ha=ha,
            fontsize=10,
            fontweight="bold",
            color=colour,
        )

    ax.set_xlim(0, 105)
    ax.set_xlabel("Confidence (%)", fontsize=10)
    ax.set_title("Top Predictions", fontsize=12, fontweight="bold", pad=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="y", labelsize=10)

    fig.tight_layout()
    plt.close(fig)  # rendered by Gradio after return; frees pyplot's reference
    return fig


# ---------------------------------------------------------------------------
# Payload helpers
# ---------------------------------------------------------------------------

def _error_payload(exc: Exception) -> list[dict]:
    """Return a safe error payload so the UI never crashes."""
    return [{
        "label":      "Error",
        "confidence": 0.0,
        "rank":       1,
        "error":      f"{type(exc).__name__}: {exc}",
    }]


def _scalar_prediction_payload(value: float, label: str = "Prediction") -> list[dict]:
    return [{
        "label": label,
        "confidence": 1.0,
        "rank": 1,
        "value": float(value),
    }]

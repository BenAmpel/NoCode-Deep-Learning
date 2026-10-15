"""Constants and presentation helpers shared by the layout and the callbacks.

Moved out of app.py unchanged; app.py re-exports the names its callers use.
"""
from __future__ import annotations
import base64, html, io, logging, os, re
from functools import lru_cache
from pathlib import Path
from typing import Any

import gradio as gr

from models.registry import get_models, get_compatible_tasks
from ui.project_state import load_project_state
from ui.tooltips import TOOLTIPS

logger = logging.getLogger("nocode_dl")

# Project root — used to build CWD-independent absolute paths
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_OUTPUTS_ROOT = _PROJECT_ROOT / "outputs"
_BUILTIN_IMAGE_TUTORIAL_PATH      = _PROJECT_ROOT / "fixtures" / "mnist_tutorial"
_BUILTIN_TABULAR_TUTORIAL_PATH    = _PROJECT_ROOT / "fixtures" / "iris_tutorial"
_BUILTIN_TEXT_TUTORIAL_PATH       = _PROJECT_ROOT / "fixtures" / "newsgroups_tutorial"
_BUILTIN_AUDIO_TUTORIAL_PATH      = _PROJECT_ROOT / "fixtures" / "speechcommands_tutorial"
_BUILTIN_TIMESERIES_TUTORIAL_PATH = _PROJECT_ROOT / "fixtures" / "timeseries_tutorial"
_BUILTIN_VIDEO_TUTORIAL_PATH      = _PROJECT_ROOT / "fixtures" / "video_tutorial"


def _path_signature(dpath: str | os.PathLike | None) -> tuple[str, float, bool]:
    if not dpath:
        return ("", 0.0, False)
    try:
        path = Path(dpath).expanduser().resolve()
        stat = path.stat()
        return (str(path), stat.st_mtime, path.is_dir())
    except Exception:
        return (str(dpath), 0.0, False)


def _get_pyplot():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def _pil_image_to_data_url(image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _cleaning_preview_empty_html(message: str) -> str:
    return (
        '<div class="studio-quality-lab">'
        '<h3 style="margin:0 0 10px 0;color:var(--color-text-primary);">Data Quality Lab</h3>'
        f'<p style="margin:0;color:var(--color-text-secondary);">{html.escape(message)}</p>'
        '</div>'
    )


def _render_quality_lab_html(title: str, subtitle: str, chips: list[str], cards_html: list[str]) -> str:
    chips_html = "".join(
        f'<span style="display:inline-flex;align-items:center;padding:7px 12px;border-radius:999px;'
        f'background:var(--color-surface-tint);border:1px solid var(--color-border);margin:0 8px 8px 0;'
        f'font-size:12px;font-weight:600;color:var(--color-text-on-surface);">{html.escape(chip)}</span>'
        for chip in chips
    )
    cards = "".join(cards_html)
    return (
        '<div class="studio-quality-lab" style="margin-top:14px;padding:18px 20px;border:1px solid var(--color-border);'
        'border-radius:20px;background:linear-gradient(180deg,var(--color-surface) 0%,var(--color-surface-soft) 100%);box-shadow:var(--shadow-sm);">'
        f'<h3 style="margin:0 0 8px 0;color:var(--color-text-primary);">{html.escape(title)}</h3>'
        f'<p style="margin:0 0 14px 0;color:var(--color-text-secondary);line-height:1.5;">{html.escape(subtitle)}</p>'
        f'<div style="margin:0 0 12px 0;">{chips_html or "<span style=\"color:var(--color-text-muted);font-size:12px;\">No visible changes selected.</span>"}</div>'
        f'<div style="display:grid;gap:14px;">{cards}</div>'
        '</div>'
    )


def _build_text_quality_lab_html(
    preview_df,
    text_col: str,
    *,
    lowercase: bool,
    strip_urls: bool,
    strip_punctuation: bool,
    remove_stopwords: bool,
    deduplicate: bool,
    apply_stemming: bool,
    apply_lemmatization: bool,
) -> str:
    if preview_df is None or preview_df.empty or not text_col or text_col not in preview_df.columns:
        return _cleaning_preview_empty_html("Preview a text dataset to inspect concrete before-and-after cleaning examples.")

    sample_df = preview_df[[text_col]].copy()
    sample_df[text_col] = sample_df[text_col].fillna("").astype(str)
    sample_df = sample_df[sample_df[text_col].str.strip().str.len() > 0].head(3).reset_index(drop=True)
    if sample_df.empty:
        return _cleaning_preview_empty_html("No non-empty text rows were available for the cleaning preview.")

    from data_pipeline.cleaning import clean_text_dataframe

    cleaned_df, _ = clean_text_dataframe(
        sample_df,
        text_col=text_col,
        lowercase=lowercase,
        strip_urls=strip_urls,
        strip_punctuation=strip_punctuation,
        remove_stopwords=remove_stopwords,
        deduplicate=False,
        apply_stemming=apply_stemming,
        apply_lemmatization=apply_lemmatization,
        drop_empty_rows=False,
    )

    chips: list[str] = []
    if lowercase:
        chips.append("Lowercase")
    if strip_urls:
        chips.append("Remove URLs")
    if strip_punctuation:
        chips.append("Remove punctuation")
    if remove_stopwords:
        chips.append("Remove stop words")
    if apply_stemming:
        chips.append("Stemming")
    if apply_lemmatization:
        chips.append("Lemmatization")
    if deduplicate:
        chips.append("Duplicate rows dropped during training")

    cards_html: list[str] = []
    for idx, (before, after) in enumerate(zip(sample_df[text_col].tolist(), cleaned_df[text_col].tolist()), start=1):
        cards_html.append(
            '<div style="display:grid;grid-template-columns:1fr 1fr;gap:12px;padding:14px;'
            'border:1px solid var(--color-border);border-radius:16px;background:var(--color-surface);">'
            f'<div><div style="margin-bottom:8px;font-size:12px;font-weight:700;color:var(--color-text-muted);letter-spacing:.08em;text-transform:uppercase;">Original sample {idx}</div>'
            f'<div style="padding:12px;border-radius:14px;background:var(--color-surface-soft);border:1px solid var(--color-border-row);color:var(--color-text-primary);line-height:1.55;">{html.escape(before[:360] or " ")}</div></div>'
            f'<div><div style="margin-bottom:8px;font-size:12px;font-weight:700;color:var(--color-accent-dark);letter-spacing:.08em;text-transform:uppercase;">Proposed cleaned version</div>'
            f'<div style="padding:12px;border-radius:14px;background:var(--color-surface-tint);border:1px solid var(--color-border);color:var(--color-text-primary);line-height:1.55;">{html.escape(after[:360] or " ")}</div></div>'
            '</div>'
        )

    subtitle = (
        "These are real rows from the loaded dataset with the current text-cleaning choices applied. "
        "Re-preview after changing the toggles to compare different strategies."
    )
    return _render_quality_lab_html("Data Quality Lab", subtitle, chips, cards_html)


def _build_image_quality_lab_html(
    dpath: str,
    *,
    image_size_value: int,
    horizontal_flip: bool,
    vertical_flip: bool,
    rotation: bool,
    color_jitter: bool,
    grayscale: bool,
    perspective: bool,
    normalization_preset: str,
    force_grayscale: bool,
) -> str:
    path = Path(str(dpath or "")).expanduser()
    if not path.exists():
        return _cleaning_preview_empty_html("Preview an image dataset to inspect concrete before-and-after augmentation examples.")

    from PIL import Image, ImageEnhance, ImageOps

    samples = []
    for candidate in sorted(path.rglob("*")):
        if candidate.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}:
            samples.append(candidate)
        if len(samples) >= 3:
            break

    if not samples:
        return _cleaning_preview_empty_html("No supported image files were found for the cleaning preview.")

    chips: list[str] = []
    if horizontal_flip:
        chips.append("Horizontal flip")
    if vertical_flip:
        chips.append("Vertical flip")
    if rotation:
        chips.append("Rotation")
    if color_jitter:
        chips.append("Color jitter")
    if grayscale:
        chips.append("Augmentation grayscale")
    if perspective:
        chips.append("Perspective warp")
    if force_grayscale:
        chips.append("Convert to grayscale before training")
    if normalization_preset and normalization_preset != "none":
        chips.append(f"Normalization: {normalization_preset}")
    chips.append(f"Training size: {int(image_size_value)}×{int(image_size_value)}")

    cards_html: list[str] = []
    for idx, sample_path in enumerate(samples, start=1):
        with Image.open(sample_path) as img:
            original = img.convert("RGB")
            transformed = original.copy()

        if force_grayscale or grayscale:
            transformed = ImageOps.grayscale(transformed).convert("RGB")
        if horizontal_flip:
            transformed = ImageOps.mirror(transformed)
        if vertical_flip:
            transformed = ImageOps.flip(transformed)
        if rotation:
            transformed = transformed.rotate(15, resample=Image.Resampling.BICUBIC, fillcolor=(245, 248, 244))
        if color_jitter:
            transformed = ImageEnhance.Color(transformed).enhance(1.45)
            transformed = ImageEnhance.Contrast(transformed).enhance(1.08)
        if perspective:
            w, h = transformed.size
            transformed = transformed.transform(
                (w, h),
                Image.Transform.QUAD,
                (8, 6, w - 8, 0, w - 2, h - 8, 0, h - 2),
                resample=Image.Resampling.BICUBIC,
            )

        original_preview = ImageOps.contain(original, (220, 220))
        transformed_preview = ImageOps.contain(transformed, (220, 220))
        original_url = _pil_image_to_data_url(original_preview)
        transformed_url = _pil_image_to_data_url(transformed_preview)
        label = sample_path.parent.name

        cards_html.append(
            '<div style="display:grid;grid-template-columns:1fr 1fr;gap:12px;padding:14px;'
            'border:1px solid var(--color-border);border-radius:16px;background:var(--color-surface);">'
            f'<div><div style="margin-bottom:8px;font-size:12px;font-weight:700;color:var(--color-text-muted);letter-spacing:.08em;text-transform:uppercase;">Original · class {html.escape(label)}</div>'
            f'<div style="padding:10px;border-radius:14px;background:var(--color-surface-soft);border:1px solid var(--color-border-row);text-align:center;"><img src="{original_url}" alt="Original sample {idx}" style="max-width:100%;max-height:220px;border-radius:12px;"/></div></div>'
            f'<div><div style="margin-bottom:8px;font-size:12px;font-weight:700;color:var(--color-accent-dark);letter-spacing:.08em;text-transform:uppercase;">Proposed transformed version</div>'
            f'<div style="padding:10px;border-radius:14px;background:var(--color-surface-tint);border:1px solid var(--color-border);text-align:center;"><img src="{transformed_url}" alt="Transformed sample {idx}" style="max-width:100%;max-height:220px;border-radius:12px;"/></div></div>'
            '</div>'
        )

    subtitle = (
        "These previews use real images from the dataset and deterministic versions of the currently selected image transforms. "
        "They are meant to show the direction of change before training."
    )
    return _render_quality_lab_html("Data Quality Lab", subtitle, chips, cards_html)


# Keep YOLO options local so the app doesn't touch cv2/ultralytics at startup.
YOLO_MODELS = {
    "YOLOv8 Nano  (~6 MB  · fastest)":   "yolov8n.pt",
    "YOLOv8 Small (~22 MB · balanced)":  "yolov8s.pt",
    "YOLOv8 Medium (~52 MB · accurate)": "yolov8m.pt",
}

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────
MODALITIES = ["graph", "image", "text", "tabular", "timeseries", "audio", "video"]
PROJECT_MODES = ["Beginner", "Guided", "Advanced"]
OPTIMIZERS = ["adam", "adamw", "sgd"]
SCHEDULERS = ["cosine", "warmup_cosine", "step", "none"]
AUG_LEVELS = ["none", "light", "medium", "heavy"]
# "multi-label" is omitted until loaders produce multi-hot targets (see models/registry.py).
TASKS      = ["classification", "clustering", "regression", "anomaly"]


# ─────────────────────────────────────────────────────────────────────────────
# UI helpers
# ─────────────────────────────────────────────────────────────────────────────

def _loss_plot(history: list[dict]) -> Any:
    """
    Training/validation loss and metric curves for ``history``.
    """
    from matplotlib.figure import Figure

    epochs     = [h["epoch"]      for h in history]
    train_loss = [h["train_loss"] for h in history]
    val_loss   = [h["val_loss"]   for h in history]
    val_metric = [h["val_acc"]    for h in history]

    # A standalone Figure (not pyplot) per call: pyplot's shared named figure
    # let two concurrent runs draw into each other's plot, and nothing is
    # registered globally, so it is freed once Gradio has rendered it.
    # dpi=130 keeps plots sharp on high-DPI (Retina) displays.
    fig = Figure(figsize=(10, 4), dpi=130)
    ax1, ax2 = fig.subplots(1, 2)
    ax1.plot(epochs, train_loss, label="Train loss", marker="o", markersize=3)
    ax1.plot(epochs, val_loss,   label="Val loss",   marker="o", markersize=3)
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss")
    ax1.set_title("Loss"); ax1.legend()
    ax2.plot(epochs, val_metric, color="steelblue", marker="o", markersize=3)
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("Accuracy % / MAE")
    ax2.set_title("Validation Metric")
    fig.tight_layout()
    return fig


HISTORY_COLUMNS = [
    "id",
    "timestamp",
    "model",
    "modality",
    "task",
    "mode",
    "val_acc",
    "val_loss",
    "bundle",
]


def _format_eval_summary(summary: str) -> str:
    """Render evaluation text as readable markdown."""
    if not summary or not str(summary).strip():
        return "> ℹ️ Evaluation metrics will appear here after training completes."

    text = str(summary).strip()
    if "precision" in text and "recall" in text and "f1-score" in text:
        headline, _, report_block = text.partition("\n")
        accuracy_match = re.search(r"Validation Accuracy:\s*([0-9.]+)%", text)
        macro_match = re.search(r"macro avg\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s+(\d+)", text)
        weighted_match = re.search(r"weighted avg\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s+(\d+)", text)

        rows = []
        for raw_line in report_block.splitlines():
            line = raw_line.strip()
            if not line or "precision" in line or line.startswith("accuracy"):
                continue
            match = re.match(r"(.+?)\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s+(\d+)$", line)
            if not match:
                continue
            label, precision, recall, f1_score, support = match.groups()
            rows.append((label.strip(), precision, recall, f1_score, support))

        chips = []
        if accuracy_match:
            chips.append(("Accuracy", f"{accuracy_match.group(1)}%"))
        if macro_match:
            chips.append(("Macro F1", macro_match.group(3)))
        if weighted_match:
            chips.append(("Weighted F1", weighted_match.group(3)))

        chip_html = "".join(
            f'<div class="studio-report-chip"><strong>{html.escape(label)}</strong><span>{html.escape(value)}</span></div>'
            for label, value in chips
        )
        rows_html = "".join(
            "<tr>"
            f"<td>{html.escape(label)}</td>"
            f"<td>{html.escape(precision)}</td>"
            f"<td>{html.escape(recall)}</td>"
            f"<td>{html.escape(f1_score)}</td>"
            f"<td>{html.escape(support)}</td>"
            "</tr>"
            for label, precision, recall, f1_score, support in rows
        )

        return (
            f'<div class="studio-report">'
            f'<h3>{html.escape(headline.strip())}</h3>'
            f'<div class="studio-report-chip-row">{chip_html}</div>'
            f'<div class="studio-report-table-wrap">'
            f'<table class="studio-report-table">'
            f'<thead><tr><th>Label</th><th>Precision</th><th>Recall</th><th>F1-score</th><th>Support</th></tr></thead>'
            f'<tbody>{rows_html}</tbody>'
            f'</table>'
            f'</div>'
            f'</div>'
        )

    return text


def _format_status_block(title: str, body: str, *, success: bool = True) -> str:
    """Format action feedback as markdown."""
    cleaned = (body or "").strip()
    icon = "✅" if success else "⚠️"
    if not cleaned:
        return f"*{title} status will appear here.*"
    return f"### {icon} {title}\n\n```text\n{cleaned}\n```"


_DEFAULT_CLASSIFICATION_NOTE = (
    "*Classification plots will appear here for classification or multi-label runs.*"
)
_DEFAULT_EXPLANATION_NOTE = (
    "*Explanation tools will appear here when the current model and modality support them.*"
)
_DEFAULT_DIAGNOSTICS_NOTE = (
    "*Regression and anomaly diagnostics will appear here when they apply to the selected task.*"
)
_DEFAULT_SAMPLE_REVIEW_NOTE = (
    "*Sample review updates after evaluation with misclassifications or clustering structure when available.*"
)

CLASSIFICATION_METRIC_OPTIONS = [
    "accuracy",
    "precision_macro",
    "recall_macro",
    "f1_macro",
    "balanced_accuracy",
    "auc_ovr",
    "mcc",
]
REGRESSION_METRIC_OPTIONS = [
    "mae",
    "rmse",
    "r2",
    "mape",
]


def tip(key: str, fallback: str = "") -> str:
    return TOOLTIPS.get(key, fallback)


def _list_bundles() -> list[str]:
    """Return absolute paths of saved model bundles, most recent first."""
    if not _OUTPUTS_ROOT.is_dir():
        return []
    bundles = []
    for entry in sorted(_OUTPUTS_ROOT.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
        if entry.is_dir() and (entry / "labels.json").exists():
            bundles.append(str(entry))
    return bundles


def _section_intro(title: str, desc: str) -> str:
    """Reusable HTML block for section headers inside cards/groups."""
    return f'<div class="studio-section-intro"><h3>{html.escape(title)}</h3><p>{html.escape(desc)}</p></div>'


_TUTORIAL_MODALITIES = ["image", "tabular", "text", "audio", "timeseries", "video"]

_TUTORIAL_META: dict[str, dict] = {
    "image": {
        "title":    "MNIST handwritten digits",
        "desc":     "10 digit classes · 300 images each · downloads ~12 MB on first use",
        "loaded":   (
            "### Image tutorial loaded — MNIST\n\n"
            "300 PNG images per digit class (0–9). "
            "**Suggested flow:** preview class balance → train a small CNN → evaluate with GradCAM → export.\n\n"
            f"**Dataset**: `{_BUILTIN_IMAGE_TUTORIAL_PATH}`"
        ),
    },
    "tabular": {
        "title":    "Iris species classification",
        "desc":     "3 species · 150 rows · 4 numeric features · no download needed",
        "loaded":   (
            "### Tabular tutorial loaded — Iris\n\n"
            "Classic 3-class dataset: predict iris species from sepal/petal measurements. "
            "**Suggested flow:** check feature distributions → train a gradient-boosted model → inspect SHAP values.\n\n"
            f"**Dataset**: `{_BUILTIN_TABULAR_TUTORIAL_PATH / 'iris.csv'}`"
        ),
    },
    "text": {
        "title":    "20 Newsgroups topic classification",
        "desc":     "3 topic categories · ~900 documents · downloads via scikit-learn on first use",
        "loaded":   (
            "### Text tutorial loaded — 20 Newsgroups\n\n"
            "Classify news posts into Space, Hockey, or Politics. "
            "**Suggested flow:** preview class balance → train a text classifier → evaluate confusion matrix.\n\n"
            f"**Dataset**: `{_BUILTIN_TEXT_TUTORIAL_PATH / 'newsgroups.csv'}`"
        ),
    },
    "audio": {
        "title":    "Spoken digit recognition",
        "desc":     "Digits 0–9 · ~50 WAV clips per class · downloads ~10 MB on first use",
        "loaded":   (
            "### Audio tutorial loaded — Spoken Digits\n\n"
            "Classify short spoken digit recordings (Free Spoken Digit Dataset). "
            "**Suggested flow:** preview waveform distribution → train a 1-D CNN → evaluate accuracy per digit.\n\n"
            f"**Dataset**: `{_BUILTIN_AUDIO_TUTORIAL_PATH}`"
        ),
    },
    "timeseries": {
        "title":    "Synthetic signal classification",
        "desc":     "3 signal types · 200 series each · 50 timesteps · generated locally",
        "loaded":   (
            "### Timeseries tutorial loaded — Synthetic Signals\n\n"
            "Classify sinusoidal vs. ramp signals with added noise — a minimal time-series benchmark. "
            "**Suggested flow:** preview signal statistics → train a 1-D CNN → evaluate per-class accuracy.\n\n"
            f"**Dataset**: `{_BUILTIN_TIMESERIES_TUTORIAL_PATH / 'timeseries.csv'}`"
        ),
    },
    "video": {
        "title":    "Synthetic shape recognition",
        "desc":     "3 shape classes · 30 MP4 clips each · generated locally using OpenCV",
        "loaded":   (
            "### Video tutorial loaded — Synthetic Shapes\n\n"
            "Classify short video clips by moving shape (circle / square / triangle). "
            "**Suggested flow:** preview frame statistics → train a video CNN → evaluate confusion matrix.\n\n"
            f"**Dataset**: `{_BUILTIN_VIDEO_TUTORIAL_PATH}`"
        ),
    },
}


def _tutorial_markdown(modality: str = "image", loaded: bool = False) -> str:
    meta = _TUTORIAL_META.get(modality, _TUTORIAL_META["image"])
    if loaded:
        return meta["loaded"]
    return (
        f"### Tutorial: {meta['title']}\n\n"
        f"{meta['desc']}\n\n"
        "Click **Load Tutorial** to download the dataset (if needed) and pre-fill all settings."
    )


def _build_evaluation_notes(
    *,
    task: str,
    modality: str,
    model_name: str,
    sklearn_model: bool,
    cm_fig,
    roc_fig,
    gcam_fig,
    shap_fig,
    reg_fig,
    anomaly_fig,
    mis_img_fig,
    mis_df_val,
    tsne_fig,
) -> tuple[str, str, str, str]:
    """Create guided notes for each Evaluate section."""
    if task in {"classification", "multi-label"}:
        class_lines = ["### Classification views"]
        class_lines.append("- Confusion matrix is ready." if cm_fig is not None else "- Confusion matrix is not available for this run.")
        class_lines.append("- ROC curves are ready." if roc_fig is not None else "- ROC curves were not generated for this run.")
    else:
        class_lines = [
            "### Classification views",
            f"- This section is not the primary focus for **{task}** runs.",
        ]

    explain_lines = ["### Explanations"]
    if modality in {"image", "audio", "video"} and not sklearn_model:
        explain_lines.append("- Grad-CAM is ready." if gcam_fig is not None else "- Grad-CAM was not generated for this run.")
    else:
        explain_lines.append(f"- Grad-CAM does not apply to **{modality}** runs with the current model.")
    if modality == "tabular":
        explain_lines.append("- SHAP feature importance is ready." if shap_fig is not None else "- SHAP was not available for this model or run.")
    else:
        explain_lines.append("- SHAP is mainly used for tabular runs.")

    diagnostics_lines = ["### Prediction diagnostics"]
    if task == "regression":
        diagnostics_lines.append("- Regression diagnostics are ready." if reg_fig is not None else "- Regression diagnostics were not generated.")
    else:
        diagnostics_lines.append(f"- Regression diagnostics do not apply to **{task}** runs.")
    if task == "anomaly" or model_name == "Autoencoder":
        diagnostics_lines.append("- Anomaly reconstruction analysis is ready." if anomaly_fig is not None else "- Anomaly diagnostics were not generated.")
    else:
        diagnostics_lines.append("- Anomaly diagnostics are reserved for anomaly or autoencoder workflows.")

    review_lines = ["### Sample review"]
    if mis_df_val is not None or mis_img_fig is not None:
        review_lines.append("- Misclassified examples are available below.")
    elif task in {"classification", "multi-label", "regression"} and model_name != "Autoencoder":
        review_lines.append("- No sample review artifacts were produced for this run.")
    else:
        review_lines.append("- Misclassification review is not central to this task.")
    if tsne_fig is not None:
        review_lines.append("- Clustering structure is available in the t-SNE plot.")
    elif task == "clustering" or model_name == "Autoencoder":
        review_lines.append("- Clustering/t-SNE was skipped or unavailable for this run.")
    else:
        review_lines.append("- t-SNE visualisation is reserved for clustering-style runs.")

    return (
        "\n".join(class_lines),
        "\n".join(explain_lines),
        "\n".join(diagnostics_lines),
        "\n".join(review_lines),
    )


def _format_eval_kpis(summary: str, history: list[dict], task: str) -> str:
    """Render a compact KPI strip for the Evaluate tab."""
    metric_label = "Validation metric"
    metric_value = "Not available"
    accuracy_match = re.search(r"Validation Accuracy:\s*([0-9.]+)%", summary or "")
    if accuracy_match:
        metric_label = "Validation accuracy"
        metric_value = f"{accuracy_match.group(1)}%"
    elif history:
        last_metric = history[-1].get("val_acc")
        if last_metric is not None:
            metric_value = str(last_metric)

    val_loss = "Not available"
    if history and history[-1].get("val_loss") is not None:
        val_loss = str(history[-1]["val_loss"])

    cards = [
        ("Run type", f"{task.replace('_', ' ').title()} run"),
        (metric_label, metric_value),
        ("Validation loss", val_loss),
    ]
    card_html = "".join(
        f'<div class="studio-kpi"><strong>{html.escape(label)}</strong><span>{html.escape(value)}</span></div>'
        for label, value in cards
    )
    return f'<div class="studio-kpi-grid studio-kpi-grid--compact">{card_html}</div>'


def _metric_display_value(metric_name: str, value):
    if value is None:
        return None
    if metric_name in {"accuracy", "precision_macro", "recall_macro", "f1_macro", "balanced_accuracy"}:
        return round(float(value) * 100.0, 2)
    return round(float(value), 4)


# ── Dashboard helpers ─────────────────────────────────────────────────────

def _build_dashboard_kpi(
    task: str, modality: str, history: list[dict],
    metrics_payload: dict, model_name: str, training_mode: str,
) -> str:
    """Build the KPI strip HTML for the executive dashboard."""
    task_label = (task or "unknown").replace("_", " ").title()

    # Primary metric
    if task in ("classification", "multi-label"):
        metric_val = metrics_payload.get("accuracy")
        if metric_val is not None:
            metric_val = f"{float(metric_val) * 100:.1f}%"
            raw = float(metrics_payload["accuracy"])
            color = "#16856f" if raw >= 0.8 else "#c4820e" if raw >= 0.6 else "#c42b1c"
        else:
            metric_val = f"{history[-1]['val_acc']:.1f}%" if history else "—"
            color = "var(--color-text-primary)"
        metric_label = "Validation accuracy"
    elif task == "regression":
        metric_val = metrics_payload.get("mae")
        metric_label = "Validation MAE"
        metric_val = f"{float(metric_val):.4f}" if metric_val else "—"
        color = "var(--color-text-primary)"
    elif task == "clustering":
        metric_val = metrics_payload.get("silhouette")
        metric_label = "Silhouette"
        metric_val = f"{float(metric_val):.3f}" if metric_val else "—"
        color = "var(--color-text-primary)"
    else:
        metric_val, metric_label, color = "—", "Metric", "var(--color-text-primary)"

    # Val loss
    val_loss = f"{history[-1]['val_loss']:.4f}" if history else "—"

    # Epochs
    n_epochs = len(history) if history else 0
    epoch_text = f"{n_epochs}"

    cards = [
        ("Task", f"{task_label} · {modality}"),
        (metric_label, f'<span style="color:{color};font-size:1.5rem;">{metric_val}</span>'),
        ("Val Loss", val_loss),
        ("Model", f"{model_name} ({training_mode})"),
    ]
    card_html = "".join(
        f'<div class="studio-kpi"><strong>{html.escape(label)}</strong><span>{value}</span></div>'
        for label, value in cards
    )
    return f'<div class="studio-kpi-grid">{card_html}</div>'


def _pick_primary_chart(task, cm_fig, reg_fig, tsne_fig, anomaly_fig):
    """Return the most relevant diagnostic chart for the task."""
    if task in ("classification", "multi-label"):
        return cm_fig
    if task == "regression":
        return reg_fig
    if task == "clustering":
        return tsne_fig
    if task == "anomaly":
        return anomaly_fig
    return cm_fig or reg_fig or tsne_fig or anomaly_fig


def _build_action_items(
    task: str, history: list[dict], metrics_payload: dict, model_name: str,
) -> str:
    """Generate actionable recommendations based on training results."""
    items = []

    if not history:
        return "### Recommendations\n\n*Train a model to see recommendations.*"

    # Check for overfitting
    if len(history) >= 3:
        last = history[-1]
        if last["val_loss"] > last["train_loss"] * 1.3:
            gap = (last["val_loss"] - last["train_loss"]) / max(last["train_loss"], 1e-6) * 100
            items.append(
                f"**Possible overfitting** — validation loss is {gap:.0f}% higher than training loss. "
                "Consider more training data, stronger regularization (dropout), or data augmentation."
            )

    # Check for underfitting
    if task in ("classification", "multi-label"):
        acc = metrics_payload.get("accuracy")
        if acc is not None and float(acc) < 0.7:
            items.append(
                f"**Low accuracy ({float(acc)*100:.1f}%)** — the model may be underfitting. "
                "Try a larger architecture, more epochs, or a higher learning rate."
            )
        f1 = metrics_payload.get("f1_macro")
        if acc and f1 and abs(float(acc) - float(f1)) > 0.15:
            items.append(
                "**Accuracy/F1 gap** — high accuracy but lower F1 suggests class imbalance. "
                "Check the confusion matrix for classes the model ignores."
            )

    # Check early stopping
    if len(history) < 10 and len(history) >= 3:
        last_losses = [h["val_loss"] for h in history[-3:]]
        if all(last_losses[i] >= last_losses[i-1] for i in range(1, len(last_losses))):
            items.append(
                f"**Early stopping likely triggered** at epoch {len(history)}. "
                "The model converged quickly — try a lower learning rate for finer tuning."
            )

    # Check for good results
    if task in ("classification", "multi-label"):
        acc = metrics_payload.get("accuracy")
        if acc and float(acc) >= 0.9:
            items.append(
                f"**Strong performance ({float(acc)*100:.1f}% accuracy)** — "
                "verify on held-out data and check the confusion matrix for edge cases before deploying."
            )

    # Always suggest next steps
    items.append(
        "**Next steps** — review the Evaluate tab for detailed diagnostics, "
        "try the model on new inputs in Try Your Model, and export via Export & History."
    )

    lines = ["### Recommendations", ""]
    for item in items:
        lines.append(f"- {item}")
    return "\n".join(lines)


def _slug_part(value: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9]+", "_", (value or "").strip().lower()).strip("_")
    return cleaned or "project"


def suggest_bundle_name(dpath: str, mod: str, model: str, task: str, current_name: str) -> str:
    current = (current_name or "").strip()
    if current and current not in {"my_model", "project"}:
        return current

    source = ""
    if dpath:
        source = Path(str(dpath)).stem
    parts = [
        _slug_part(source) if source else _slug_part(mod or "dataset"),
        _slug_part(model or "model"),
        _slug_part(task or "task"),
    ]
    return "_".join(parts[:3])


def _workspace_status_markdown(
    dpath: str,
    mod: str,
    label: str,
    model: str,
    task: str,
    bundle_name: str,
    bundle_path: str,
) -> tuple[str, str]:
    dataset_ready = bool(str(dpath or "").strip())
    model_ready = bool(str(model or "").strip())
    trained = bool(str(bundle_path or "").strip())
    effective_bundle_name = suggest_bundle_name(dpath, mod, model, task, bundle_name)

    dataset_status = "Ready" if dataset_ready else "Waiting"
    model_status = "Ready" if model_ready else "Waiting"
    train_status = "Complete" if trained else ("Ready to run" if dataset_ready and model_ready else "Blocked")

    snapshot = [
        "### Workspace cockpit",
        "",
        f"- **Dataset**: {dataset_status}",
        f"- **Modality**: `{mod or 'not set'}`",
        f"- **Label column**: `{label or 'not set'}`",
        f"- **Model**: `{model or 'not selected'}`",
        f"- **Task**: `{task or 'not selected'}`",
        f"- **Bundle name**: `{effective_bundle_name or 'not set'}`",
        f"- **Training**: {train_status}",
    ]
    if trained:
        snapshot.append(f"- **Latest bundle**: `{bundle_path}`")

    if not dataset_ready:
        next_step = (
            "### Next step\n\n"
            "Upload a dataset or paste a local path, then click **Preview Dataset**. "
            "The app will infer columns and help set up the workflow for you."
        )
    elif not model_ready:
        next_step = (
            "### Next step\n\n"
            "Move to **Model**, keep the suggested defaults if you want a quick start, "
            "and adjust only the task or architecture you actually want to explore."
        )
    elif not trained:
        next_step = (
            "### Next step\n\n"
            "Go to **Train** and run a short baseline first. After that, review **Evaluate** "
            "and try the saved bundle in **Try Your Model**."
        )
    else:
        next_step = (
            "### Next step\n\n"
            "Your latest bundle is ready. Review the evaluation dashboard, try a prediction, "
            "or export a serving bundle from **Export & History**."
        )

    return "\n".join(snapshot), next_step


def _project_mode_default() -> str:
    return str(load_project_state().get("project_mode", "Guided") or "Guided")


def _workflow_mode_updates(project_mode: str):
    mode = project_mode or "Guided"
    is_beginner = mode == "Beginner"
    is_guided = mode == "Guided"
    is_advanced = mode == "Advanced"
    return (
        gr.update(visible=True),             # split_cleaning_group
        gr.update(visible=not is_beginner),  # advanced_training_group
        gr.update(visible=is_advanced),      # sequence_options_group
        gr.update(visible=is_advanced),      # sklearn_options_group
        gr.update(visible=True),             # column_mapping_group
        gr.update(visible=not is_beginner),  # hyperparams_group
        gr.update(visible=True),             # model_selection_group
        gr.update(visible=not is_beginner),  # evaluation_report_group
        gr.update(visible=not is_beginner),  # export_utilities_group
        gr.update(visible=not is_beginner),  # history_group
        gr.update(visible=True),             # explanation_group_top
    )


def _why_this_matters_markdown(project_mode: str, mod: str, model: str, task: str, recommendation: dict | None = None) -> str:
    mode = project_mode or "Guided"
    recommendation = recommendation or {}
    if not model:
        return (
            "### Why this matters\n\n"
            f"**{mode} mode** is active. Start by previewing the dataset so the workspace can recommend a safe baseline. "
            "The goal of the first pass is not perfect tuning. It is to confirm the data, task, and model family line up."
        )

    rec_model = recommendation.get("model_name")
    rec_reason = recommendation.get("rationale", [])
    next_step = "Run a short baseline next, then trust the evaluation report before changing many settings."
    lines = ["### Why this matters", ""]
    if rec_model:
        lines.append(f"- The current guided baseline favours **{rec_model}** for **{mod}** data.")
    elif model:
        lines.append(f"- You are working with **{model}** for a **{task}** task.")
    if rec_reason:
        lines.append(f"- {rec_reason[0]}")
    if task == "classification":
        lines.append("- Accuracy is useful, but class imbalance means F1 and per-class review matter too.")
    elif task == "regression":
        lines.append("- Focus on MAE/RMSE and residual patterns, not just a single headline score.")
    elif task == "clustering":
        lines.append("- Clustering quality is about whether the learned structure is stable and interpretable, not just whether a chart looks separated.")
    if mode == "Beginner":
        lines = lines[:4]
    elif mode == "Advanced":
        lines.append("- Advanced mode keeps more controls visible, so change one variable at a time if you want clean comparisons.")
    lines.append(f"- {next_step}")
    return "\n".join(lines)


def _apply_recommendation_payload(modality_name: str, rec: dict | None):
    rec = rec or {}
    model_choices = get_models(modality_name, rec.get("training_mode", "fine-tune"))
    task_choices = get_compatible_tasks(modality_name)
    if modality_name in {"tabular", "timeseries"} and "regression" not in task_choices:
        task_choices = task_choices + ["regression"]
    model_value = rec.get("model_name")
    if model_value not in model_choices and model_choices:
        model_value = model_choices[0]
    task_value = rec.get("task")
    if task_value not in task_choices and task_choices:
        task_value = task_choices[0]
    return (
        gr.update(value=rec.get("training_mode", "fine-tune")),
        gr.update(choices=model_choices, value=model_value),
        gr.update(choices=task_choices, value=task_value),
        gr.update(value=rec.get("augmentation", "light")),
        gr.update(value=rec.get("image_size", 160)),
        gr.update(value=rec.get("batch_size", 16)),
        gr.update(value=rec.get("epochs", 8)),
        gr.update(value=rec.get("dropout", 0.3)),
        gr.update(value=rec.get("scheduler", "cosine")),
        gr.update(value=bool(rec.get("use_class_weights", False))),
    )


@lru_cache(maxsize=32)
def _read_structured_preview_cached(path_sig):
    if not path_sig[0]:
        return None

    candidate = path_sig[0]
    if path_sig[2]:
        graph_nodes = os.path.join(candidate, "nodes.csv")
        if os.path.isfile(graph_nodes):
            candidate = graph_nodes
        else:
            structured = []
            for root, _dirs, files in os.walk(candidate):
                for name in files:
                    if name.lower().endswith((".csv", ".tsv", ".json")):
                        structured.append(os.path.join(root, name))
            if len(structured) == 1:
                candidate = structured[0]
            else:
                return None

    from data_pipeline.io_utils import read_structured_file

    lower = candidate.lower()
    if lower.endswith((".csv", ".tsv", ".json")):
        return read_structured_file(candidate, nrows=500)
    return None


def _read_structured_preview(dpath):
    return _read_structured_preview_cached(_path_signature(dpath))


def _suggest_tabular_example(dpath: str, mod: str, label_col: str, current_value: str) -> str:
    if (current_value or "").strip():
        return current_value
    if mod != "tabular":
        return current_value or ""
    try:
        df = _read_structured_preview(dpath)
        if df is None or df.empty:
            return current_value or ""
        row = df.iloc[0].to_dict()
        if label_col in row:
            row.pop(label_col, None)
        if not row:
            return current_value or ""
        return "{\n" + ",\n".join(
            f'  "{k}": {repr(v) if isinstance(v, str) else v}'
            for k, v in row.items()
        ) + "\n}"
    except Exception:
        return current_value or ""


def _suggest_timeseries_example(dpath: str, mod: str, label_col: str, time_col: str, current_value: str) -> str:
    if (current_value or "").strip():
        return current_value
    if mod != "timeseries":
        return current_value or ""
    try:
        df = _read_structured_preview(dpath)
        if df is None or df.empty:
            return current_value or ""
        feature_cols = [c for c in df.columns if c not in {label_col, time_col}]
        if not feature_cols:
            return current_value or ""
        window = df[feature_cols].head(6).to_dict(orient="records")
        lines = []
        for row in window:
            entries = ", ".join(
                f'"{k}": {repr(v) if isinstance(v, str) else float(v) if v is not None else 0}'
                for k, v in row.items()
            )
            lines.append(f"  {{{entries}}}")
        return "[\n" + ",\n".join(lines) + "\n]"
    except Exception:
        return current_value or ""


def _format_live_training_metrics_card(metrics: dict | None = None) -> str:
    metrics = metrics or {}

    def _value(key: str, default: str = "—") -> str:
        value = metrics.get(key)
        if value in (None, ""):
            return default
        return str(value)

    cards = [
        ("Phase", _value("phase", "Waiting")),
        ("Epoch", _value("epoch_text")),
        ("Train loss", _value("train_loss")),
        ("Val loss", _value("val_loss")),
        ("Metric", _value("metric")),
        ("Learning rate", _value("lr")),
    ]
    card_html = "".join(
        f'<div class="studio-kpi"><strong>{html.escape(label)}</strong><span>{html.escape(value)}</span></div>'
        for label, value in cards
    )
    return f'<div class="studio-kpi-grid studio-kpi-grid--compact">{card_html}</div>'

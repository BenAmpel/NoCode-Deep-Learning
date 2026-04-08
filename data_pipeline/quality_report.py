"""
Structured preflight dataset quality reporting for the guided workflow.
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any


def build_quality_report(
    data_path: str,
    modality: str,
    label_col: str,
    stats: dict[str, Any],
    validation_messages: list[str] | None = None,
    preview_df=None,
) -> dict[str, Any]:
    report = {
        "status": "ready",
        "blocking_issues": [],
        "warnings": [],
        "info": [],
        "suggested_actions": [],
    }

    validation_messages = validation_messages or []
    if "error" in stats:
        report["status"] = "blocked"
        report["blocking_issues"].append(str(stats["error"]))
        return report

    for msg in validation_messages:
        if "error" in msg.lower() or "not found" in msg.lower():
            report["blocking_issues"].append(msg)
        elif "warning" in msg.lower() or "⚠" in msg:
            report["warnings"].append(msg)
        else:
            report["info"].append(msg)

    class_counts = stats.get("class_counts", {}) or {}
    if stats.get("imbalance_ratio", 1.0) >= 3.0:
        report["warnings"].append(
            f"Class imbalance is {stats['imbalance_ratio']:.2f}x. Evaluation should include F1, recall, and per-class review."
        )
        report["suggested_actions"].append(
            "Enable class weighting or choose a stronger baseline before tuning many hyperparameters."
        )
        if modality == "image":
            report["suggested_actions"].append(
                "For image data, pair class weighting with a compact model and moderate augmentation before increasing image size."
            )

    tiny_classes = [name for name, count in class_counts.items() if count < 10]
    if tiny_classes:
        report["warnings"].append(
            f"Tiny classes detected: {', '.join(map(str, tiny_classes[:6]))}."
        )
        report["suggested_actions"].append(
            "Merge, collect, or review tiny classes before treating the evaluation as final."
        )

    if modality == "image":
        _image_quality_checks(Path(data_path), report)
    elif modality in {"tabular", "text", "timeseries"} and preview_df is not None:
        _structured_quality_checks(preview_df, label_col, modality, report)

    if report["blocking_issues"]:
        report["status"] = "blocked"
    elif report["warnings"]:
        report["status"] = "review"
    else:
        report["status"] = "ready"

    if not report["suggested_actions"]:
        report["suggested_actions"].append("Preview looks healthy. A short baseline run is the next best step.")

    return report


def _image_quality_checks(data_path: Path, report: dict[str, Any]) -> None:
    if not data_path.exists() or not data_path.is_dir():
        return
    image_paths = [p for p in data_path.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}]
    if not image_paths:
        report["blocking_issues"].append("No supported image files were found in the selected folder.")
        return

    stems = Counter(path.stem for path in image_paths)
    duplicates = [stem for stem, count in stems.items() if count > 1]
    if duplicates:
        report["warnings"].append(
            f"Repeated image basenames detected ({len(duplicates)}). This can hide duplicate content across splits."
        )
    unreadable = 0
    try:
        from PIL import Image

        for path in image_paths[:200]:
            try:
                with Image.open(path) as img:
                    img.verify()
            except Exception:
                unreadable += 1
    except Exception:
        unreadable = 0
    if unreadable:
        report["warnings"].append(
            f"{unreadable} sampled images could not be verified cleanly."
        )
        report["suggested_actions"].append("Enable file verification before training if the dataset came from multiple sources.")


def _structured_quality_checks(df, label_col: str, modality: str, report: dict[str, Any]) -> None:
    import numpy as np
    import pandas as pd

    if label_col not in df.columns:
        report["blocking_issues"].append(f"Label column '{label_col}' was not found in the previewed file.")
        return

    duplicate_rows = int(df.duplicated().sum())
    if duplicate_rows:
        report["warnings"].append(f"{duplicate_rows} duplicate rows were found in the preview sample.")

    missing_labels = int(df[label_col].isna().sum())
    if missing_labels:
        report["warnings"].append(f"{missing_labels} rows have missing labels and will be dropped automatically.")

    suspicious_columns: list[str] = []
    leakage_columns: list[str] = []
    for col in df.columns:
        series = df[col]
        nunique = int(series.nunique(dropna=True))
        if col != label_col and len(df) > 0 and nunique >= max(5, int(len(df) * 0.95)):
            suspicious_columns.append(col)
        lowered = col.lower()
        if any(token in lowered for token in ("id", "uuid", "filename", "path", "index")):
            leakage_columns.append(col)

    if suspicious_columns:
        report["warnings"].append(
            f"Near-unique columns detected: {', '.join(suspicious_columns[:6])}."
        )
        report["suggested_actions"].append("Review whether ID-like columns should be removed before training.")

    if leakage_columns:
        report["warnings"].append(
            f"Potential leakage columns detected: {', '.join(leakage_columns[:6])}."
        )

    if modality == "tabular":
        cat_cols = df.drop(columns=[label_col], errors="ignore").select_dtypes(include=["object", "category"]).columns.tolist()
        if cat_cols:
            report["info"].append(
                f"Categorical columns will be one-hot encoded: {', '.join(cat_cols[:6])}."
            )
            report["info"].append(
                f"Encoding review: {len(cat_cols)} categorical columns will expand into indicator features during preprocessing."
            )
        numeric_cols = df.drop(columns=[label_col], errors="ignore").select_dtypes(include=[np.number]).columns.tolist()
        if not numeric_cols:
            report["warnings"].append("No numeric feature columns were detected in the preview sample.")
        else:
            report["info"].append(
                f"Scaling review: {len(numeric_cols)} numeric columns will be scaled using the selected preprocessing strategy."
            )
    if modality == "text":
        text_like = None
        for col in df.columns:
            if col == label_col:
                continue
            if df[col].dtype == object:
                text_like = col
                break
        if text_like:
            duplicate_text = int(df[text_like].fillna("").astype(str).duplicated().sum())
            if duplicate_text:
                report["warnings"].append(f"{duplicate_text} duplicate text rows were found in '{text_like}'.")
            report["info"].append(_language_hint(df[text_like].fillna("").astype(str).head(150).tolist()))


def _language_hint(samples: list[str]) -> str:
    joined = " ".join(samples).lower()
    english_hits = sum(joined.count(token) for token in (" the ", " and ", " is ", " to ", " of "))
    if english_hits >= 5:
        return "Language signal: the preview looks mostly English, so standard English preprocessing defaults are likely appropriate."
    return "Language signal: language could not be identified confidently from the preview, so keep text cleaning conservative."


def quality_report_markdown(report: dict[str, Any]) -> str:
    status_map = {
        "blocked": "### Data Quality Report\n\nStatus: **Blocked**",
        "review": "### Data Quality Report\n\nStatus: **Needs review**",
        "ready": "### Data Quality Report\n\nStatus: **Ready**",
    }
    lines = [status_map.get(report.get("status", "ready"), "### Data Quality Report")]
    for heading, key in (
        ("Blocking issues", "blocking_issues"),
        ("Warnings", "warnings"),
        ("Informational findings", "info"),
        ("Suggested actions", "suggested_actions"),
    ):
        values = report.get(key, [])
        if not values:
            continue
        lines.extend(["", f"#### {heading}"])
        lines.extend(f"- {value}" for value in values)
    if len(lines) == 1:
        lines.extend(["", "- No notable issues were detected in the previewed dataset."])
    return "\n".join(lines)

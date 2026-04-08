"""
Dataset statistics: class distribution, sample counts, imbalance detection, basic previews.
Returns structured dicts and matplotlib figures for display in the UI.
"""
from __future__ import annotations
from pathlib import Path
from typing import Optional
import numpy as np
import matplotlib.pyplot as plt

VISUAL_EXTS  = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}
AUDIO_EXTS   = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}
VIDEO_EXTS   = {".mp4", ".avi", ".mov", ".mkv", ".webm"}

def compute_stats(
    data_path: str,
    modality: str,
    label_col: str = "label",
    subset_enabled: bool = False,
    subset_percent: float = 100.0,
    subset_seed: int = 42,
) -> dict:
    """
    Returns a dict with keys:
        n_samples, n_classes, classes, class_counts (dict),
        imbalance_ratio, warnings (list[str]), summary (str)
    """
    data_path = Path(data_path)
    if not data_path.exists():
        return {"error": f"Path does not exist: {data_path}"}

    if modality in ("image", "audio", "video"):
        return _folder_stats(data_path, modality)
    else:
        return _csv_stats(data_path, label_col, subset_enabled=subset_enabled, subset_percent=subset_percent, subset_seed=subset_seed)

def _folder_stats(data_path: Path, modality: str) -> dict:
    ext_map = {"image": VISUAL_EXTS, "audio": AUDIO_EXTS, "video": VIDEO_EXTS}
    valid_exts = ext_map[modality]
    classes = sorted(d.name for d in data_path.iterdir() if d.is_dir())
    if not classes:
        return {"error": "No class subdirectories found. Expected: data_path/<class_name>/files"}

    class_counts = {}
    for cls in classes:
        count = sum(1 for f in (data_path / cls).iterdir() if f.suffix.lower() in valid_exts)
        class_counts[cls] = count

    return _build_stats(class_counts)

def _csv_stats(
    data_path: Path,
    label_col: str,
    *,
    subset_enabled: bool = False,
    subset_percent: float = 100.0,
    subset_seed: int = 42,
) -> dict:
    from data_pipeline.io_utils import apply_random_subset, drop_missing_label_rows, read_structured_file
    try:
        df = read_structured_file(data_path)
    except Exception as e:
        return {"error": f"Could not read CSV: {e}"}
    if label_col not in df.columns:
        return {"error": f"Label column '{label_col}' not found. Available: {list(df.columns)}"}
    df, dropped_missing = drop_missing_label_rows(df, label_col)
    if df.empty:
        return {"error": f"No rows remain after dropping missing labels from '{label_col}'."}
    sampled_rows = len(df)
    if subset_enabled:
        df, sampled_rows = apply_random_subset(df, enabled=True, subset_percent=subset_percent, subset_seed=subset_seed)

    class_counts = dict(df[label_col].astype(str).value_counts())
    stats = _build_stats(class_counts, n_features=len(df.columns) - 1)
    if dropped_missing:
        stats.setdefault("warnings", []).append(
            f"ℹ️  Dropped {dropped_missing} row(s) with missing labels before counting classes."
        )
        stats["summary"] += f"\nDropped missing-label rows: {dropped_missing}"
    if subset_enabled and subset_percent < 100:
        stats.setdefault("warnings", []).append(
            f"ℹ️  Previewing a random subset of {sampled_rows} rows ({float(subset_percent):.2f}% of the cleaned dataset)."
        )
        stats["summary"] += f"\nSubset preview : {sampled_rows} rows ({float(subset_percent):.2f}%)"
    return stats

def _build_stats(class_counts: dict, n_features: Optional[int] = None) -> dict:
    counts   = list(class_counts.values())
    total    = sum(counts)
    n_cls    = len(counts)
    ratio    = max(counts) / (min(counts) + 1e-8)
    warnings = []

    if total == 0:
        return {"error": "No samples found."}
    if ratio > 5:
        warnings.append(f"⚠️  Severe class imbalance (ratio {ratio:.1f}x). Consider weighted loss or oversampling.")
    elif ratio > 2:
        warnings.append(f"ℹ️  Mild class imbalance (ratio {ratio:.1f}x).")
    for cls, cnt in class_counts.items():
        if cnt < 10:
            warnings.append(f"⚠️  Class '{cls}' has only {cnt} samples — may cause poor generalisation.")
    if total < 100:
        warnings.append("ℹ️  Small dataset (< 100 samples). Consider fine-tuning over training from scratch.")

    lines = [
        f"Total samples : {total}",
        f"Classes       : {n_cls}",
    ]
    if n_features:
        lines.append(f"Features      : {n_features}")
    lines.append(f"Imbalance ratio: {ratio:.2f}x")
    lines += [""] + (warnings if warnings else ["✅  No major data issues detected."])
    summary = "\n".join(lines)

    return {
        "n_samples":       total,
        "n_classes":       n_cls,
        "classes":         list(class_counts.keys()),
        "class_counts":    class_counts,
        "imbalance_ratio": round(ratio, 2),
        "warnings":        warnings,
        "summary":         summary,
    }

def plot_class_distribution(stats: dict) -> plt.Figure | None:
    if "error" in stats or not stats.get("class_counts"):
        return None
    classes = list(stats["class_counts"].keys())
    counts  = list(stats["class_counts"].values())
    n = len(classes)
    colors = plt.cm.tab20(np.linspace(0, 1, n))
    fig, ax = plt.subplots(figsize=(max(6, n * 0.6), 4))
    bars = ax.bar(classes, counts, color=colors)
    ax.bar_label(bars, padding=3, fontsize=8)
    ax.set_xlabel("Class")
    ax.set_ylabel("Sample count")
    ax.set_title("Class Distribution")
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right", fontsize=8)
    fig.tight_layout()
    return fig

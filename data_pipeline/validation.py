"""
Pre-training dataset validation.
Returns a list of human-readable warning / error strings.
"""
from __future__ import annotations
from pathlib import Path

VISUAL_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}
AUDIO_EXTS  = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}
VIDEO_EXTS  = {".mp4", ".avi", ".mov", ".mkv", ".webm"}

def validate_dataset(
    data_path: str,
    modality: str,
    label_col: str = "label",
    subset_enabled: bool = False,
    subset_percent: float = 100.0,
    subset_seed: int = 42,
) -> list[str]:
    """Returns list of warning strings; empty list = all OK."""
    path = Path(data_path)
    issues: list[str] = []

    if not path.exists():
        return [f"❌  Path not found: {path}"]

    if modality in ("image", "audio", "video"):
        issues += _validate_folder(path, modality)
    else:
        issues += _validate_csv(
            path,
            modality,
            label_col,
            subset_enabled=subset_enabled,
            subset_percent=subset_percent,
            subset_seed=subset_seed,
        )

    return issues

def _validate_folder(path: Path, modality: str) -> list[str]:
    ext_map = {"image": VISUAL_EXTS, "audio": AUDIO_EXTS, "video": VIDEO_EXTS}
    valid   = ext_map[modality]
    issues  = []

    subdirs = [d for d in path.iterdir() if d.is_dir()]
    if len(subdirs) < 2:
        issues.append(f"❌  Found {len(subdirs)} class folder(s). Need at least 2 for classification.")

    for d in subdirs:
        files = [f for f in d.iterdir() if f.suffix.lower() in valid]
        if len(files) == 0:
            issues.append(f"❌  Class folder '{d.name}' has no valid {modality} files.")
        elif len(files) < 5:
            issues.append(f"⚠️  Class '{d.name}' has only {len(files)} samples.")

    # Check for hidden files / non-media files at top level
    stray = [f for f in path.iterdir() if f.is_file()]
    if stray:
        issues.append(f"ℹ️  {len(stray)} file(s) found at the top level of the folder (expected only subdirs).")

    return issues

def _validate_csv(
    path: Path,
    modality: str,
    label_col: str,
    *,
    subset_enabled: bool = False,
    subset_percent: float = 100.0,
    subset_seed: int = 42,
) -> list[str]:
    from data_pipeline.io_utils import apply_random_subset, drop_missing_label_rows, read_structured_file
    issues = []
    try:
        df = read_structured_file(path)
    except Exception as e:
        return [f"❌  Cannot read CSV: {e}"]

    if label_col not in df.columns:
        issues.append(f"❌  Column '{label_col}' not in CSV. Available: {list(df.columns)}")
        return issues

    if modality == "text":
        # Check for a text column (heuristic: largest string column)
        str_cols = df.select_dtypes(include="object").columns.tolist()
        if len(str_cols) < 2:
            issues.append("⚠️  Only one string column found. Make sure to specify the text column name.")

    n_missing = df[label_col].isna().sum()
    if n_missing > 0:
        issues.append(f"⚠️  {n_missing} rows have missing labels and will be dropped.")
        df, _ = drop_missing_label_rows(df, label_col)
    if subset_enabled and subset_percent < 100:
        df, sampled_rows = apply_random_subset(df, enabled=True, subset_percent=subset_percent, subset_seed=subset_seed)
        issues.append(f"ℹ️  Validation is using a random subset of {sampled_rows} rows ({float(subset_percent):.2f}%).")

    n_classes = df[label_col].nunique()
    if n_classes < 2:
        issues.append(f"❌  Only {n_classes} unique class(es) found. Need at least 2.")
    if n_classes > 100:
        issues.append(f"⚠️  {n_classes} classes detected — is this correct?")

    if modality == "timeseries":
        num_cols = df.select_dtypes(include="number").columns.tolist()
        feat_cols = [c for c in num_cols if c != label_col]
        if len(feat_cols) == 0:
            issues.append("❌  No numeric feature columns found for time-series.")

    return issues

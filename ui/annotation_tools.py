from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

from data_pipeline.io_utils import read_structured_file


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}
AUDIO_SUFFIXES = {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".aac"}


def _iter_labeled_files(dataset_dir: str | Path, suffixes: set[str]) -> list[Path]:
    root = Path(dataset_dir).expanduser()
    if not root.exists() or not root.is_dir():
        raise ValueError(f"Dataset directory not found: {root}")
    files = [
        p for p in root.rglob("*")
        if p.is_file() and p.suffix.lower() in suffixes and p.parent != root
    ]
    return sorted(files)


def list_folder_labels(dataset_dir: str | Path, suffixes: set[str]) -> list[str]:
    labels = sorted({p.parent.name for p in _iter_labeled_files(dataset_dir, suffixes)})
    return labels


def list_folder_samples(dataset_dir: str | Path, suffixes: set[str], limit: int = 250) -> list[tuple[str, str]]:
    root = Path(dataset_dir).expanduser()
    samples = []
    for path in _iter_labeled_files(root, suffixes)[:limit]:
        rel = str(path.relative_to(root))
        samples.append((f"{path.parent.name} · {rel}", rel))
    return samples


def preview_folder_sample(dataset_dir: str | Path, rel_path: str) -> tuple[str, str]:
    root = Path(dataset_dir).expanduser()
    sample_path = root / rel_path
    if not sample_path.exists():
        raise ValueError(f"Sample not found: {sample_path}")
    return str(sample_path), sample_path.parent.name


def relabel_folder_sample(dataset_dir: str | Path, rel_path: str, new_label: str) -> tuple[str, str]:
    root = Path(dataset_dir).expanduser()
    sample_path = root / rel_path
    if not sample_path.exists():
        raise ValueError(f"Sample not found: {sample_path}")
    clean_label = (new_label or "").strip()
    if not clean_label:
        raise ValueError("Provide a new label before saving.")
    target_dir = root / clean_label
    target_dir.mkdir(parents=True, exist_ok=True)
    target_path = target_dir / sample_path.name
    if target_path.exists():
        stem = sample_path.stem
        suffix = sample_path.suffix
        counter = 1
        while target_path.exists():
            target_path = target_dir / f"{stem}_{counter}{suffix}"
            counter += 1
    shutil.move(str(sample_path), str(target_path))
    return str(target_path.relative_to(root)), clean_label


def list_text_annotation_rows(
    data_path: str | Path,
    text_col: str,
    label_col: str,
    limit: int = 250,
) -> tuple[list[tuple[str, int]], list[str]]:
    df = read_structured_file(data_path)
    if text_col not in df.columns:
        raise ValueError(f"Text column '{text_col}' was not found.")
    if label_col not in df.columns:
        raise ValueError(f"Label column '{label_col}' was not found.")
    choices: list[tuple[str, int]] = []
    for idx, row in df.head(limit).iterrows():
        text_preview = str(row.get(text_col, ""))[:90].replace("\n", " ").strip()
        label = str(row.get(label_col, ""))
        choices.append((f"Row {idx} · {label} · {text_preview}", int(idx)))
    labels = sorted({str(v) for v in df[label_col].dropna().tolist() if str(v).strip()})
    return choices, labels


def preview_text_row(data_path: str | Path, row_index: int, text_col: str, label_col: str) -> tuple[str, str]:
    df = read_structured_file(data_path)
    if int(row_index) not in df.index:
        raise ValueError(f"Row {row_index} was not found in the dataset.")
    row = df.loc[int(row_index)]
    text = str(row.get(text_col, ""))
    current_label = str(row.get(label_col, ""))
    return text, current_label


def save_text_relabel(data_path: str | Path, row_index: int, label_col: str, new_label: str) -> str:
    path = Path(data_path).expanduser()
    df = read_structured_file(path)
    if int(row_index) not in df.index:
        raise ValueError(f"Row {row_index} was not found in the dataset.")
    clean_label = (new_label or "").strip()
    if not clean_label:
        raise ValueError("Provide a new label before saving.")
    df.loc[int(row_index), label_col] = clean_label
    suffix = path.suffix.lower()
    if suffix == ".csv":
        df.to_csv(path, index=False)
    elif suffix == ".tsv":
        df.to_csv(path, index=False, sep="\t")
    elif suffix == ".json":
        df.to_json(path, orient="records", indent=2)
    else:
        raise ValueError(f"Unsupported structured file type: {path.suffix}")
    return clean_label


def format_text_annotation_preview(text: str, current_label: str, row_index: int) -> str:
    escaped = (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )
    return (
        "### Text sample\n\n"
        f"- **Row**: `{row_index}`\n"
        f"- **Current label**: `{current_label or 'unlabeled'}`\n\n"
        f"> {escaped}"
    )


def review_object_boxes(image_path: str, model_key: str, conf: float, iou: float):
    from detection.yolo_detector import review_image_detections

    annotated_rgb, rows, markdown = review_image_detections(image_path, model_key, conf=conf, iou=iou)
    state = {
        "image_path": image_path,
        "rows": rows,
    }
    return annotated_rgb, rows, markdown, state


def save_object_box_review(image_path: str, rows: Any, output_dir: str | Path | None = None) -> str:
    import pandas as pd

    path = Path(image_path).expanduser()
    if not path.exists():
        raise ValueError(f"Image not found: {path}")
    if output_dir is None:
        output_root = Path("outputs") / "object_box_reviews"
    else:
        output_root = Path(output_dir).expanduser()
    output_root.mkdir(parents=True, exist_ok=True)

    if isinstance(rows, pd.DataFrame):
        df = rows.copy()
    else:
        df = pd.DataFrame(rows or [])
    if df.empty:
        raise ValueError("No reviewed boxes were available to save.")
    if "keep" in df.columns:
        df = df[df["keep"].fillna(True).astype(bool)]
    if df.empty:
        raise ValueError("All boxes were marked as skipped. Keep at least one box to save an annotation file.")

    df = df.reset_index(drop=True)
    class_names = []
    for label in df["label"].astype(str).tolist():
        if label not in class_names:
            class_names.append(label)
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}

    from PIL import Image

    with Image.open(path) as img:
        width, height = img.size

    txt_path = output_root / f"{path.stem}.txt"
    classes_path = output_root / f"{path.stem}.classes.txt"
    json_path = output_root / f"{path.stem}.review.json"

    yolo_lines = []
    for _, row in df.iterrows():
        x1, y1, x2, y2 = float(row["x1"]), float(row["y1"]), float(row["x2"]), float(row["y2"])
        x_center = ((x1 + x2) / 2.0) / max(width, 1)
        y_center = ((y1 + y2) / 2.0) / max(height, 1)
        box_w = max(x2 - x1, 0.0) / max(width, 1)
        box_h = max(y2 - y1, 0.0) / max(height, 1)
        cls_idx = class_to_idx[str(row["label"])]
        yolo_lines.append(f"{cls_idx} {x_center:.6f} {y_center:.6f} {box_w:.6f} {box_h:.6f}")

    txt_path.write_text("\n".join(yolo_lines) + "\n")
    classes_path.write_text("\n".join(class_names) + "\n")
    json_path.write_text(json.dumps(df.to_dict(orient="records"), indent=2))
    return str(txt_path.resolve())

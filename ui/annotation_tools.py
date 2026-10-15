from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import shutil
from pathlib import Path
from typing import Any

from data_pipeline.io_utils import read_structured_file


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}
AUDIO_SUFFIXES = {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".aac"}

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_CSV_ENCODINGS = ("utf-8", "utf-8-sig", "cp1252", "latin-1")  # same order as read_structured_file


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


def _validate_folder_label(label: str) -> str:
    """A class label becomes a folder name directly under the dataset root."""
    clean = (label or "").strip()
    if not clean:
        raise ValueError("Provide a new label before saving.")
    if clean in {".", ".."} or clean.startswith(".") or any(c in clean for c in "/\\:\0"):
        raise ValueError(
            f"'{clean}' can't be used as a class label: labels become folder names, so they "
            "can't contain /, \\, : or start with a dot."
        )
    return clean


def relabel_folder_sample(dataset_dir: str | Path, rel_path: str, new_label: str) -> tuple[str, str]:
    root = Path(dataset_dir).expanduser().resolve()
    sample_path = (root / rel_path).resolve()
    if not sample_path.is_relative_to(root):
        raise ValueError(f"Sample path is outside the dataset folder: {rel_path}")
    if not sample_path.exists():
        raise ValueError(f"Sample not found: {sample_path}")
    clean_label = _validate_folder_label(new_label)
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


# ---------------------------------------------------------------------------
# Safe in-place relabelling
# ---------------------------------------------------------------------------

def backup_path_for(path: Path) -> Path:
    """Where the untouched original is kept: data.csv -> data.original.csv."""
    return path.with_name(f"{path.stem}.original{path.suffix}")


def _keep_original(path: Path) -> None:
    backup = backup_path_for(path)
    if not backup.exists():
        shutil.copy2(path, backup)


def _atomic_write(path: Path, data: str, encoding: str) -> None:
    tmp = path.with_name(f".{path.name}.tmp")
    with open(tmp, "w", encoding=encoding, newline="") as fh:
        fh.write(data)
    os.replace(tmp, path)


def _relabel_delimited(path: Path, row_index: int, label_col: str, label: str, delimiter: str) -> None:
    """Change one cell, leaving every other cell exactly as written.

    Rows are numbered like pandas numbers them: data rows after the header,
    skipping blank lines.
    """
    for encoding in _CSV_ENCODINGS:
        try:
            raw = path.read_bytes().decode(encoding)
            break
        except UnicodeDecodeError:
            continue
    else:
        raise ValueError(f"Could not decode {path.name} as text.")
    if encoding == "utf-8-sig" or raw.startswith("﻿"):
        encoding, raw = "utf-8-sig", raw.lstrip("﻿")

    rows = list(csv.reader(io.StringIO(raw, newline=""), delimiter=delimiter))
    if not rows or label_col not in rows[0]:
        raise ValueError(f"Label column '{label_col}' was not found.")
    col = rows[0].index(label_col)
    data_rows = [i for i, r in enumerate(rows) if i > 0 and any(cell.strip() for cell in r)]
    if not 0 <= row_index < len(data_rows):
        raise ValueError(f"Row {row_index} was not found in the dataset.")
    target = rows[data_rows[row_index]]
    target.extend([""] * (len(rows[0]) - len(target)))
    target[col] = label

    out = io.StringIO(newline="")
    line_end = "\r\n" if "\r\n" in raw else "\n"
    csv.writer(out, delimiter=delimiter, lineterminator=line_end).writerows(rows)
    _atomic_write(path, out.getvalue(), encoding)


def _relabel_json(path: Path, row_index: int, label_col: str, label: str) -> None:
    """Update one record in a JSON-lines file or a JSON array, keeping the format."""
    raw = path.read_text(encoding="utf-8")
    lines = [ln for ln in raw.splitlines() if ln.strip()]
    try:
        records = [json.loads(ln) for ln in lines]
        is_lines = len(records) != 1 or not isinstance(records[0], list)
    except json.JSONDecodeError:
        records, is_lines = None, False
    if not is_lines:
        records = json.loads(raw)
    if not isinstance(records, list) or not 0 <= row_index < len(records):
        raise ValueError(f"Row {row_index} was not found in the dataset.")
    records[row_index][label_col] = label
    if is_lines:
        data = "\n".join(json.dumps(r, ensure_ascii=False) for r in records) + "\n"
    else:
        data = json.dumps(records, indent=2, ensure_ascii=False) + "\n"
    _atomic_write(path, data, "utf-8")


def save_text_relabel(data_path: str | Path, row_index: int, label_col: str, new_label: str) -> str:
    """Change one row's label in the source file.

    Only that cell changes: the file is not round-tripped through pandas,
    which would alter other values (leading zeros, number formatting, JSON
    layout). The untouched original is kept next to it as
    ``<name>.original<ext>`` the first time it's edited, and every write is
    atomic, so an interrupted save can't truncate the dataset.
    """
    path = Path(data_path).expanduser()
    clean_label = (new_label or "").strip()
    if not clean_label:
        raise ValueError("Provide a new label before saving.")
    suffix = path.suffix.lower()
    if suffix not in {".csv", ".tsv", ".json"}:
        raise ValueError(f"Unsupported structured file type: {path.suffix}")
    _keep_original(path)
    if suffix == ".json":
        _relabel_json(path, int(row_index), label_col, clean_label)
    else:
        _relabel_delimited(path, int(row_index), label_col, clean_label, "," if suffix == ".csv" else "\t")
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


# ---------------------------------------------------------------------------
# Object-box review (YOLO format)
# ---------------------------------------------------------------------------

def review_object_boxes(image_path: str, model_key: str, conf: float, iou: float):
    from detection.yolo_detector import review_image_detections

    annotated_rgb, rows, markdown = review_image_detections(image_path, model_key, conf=conf, iou=iou)
    state = {
        "image_path": image_path,
        "rows": rows,
    }
    return annotated_rgb, rows, markdown, state


def _shared_class_index(output_root: Path, labels: list[str]) -> dict[str, int]:
    """Class ids shared by every label file in ``output_root``.

    YOLO datasets need one class list per dataset. Numbering classes per image
    made "dog" 0 in one file and 1 in the next, silently corrupting any
    dataset built from the reviews. New labels are appended, so existing
    ids never change.
    """
    classes_path = output_root / "classes.txt"
    classes = [c for c in classes_path.read_text(encoding="utf-8").splitlines() if c.strip()] \
        if classes_path.exists() else []
    for label in labels:
        if label not in classes:
            classes.append(label)
    _atomic_write(classes_path, "\n".join(classes) + "\n", "utf-8")
    return {name: idx for idx, name in enumerate(classes)}


def save_object_box_review(image_path: str, rows: Any, output_dir: str | Path | None = None) -> str:
    import pandas as pd

    path = Path(image_path).expanduser().resolve()
    if not path.exists():
        raise ValueError(f"Image not found: {path}")
    # Anchored to the project, not the working directory the app started in.
    output_root = Path(output_dir).expanduser() if output_dir is not None \
        else _PROJECT_ROOT / "outputs" / "object_box_reviews"
    output_root.mkdir(parents=True, exist_ok=True)

    df = rows.copy() if isinstance(rows, pd.DataFrame) else pd.DataFrame(rows or [])
    if df.empty:
        raise ValueError("No reviewed boxes were available to save.")
    if "keep" in df.columns:
        df = df[df["keep"].fillna(True).astype(bool)]
    if df.empty:
        raise ValueError("All boxes were marked as skipped. Keep at least one box to save an annotation file.")

    df = df.reset_index(drop=True)
    class_to_idx = _shared_class_index(output_root, df["label"].astype(str).tolist())

    from PIL import Image

    with Image.open(path) as img:
        width, height = img.size

    # Same-named images from different folders get different files.
    name = f"{path.stem}-{hashlib.sha1(str(path).encode('utf-8')).hexdigest()[:8]}"
    txt_path = output_root / f"{name}.txt"
    json_path = output_root / f"{name}.review.json"

    def _unit(value: float) -> float:
        return min(max(value, 0.0), 1.0)

    yolo_lines = []
    for _, row in df.iterrows():
        x1, y1, x2, y2 = float(row["x1"]), float(row["y1"]), float(row["x2"]), float(row["y2"])
        x_center = _unit(((x1 + x2) / 2.0) / max(width, 1))
        y_center = _unit(((y1 + y2) / 2.0) / max(height, 1))
        box_w = _unit(max(x2 - x1, 0.0) / max(width, 1))
        box_h = _unit(max(y2 - y1, 0.0) / max(height, 1))
        cls_idx = class_to_idx[str(row["label"])]
        yolo_lines.append(f"{cls_idx} {x_center:.6f} {y_center:.6f} {box_w:.6f} {box_h:.6f}")

    _atomic_write(txt_path, "\n".join(yolo_lines) + "\n", "utf-8")
    review = {"image_path": str(path), "boxes": df.to_dict(orient="records")}
    _atomic_write(json_path, json.dumps(review, indent=2, default=str), "utf-8")
    return str(txt_path.resolve())

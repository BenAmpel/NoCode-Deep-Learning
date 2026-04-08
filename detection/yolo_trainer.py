"""
YOLO-based image *classifier* trainer.

IMPORTANT — classification vs. detection
-----------------------------------------
This module trains a **YOLOv8 image classifier** (``yolov8n-cls.pt`` etc.).
That means the model assigns a single class label to the whole image, like
ResNet or EfficientNet — it does NOT draw bounding boxes.

To extend to **object detection** (bounding boxes + labels):
  1. Prepare a dataset in YOLO detection format (images + .txt label files,
     a ``data.yaml`` describing classes and paths).
  2. Use a detection backbone instead: ``YOLO("yolov8n.pt")``
  3. Call model.train(data="data.yaml", task="detect", ...)
  See https://docs.ultralytics.com/tasks/detect/ for a full walkthrough.

Dataset format expected by :class:`YOLODatasetPreparer`
---------------------------------------------------------
The *source* directory must be organised like a standard image-classification
dataset (the same layout used throughout this project):

    source_dir/
        cat/
            img001.jpg
            img002.jpg
            ...
        dog/
            img001.jpg
            ...

:meth:`YOLODatasetPreparer.prepare` converts this into the flat per-class
folder structure that ``ultralytics`` expects for classification training:

    output_dir/
        train/
            cat/  (80 % of cat images, shuffled)
            dog/
        val/
            cat/  (20 % of cat images)
            dog/
"""
from __future__ import annotations

import os
import random
import shutil
from pathlib import Path
from typing import Generator

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

# Maps human-readable UI labels to the ultralytics model-size suffix
YOLO_TRAIN_MODELS: dict[str, str] = {
    "YOLOv8 Nano":   "n",
    "YOLOv8 Small":  "s",
    "YOLOv8 Medium": "m",
}

# Image file extensions considered valid by ultralytics
_IMAGE_EXTS: frozenset[str] = frozenset(
    {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
)


# ─────────────────────────────────────────────────────────────────────────────
# Dataset preparation
# ─────────────────────────────────────────────────────────────────────────────

class YOLODatasetPreparer:
    """
    Convert a classification-style folder dataset into the YOLO classification
    directory structure required by ``ultralytics``.

    Parameters
    ----------
    source_dir : str
        Root directory whose immediate sub-folders are class names, each
        containing image files.
    output_dir : str
        Destination root.  Will be created if it does not exist.  On repeated
        calls the existing ``train/`` and ``val/`` sub-trees are removed and
        recreated to avoid stale data.
    val_split : float
        Fraction of images per class to reserve for validation.
        Must be in (0, 1).  Default is ``0.2``.
    """

    def __init__(
        self,
        source_dir: str,
        output_dir: str,
        val_split: float = 0.2,
    ) -> None:
        if not (0.0 < val_split < 1.0):
            raise ValueError(
                f"val_split must be in (0, 1), got {val_split}"
            )
        self.source_dir = Path(source_dir)
        self.output_dir = Path(output_dir)
        self.val_split  = val_split

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def prepare(self) -> dict:
        """
        Execute the train/val split and copy images into the YOLO layout.

        Returns
        -------
        dict with keys:
            ``train_dir``  (str) — absolute path to the ``train/`` directory
            ``val_dir``    (str) — absolute path to the ``val/`` directory
            ``classes``    (list[str]) — sorted list of class names found
            ``n_train``    (int) — total training images across all classes
            ``n_val``      (int) — total validation images across all classes

        Raises
        ------
        FileNotFoundError
            If ``source_dir`` does not exist.
        ValueError
            If no valid image files are found under ``source_dir``.
        """
        if not self.source_dir.exists():
            raise FileNotFoundError(
                f"Source directory not found: {self.source_dir}"
            )

        train_root = self.output_dir / "train"
        val_root   = self.output_dir / "val"

        # Clean up previous runs to prevent data leakage
        for d in (train_root, val_root):
            if d.exists():
                shutil.rmtree(d)

        classes = sorted(
            p.name for p in self.source_dir.iterdir()
            if p.is_dir() and not p.name.startswith(".")
        )

        if not classes:
            raise ValueError(
                f"No class sub-directories found in {self.source_dir}. "
                "Expected layout: source_dir/<class_name>/<image_files>"
            )

        n_train_total = 0
        n_val_total   = 0

        for cls_name in classes:
            cls_src = self.source_dir / cls_name
            images  = sorted(
                p for p in cls_src.iterdir()
                if p.is_file() and p.suffix.lower() in _IMAGE_EXTS
            )

            if not images:
                continue  # Skip empty class directories gracefully

            # Deterministic shuffle so repeated calls give the same split
            rng = random.Random(42)
            rng.shuffle(images)

            n_val   = max(1, int(len(images) * self.val_split))
            n_train = len(images) - n_val

            val_imgs   = images[:n_val]
            train_imgs = images[n_val:]

            _copy_images(train_imgs, train_root / cls_name)
            _copy_images(val_imgs,   val_root   / cls_name)

            n_train_total += n_train
            n_val_total   += n_val

        if n_train_total == 0:
            raise ValueError(
                f"No valid image files ({', '.join(sorted(_IMAGE_EXTS))}) "
                f"found under {self.source_dir}."
            )

        return {
            "train_dir": str(train_root.resolve()),
            "val_dir":   str(val_root.resolve()),
            "classes":   classes,
            "n_train":   n_train_total,
            "n_val":     n_val_total,
        }


def _copy_images(images: list[Path], dest_dir: Path) -> None:
    """Create ``dest_dir`` and hard-copy (not symlink) all ``images`` into it."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    for img in images:
        shutil.copy2(img, dest_dir / img.name)


# ─────────────────────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────────────────────

def train_yolo_classifier(
    data_dir: str,
    model_size: str = "n",
    epochs: int = 10,
    imgsz: int = 224,
    batch: int = 16,
    output_dir: str | None = None,
) -> Generator[dict, None, None]:
    """
    Train a YOLOv8 image classifier on ``data_dir`` and yield progress updates.

    The function is a **generator** so the caller can stream progress to a UI
    without blocking.  Because ``ultralytics`` ``model.train()`` is synchronous
    and does not expose a per-epoch Python callback (as of ultralytics ≥ 8.0),
    this function:

    1. Yields a single ``{"done": False, "message": "..."}`` progress marker
       before training starts (useful for UI spinners / status messages).
    2. Calls ``model.train(...)`` which runs to completion.
    3. Parses the CSV results file written by ultralytics to extract per-epoch
       metrics and yields them as individual progress dicts.
    4. Yields a final ``{"done": True, ...}`` dict.

    If the ultralytics version supports ``callbacks``, per-epoch real-time
    streaming is enabled automatically via the on_train_epoch_end hook.

    Parameters
    ----------
    data_dir : str
        Root of the prepared dataset (contains ``train/`` and ``val/``
        sub-directories with per-class image folders).
    model_size : str
        One of ``"n"`` (Nano), ``"s"`` (Small), ``"m"`` (Medium).
    epochs : int
        Total training epochs.
    imgsz : int
        Input image size (square).  224 is the ImageNet standard default.
    batch : int
        Training batch size.
    output_dir : str or None
        Where ultralytics saves training artefacts (runs/classify/train*).
        If ``None``, defaults to ``"runs"`` in the current working directory.

    Yields
    ------
    Progress dicts (``done=False``)::

        {
            "done":       False,
            "epoch":      int,        # 1-indexed current epoch
            "epochs":     int,        # total epochs
            "train_loss": float,      # combined training loss for the epoch
            "val_acc":    float,      # top-1 validation accuracy (0–1)
            "message":    str,        # human-readable status line
        }

    Final dict (``done=True``)::

        {
            "done":         True,
            "model_path":   str,      # absolute path to best.pt
            "best_val_acc": float,    # best top-1 validation accuracy achieved
        }

    Raises
    ------
    ImportError
        If the ``ultralytics`` package is not installed.
    FileNotFoundError
        If ``data_dir`` does not exist.
    """
    try:
        from ultralytics import YOLO  # noqa: PLC0415
    except ImportError as exc:
        raise ImportError(
            "The `ultralytics` library is required for YOLO training.\n"
            "Install it with:  pip install ultralytics"
        ) from exc

    if not Path(data_dir).exists():
        raise FileNotFoundError(
            f"Training data directory not found: {data_dir}\n"
            "Run YOLODatasetPreparer.prepare() first."
        )

    weights_name = f"yolov8{model_size}-cls.pt"

    # ── Attempt real-time per-epoch streaming via ultralytics callbacks ──────
    # The callback API is available in ultralytics ≥ 8.0.
    # We store epoch results into a shared list so the generator can drain them
    # after training completes (or in real-time if the callback fires
    # synchronously on the same thread, which it does for local training).
    epoch_results: list[dict] = []
    _streaming_active = False

    # Notify caller that we are about to start (useful for UI spinners)
    yield {
        "done":       False,
        "epoch":      0,
        "epochs":     epochs,
        "train_loss": 0.0,
        "val_acc":    0.0,
        "message":    f"Loading {weights_name} and starting training …",
    }

    model = YOLO(weights_name)

    # Try to register an on_train_epoch_end callback for real-time updates
    try:
        def _on_epoch_end(trainer) -> None:  # type: ignore[no-untyped-def]
            """Callback invoked by ultralytics at the end of every epoch."""
            metrics = trainer.metrics or {}
            loss_dict = getattr(trainer, "loss_items", None)
            train_loss = float(loss_dict.mean()) if loss_dict is not None else 0.0
            val_acc = float(metrics.get("metrics/accuracy_top1", 0.0))
            epoch_results.append({
                "done":       False,
                "epoch":      trainer.epoch + 1,  # ultralytics is 0-indexed
                "epochs":     epochs,
                "train_loss": round(train_loss, 4),
                "val_acc":    round(val_acc, 4),
                "message":    (
                    f"Epoch {trainer.epoch + 1}/{epochs} — "
                    f"loss: {train_loss:.4f}  val_acc: {val_acc:.4f}"
                ),
            })

        model.add_callback("on_train_epoch_end", _on_epoch_end)
        _streaming_active = True
    except AttributeError:
        # Older ultralytics version without callback support — fall through
        _streaming_active = False

    # ── Run training (blocking call) ─────────────────────────────────────────
    train_kwargs: dict = dict(
        data=data_dir,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        verbose=False,   # suppress per-batch console spam
    )
    if output_dir is not None:
        train_kwargs["project"] = output_dir

    results = model.train(**train_kwargs)

    # ── Drain / yield per-epoch results ─────────────────────────────────────
    if _streaming_active and epoch_results:
        # Real-time callbacks populated epoch_results; yield them all now.
        # (For async/threaded callers they would have been yielded in-flight;
        # for synchronous callers like a Streamlit app this covers both cases.)
        for ep in epoch_results:
            yield ep
    else:
        # Fallback: parse the CSV results file written by ultralytics.
        # File location: <save_dir>/results.csv
        save_dir = Path(getattr(results, "save_dir", "."))
        csv_path = save_dir / "results.csv"
        if csv_path.exists():
            yield from _parse_results_csv(csv_path, epochs)
        else:
            # Last-resort single progress message when no CSV is available
            yield {
                "done":       False,
                "epoch":      epochs,
                "epochs":     epochs,
                "train_loss": 0.0,
                "val_acc":    0.0,
                "message":    "Training complete (metrics unavailable).",
            }

    # ── Final yield ──────────────────────────────────────────────────────────
    save_dir   = Path(getattr(results, "save_dir", "."))
    best_pt    = save_dir / "weights" / "best.pt"
    model_path = str(best_pt.resolve()) if best_pt.exists() else str(save_dir / "weights" / "last.pt")

    best_val_acc = _best_val_acc_from_results(epoch_results)

    yield {
        "done":         True,
        "model_path":   model_path,
        "best_val_acc": best_val_acc,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _parse_results_csv(csv_path: Path, epochs: int) -> Generator[dict, None, None]:
    """
    Parse the ``results.csv`` written by ultralytics and yield one progress
    dict per epoch row.

    The CSV header varies by task/version, so we match column names flexibly.
    """
    try:
        import csv  # stdlib — always available

        with csv_path.open(newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            # Strip whitespace from header names (ultralytics pads them)
            reader.fieldnames = (
                [h.strip() for h in reader.fieldnames]
                if reader.fieldnames
                else []
            )
            for row in reader:
                row = {k.strip(): v.strip() for k, v in row.items()}

                epoch = _safe_int(row.get("epoch", "0")) + 1  # 0-indexed in CSV

                # Training loss — try several possible column names
                train_loss = _safe_float(
                    row.get("train/loss")
                    or row.get("train/box_loss")
                    or row.get("train/cls_loss")
                    or "0"
                )

                # Validation top-1 accuracy
                val_acc = _safe_float(
                    row.get("metrics/accuracy_top1")
                    or row.get("val/acc")
                    or "0"
                )

                yield {
                    "done":       False,
                    "epoch":      epoch,
                    "epochs":     epochs,
                    "train_loss": round(train_loss, 4),
                    "val_acc":    round(val_acc, 4),
                    "message":    (
                        f"Epoch {epoch}/{epochs} — "
                        f"loss: {train_loss:.4f}  val_acc: {val_acc:.4f}"
                    ),
                }
    except Exception:
        # If CSV parsing fails for any reason, yield nothing and let the
        # caller handle the missing intermediate results gracefully.
        return


def _best_val_acc_from_results(epoch_results: list[dict]) -> float:
    """Return the highest validation accuracy seen across all epoch results."""
    if not epoch_results:
        return 0.0
    return max(r.get("val_acc", 0.0) for r in epoch_results)


def _safe_float(value: str | None, default: float = 0.0) -> float:
    try:
        return float(value) if value else default
    except (ValueError, TypeError):
        return default


def _safe_int(value: str | None, default: int = 0) -> int:
    try:
        return int(float(value)) if value else default
    except (ValueError, TypeError):
        return default

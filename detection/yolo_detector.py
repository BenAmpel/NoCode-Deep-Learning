"""
YOLO-based object detection — processes images and videos using YOLOv8.
Weights are downloaded automatically on first use via the ultralytics package.
"""
from __future__ import annotations

import os
import time
from collections import defaultdict
from pathlib import Path
from typing import Generator

import cv2
import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# Available models
# ─────────────────────────────────────────────────────────────────────────────

YOLO_MODELS: dict[str, str] = {
    "YOLOv8 Nano  (~6 MB  · fastest)":   "yolov8n.pt",
    "YOLOv8 Small (~22 MB · balanced)":  "yolov8s.pt",
    "YOLOv8 Medium (~52 MB · accurate)": "yolov8m.pt",
}

# Deterministic per-class BGR colours (80 COCO classes)
_PALETTE = np.random.default_rng(42).integers(60, 240, (80, 3), dtype=np.uint8)


# ─────────────────────────────────────────────────────────────────────────────
# Model loader (cached)
# ─────────────────────────────────────────────────────────────────────────────

_model_cache: dict[str, object] = {}


def _get_model(model_key: str):
    """Lazy-load and cache a YOLO model; downloads weights on first call."""
    if model_key in _model_cache:
        return _model_cache[model_key]
    if model_key not in YOLO_MODELS:
        available = ", ".join(YOLO_MODELS.keys())
        raise ValueError(
            f"Unknown YOLO model '{model_key}'. Available options: {available}"
        )
    try:
        from ultralytics import YOLO
    except ImportError:
        raise ImportError(
            "ultralytics is not installed.\n"
            "Install it with:  pip install ultralytics"
        )
    weights = YOLO_MODELS[model_key]
    model = YOLO(weights)
    _model_cache[model_key] = model
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Image detection
# ─────────────────────────────────────────────────────────────────────────────

def run_image_detection(
    image_path: str,
    model_key: str,
    conf: float = 0.25,
    iou: float  = 0.45,
) -> tuple[np.ndarray, dict, int]:
    """
    Run YOLO on a single image.

    Returns
    -------
    annotated_bgr : np.ndarray   BGR image with bounding boxes drawn
    stats         : dict         {class_name: {"count": int, "avg_conf": float}}
    n_detections  : int          total number of boxes
    """
    model   = _get_model(model_key)
    results = model(image_path, conf=conf, iou=iou, verbose=False)[0]
    annotated = results.plot()          # returns BGR ndarray
    stats, n  = _compute_stats([results])
    return annotated, stats, n


def review_image_detections(
    image_path: str,
    model_key: str,
    conf: float = 0.25,
    iou: float = 0.45,
) -> tuple[np.ndarray, list[dict], str]:
    """Run detection on one image and return editable box rows for review."""
    model = _get_model(model_key)
    results = model(image_path, conf=conf, iou=iou, verbose=False)[0]
    annotated = results.plot()[:, :, ::-1]
    rows: list[dict] = []
    if results.boxes is not None:
        for idx, box in enumerate(results.boxes):
            cls_id = int(box.cls[0])
            cls_name = results.names.get(cls_id, str(cls_id))
            conf_val = float(box.conf[0])
            x1, y1, x2, y2 = [float(v) for v in box.xyxy[0].tolist()]
            rows.append(
                {
                    "keep": True,
                    "label": cls_name,
                    "confidence": round(conf_val, 3),
                    "x1": round(x1, 1),
                    "y1": round(y1, 1),
                    "x2": round(x2, 1),
                    "y2": round(y2, 1),
                }
            )
    markdown = (
        "### Object box review\n\n"
        f"- **Detected boxes**: `{len(rows)}`\n"
        "- Edit labels or coordinates in the table if needed, then save the review as YOLO annotations."
    )
    return annotated, rows, markdown


# ─────────────────────────────────────────────────────────────────────────────
# H.264 re-encoding for browser playback
# ─────────────────────────────────────────────────────────────────────────────

def _find_ffmpeg() -> str | None:
    """Return the ffmpeg binary path, or None if not found."""
    import shutil
    found = shutil.which("ffmpeg")
    if found:
        return found
    # Check common Homebrew / system locations
    for candidate in [
        "/opt/homebrew/bin/ffmpeg",
        "/usr/local/bin/ffmpeg",
        "/usr/bin/ffmpeg",
    ]:
        if os.path.isfile(candidate):
            return candidate
    return None


def _reencode_h264(mp4v_path: str) -> str:
    """Re-encode an mp4v file to H.264 so browsers can play it.

    Returns the path to the H.264 file (replaces the original).
    Falls back to the original path if ffmpeg is unavailable.
    """
    ffmpeg = _find_ffmpeg()
    if not ffmpeg:
        return mp4v_path

    import subprocess
    h264_path = mp4v_path.replace(".mp4", "_h264.mp4")
    try:
        subprocess.run(
            [
                ffmpeg, "-y",
                "-i", mp4v_path,
                "-c:v", "libx264",
                "-preset", "fast",
                "-crf", "23",
                "-pix_fmt", "yuv420p",
                "-movflags", "+faststart",
                h264_path,
            ],
            check=True,
            capture_output=True,
        )
        # Replace original with H.264 version
        os.replace(h264_path, mp4v_path)
        return mp4v_path
    except Exception:
        # If re-encoding fails, return whatever we have
        if os.path.isfile(h264_path):
            return h264_path
        return mp4v_path


# ─────────────────────────────────────────────────────────────────────────────
# Video detection  (generator — yields progress then final result)
# ─────────────────────────────────────────────────────────────────────────────

def run_video_detection(
    video_path: str,
    model_key:  str,
    conf:       float = 0.25,
    iou:        float = 0.45,
    output_dir: str | None = None,
) -> Generator[dict, None, None]:
    """
    Generator that processes a video frame-by-frame.

    Progress yields
    ---------------
    {"done": False, "frame": int, "total": int,
     "progress": float, "eta_seconds": float, "fps_proc": float}

    Final yield
    -----------
    {"done": True, "output_path": str, "stats": dict,
     "total_frames": int, "n_detections": int}
    """
    model = _get_model(model_key)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps          = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Build output path
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    else:
        output_dir = str(Path(video_path).parent)

    stem       = Path(video_path).stem
    out_path   = os.path.join(output_dir, f"{stem}_detected.mp4")
    fourcc     = cv2.VideoWriter_fourcc(*"mp4v")
    writer     = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

    counts      = defaultdict(int)
    conf_sum    = defaultdict(float)
    frame_idx   = 0
    t0          = time.time()

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        results   = model(frame, conf=conf, iou=iou, verbose=False)[0]
        annotated = results.plot()
        writer.write(annotated)
        _update_stats(results, counts, conf_sum)
        frame_idx += 1

        elapsed  = time.time() - t0
        fps_proc = round(frame_idx / max(elapsed, 0.01), 1)
        eta      = (elapsed / frame_idx) * (total_frames - frame_idx) if frame_idx else 0

        yield {
            "done":        False,
            "frame":       frame_idx,
            "total":       total_frames,
            "progress":    frame_idx / max(total_frames, 1),
            "eta_seconds": round(eta, 1),
            "fps_proc":    fps_proc,
        }

    cap.release()
    writer.release()

    # Re-encode with H.264 so browsers can play the video.
    # OpenCV's mp4v codec is not browser-compatible.
    out_path = _reencode_h264(out_path)

    stats, n_det = _finalize_stats(counts, conf_sum)

    yield {
        "done":         True,
        "output_path":  out_path,
        "stats":        stats,
        "total_frames": frame_idx,
        "n_detections": n_det,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Stats helpers
# ─────────────────────────────────────────────────────────────────────────────

def _update_stats(result, counts, conf_sum) -> None:
    if result.boxes is None:
        return
    for box in result.boxes:
        cls_id   = int(box.cls[0])
        cls_name = result.names.get(cls_id, str(cls_id))
        conf_val = float(box.conf[0])
        counts[cls_name]   += 1
        conf_sum[cls_name] += conf_val


def _finalize_stats(counts, conf_sum) -> tuple[dict, int]:
    stats     = {}
    total     = 0
    for name, count in sorted(counts.items(), key=lambda x: -x[1]):
        stats[name] = {
            "count":    count,
            "avg_conf": round(conf_sum[name] / count, 3),
        }
        total += count

    return stats, total


def _compute_stats(results_list) -> tuple[dict, int]:
    """
    Aggregate per-class detection counts and mean confidence.

    Returns (stats_dict, total_count).
    stats_dict: {class_name: {"count": int, "avg_conf": float}}
    """
    counts   = defaultdict(int)
    conf_sum = defaultdict(float)

    for r in results_list:
        _update_stats(r, counts, conf_sum)

    return _finalize_stats(counts, conf_sum)


def stats_to_markdown(stats: dict, total_frames: int = 1, n_detections: int = 0) -> str:
    """Render detection stats as a Markdown table."""
    if not stats:
        return "⚠️ No objects detected above the confidence threshold."

    lines = [
        f"### Detection Summary",
        f"**Total detections:** {n_detections}  |  "
        f"**Unique classes:** {len(stats)}  |  "
        f"**Frames processed:** {total_frames}",
        "",
        "| Class | Count | Avg Confidence |",
        "|:------|------:|---------------:|",
    ]
    for name, s in stats.items():
        bar   = "█" * int(s["avg_conf"] * 10)
        lines.append(f"| **{name}** | {s['count']} | {s['avg_conf']:.1%} `{bar}` |")

    return "\n".join(lines)


def make_summary_chart(stats: dict) -> "plt.Figure | None":
    """Bar chart of detection counts per class, sorted descending."""
    if not stats:
        return None
    try:
        import matplotlib.pyplot as plt
        names  = list(stats.keys())
        counts = [stats[n]["count"] for n in names]
        confs  = [stats[n]["avg_conf"] for n in names]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, max(3, len(names) * 0.4 + 1)))

        # Detection counts (horizontal bar)
        y_pos = range(len(names))
        ax1.barh(y_pos, counts, color="#4C72B0", edgecolor="white", height=0.6)
        ax1.set_yticks(y_pos)
        ax1.set_yticklabels(names, fontsize=9)
        ax1.invert_yaxis()
        ax1.set_xlabel("Detection count")
        ax1.set_title("Detections per Class")
        for i, v in enumerate(counts):
            ax1.text(v + 0.1, i, str(v), va="center", fontsize=8)

        # Average confidence (horizontal bar)
        colours = ["#2ca02c" if c >= 0.7 else "#ff7f0e" if c >= 0.5 else "#d62728"
                   for c in confs]
        ax2.barh(y_pos, confs, color=colours, edgecolor="white", height=0.6)
        ax2.set_yticks(y_pos)
        ax2.set_yticklabels(names, fontsize=9)
        ax2.invert_yaxis()
        ax2.set_xlim(0, 1)
        ax2.set_xlabel("Average confidence")
        ax2.set_title("Confidence per Class")
        for i, v in enumerate(confs):
            ax2.text(v + 0.01, i, f"{v:.0%}", va="center", fontsize=8)

        fig.tight_layout()
        return fig
    except Exception:
        return None

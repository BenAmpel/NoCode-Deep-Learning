"""Object Detection tab: YOLO on images and video, and custom training."""
from __future__ import annotations

import logging, os, re
from pathlib import Path

from ui.helpers import _OUTPUTS_ROOT
from ui.layout import Components

logger = logging.getLogger("nocode_dl")


# ── Object Detection callbacks ────────────────────────────────────────────

def detect_image(img_path, model_key, conf, iou):
    if img_path is None:
        return None, "*No image uploaded.*", None
    try:
        from detection.yolo_detector import (
            run_image_detection, stats_to_markdown, make_summary_chart)
        annotated_bgr, stats, n_det = run_image_detection(
            img_path, model_key, conf=conf, iou=iou)
        annotated_rgb = annotated_bgr[:, :, ::-1]
        md    = stats_to_markdown(stats, total_frames=1, n_detections=n_det)
        chart = make_summary_chart(stats)
        return annotated_rgb, md, chart
    except Exception as e:
        return None, f"❌ Error: {e}", None


def detect_video(vid_path, model_key, conf, iou):
    if vid_path is None:
        yield "No video uploaded.", None, "*Upload a video first.*", None
        return
    try:
        from detection.yolo_detector import (
            run_video_detection, stats_to_markdown, make_summary_chart)
        out_dir = str(_OUTPUTS_ROOT / "detections")
        os.makedirs(out_dir, exist_ok=True)

        # L2: prune annotated output files older than 1 hour to prevent unbounded disk growth
        try:
            import time as _time
            _cutoff = _time.time() - 3600
            for _f in Path(out_dir).glob("*_detected.mp4"):
                if _f.stat().st_mtime < _cutoff:
                    _f.unlink(missing_ok=True)
        except Exception as _e:
            logger.warning("Video cleanup failed: %s", _e)
        for prog in run_video_detection(
            vid_path, model_key, conf=conf, iou=iou, output_dir=out_dir
        ):
            if prog["done"]:
                stats        = prog["stats"]
                total_frames = prog["total_frames"]
                n_det        = prog["n_detections"]
                md    = stats_to_markdown(stats, total_frames=total_frames,
                                          n_detections=n_det)
                chart = make_summary_chart(stats)
                yield (f"✅ Done — {total_frames} frames, {n_det} detections.",
                       prog["output_path"], md, chart)
            else:
                pct = prog["progress"] * 100
                eta = prog["eta_seconds"]
                eta_str = (f"{int(eta)}s" if eta < 60
                           else f"{int(eta/60)}m {int(eta%60)}s")
                yield (f"Frame {prog['frame']}/{prog['total']}  "
                       f"({pct:.0f}%)  ·  {prog['fps_proc']} fps  ·  ETA {eta_str}",
                       None, "", None)
    except Exception as e:
        yield f"❌ Error: {e}", None, f"```\n{e}\n```", None

# ── YOLO custom training ──────────────────────────────────────────────────


def train_yolo_custom(data_dir, model_size_name, n_epochs, batch):
    if not data_dir:
        yield "No dataset folder provided.", "*Enter a dataset path first.*"
        return
    try:
        from detection.yolo_trainer import (
            YOLODatasetPreparer, train_yolo_classifier, YOLO_TRAIN_MODELS)
        log_lines = []
        def log(m):
            log_lines.append(m)
            return "\n".join(log_lines)

        status = log("Preparing dataset…")
        yield status, ""

        # M2: sanitize the basename from user input before using in a path
        _safe_base = re.sub(r"[^a-zA-Z0-9_\-]", "_",
                            os.path.basename(data_dir.rstrip("/"))) or "yolo_dataset"
        prep_out = str(_OUTPUTS_ROOT / "yolo_data" / _safe_base)
        preparer = YOLODatasetPreparer(data_dir, prep_out, val_split=0.2)
        info     = preparer.prepare()
        status   = log(f"✓ {info['n_train']} train / {info['n_val']} val  "
                       f"· Classes: {', '.join(info['classes'])}")
        yield status, ""

        size_key = YOLO_TRAIN_MODELS.get(model_size_name, "n")
        for prog in train_yolo_classifier(
            data_dir=prep_out, model_size=size_key,
            epochs=int(n_epochs), batch=int(batch),
        ):
            if prog["done"]:
                best_acc = prog.get("best_val_acc", 0)
                status   = log(f"\n✅ Training complete!  Best val acc: {best_acc:.1f}%")
                result_md = (
                    f"### YOLO Training Complete\n"
                    f"**Best validation accuracy:** {best_acc:.1f}%\n\n"
                    f"**Model saved to:** `{prog.get('model_path', 'N/A')}`"
                )
                yield status, result_md
            else:
                e   = prog.get("epoch", "?")
                tot = prog.get("epochs", "?")
                acc = prog.get("val_acc", 0)
                status = log(f"Epoch {e}/{tot} — val_acc={acc:.1f}%")
                yield status, ""
    except Exception as ex:
        yield f"❌ Error: {ex}", ""


def register(c: Components) -> dict:
    """Wire this area's events to the components in ``c``.

    Returns its long-running events by name, for the Stop button.
    """
    c.det_img_btn.click(
        detect_image,
        inputs=[c.det_img_input, c.det_model, c.det_conf, c.det_iou],
        outputs=[c.det_img_output, c.det_img_stats_md, c.det_img_chart],
    )

    det_vid_event = c.det_vid_btn.click(
        detect_video,
        inputs=[c.det_vid_input, c.det_model, c.det_conf, c.det_iou],
        outputs=[c.det_vid_status, c.det_vid_output, c.det_vid_stats_md, c.det_vid_chart],
        concurrency_id="heavy_jobs",
        concurrency_limit=1,
    )

    yolo_event = c.yolo_train_btn.click(
        train_yolo_custom,
        inputs=[c.yolo_data_path, c.yolo_model_size, c.yolo_epochs, c.yolo_batch],
        outputs=[c.yolo_status_box, c.yolo_result_md],
        concurrency_id="heavy_jobs",
        concurrency_limit=1,
    )

    return {"det_vid_event": det_vid_event, "yolo_event": yolo_event}

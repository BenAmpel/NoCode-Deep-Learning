"""Annotation Studio tab: relabel images, audio and text, review boxes."""
from __future__ import annotations

from pathlib import Path

import gradio as gr

from ui.layout import Components


# ── Annotation Studio callbacks ──────────────────────────────────────────

def load_image_annotation_samples(dataset_dir):
    if not dataset_dir:
        return gr.update(choices=[], value=None), gr.update(choices=[], value=None), "*Set an image dataset folder first.*"
    try:
        from ui.annotation_tools import IMAGE_SUFFIXES, list_folder_labels, list_folder_samples

        samples = list_folder_samples(dataset_dir, IMAGE_SUFFIXES)
        labels = list_folder_labels(dataset_dir, IMAGE_SUFFIXES)
        if not samples:
            return gr.update(choices=[], value=None), gr.update(choices=labels, value=None), "*No image files were found in class folders.*"
        first_value = samples[0][1]
        return gr.update(choices=samples, value=first_value), gr.update(choices=labels, value=None), f"Loaded `{len(samples)}` image samples across `{len(labels)}` labels."
    except Exception as exc:
        return gr.update(choices=[], value=None), gr.update(choices=[], value=None), f"❌ {exc}"


def preview_image_annotation_sample(dataset_dir, sample_rel):
    if not dataset_dir or not sample_rel:
        return None, "*Choose an image sample to inspect.*", gr.update(value=None)
    try:
        from ui.annotation_tools import IMAGE_SUFFIXES, list_folder_labels, preview_folder_sample

        preview_path, current_label = preview_folder_sample(dataset_dir, sample_rel)
        labels = list_folder_labels(dataset_dir, IMAGE_SUFFIXES)
        meta = (
            "### Image sample\n\n"
            f"- **Current label**: `{current_label}`\n"
            f"- **File**: `{sample_rel}`"
        )
        return preview_path, meta, gr.update(choices=labels, value=current_label)
    except Exception as exc:
        return None, f"❌ {exc}", gr.update()


def save_image_relabel(dataset_dir, sample_rel, new_label):
    if not dataset_dir or not sample_rel:
        return gr.update(), gr.update(), "*Choose an image sample first.*", None, "*Choose an image sample first.*"
    try:
        from ui.annotation_tools import IMAGE_SUFFIXES, list_folder_labels, list_folder_samples, preview_folder_sample, relabel_folder_sample

        new_rel, current_label = relabel_folder_sample(dataset_dir, sample_rel, new_label)
        samples = list_folder_samples(dataset_dir, IMAGE_SUFFIXES)
        labels = list_folder_labels(dataset_dir, IMAGE_SUFFIXES)
        preview_path, _ = preview_folder_sample(dataset_dir, new_rel)
        meta = (
            "### Image sample\n\n"
            f"- **Current label**: `{current_label}`\n"
            f"- **File**: `{new_rel}`"
        )
        return (
            gr.update(choices=samples, value=new_rel),
            gr.update(choices=labels, value=current_label),
            f"✅ Moved sample to label `{current_label}`.",
            preview_path,
            meta,
        )
    except Exception as exc:
        return gr.update(), gr.update(), f"❌ {exc}", None, "*Save failed.*"


def load_audio_annotation_samples(dataset_dir):
    if not dataset_dir:
        return gr.update(choices=[], value=None), gr.update(choices=[], value=None), "*Set an audio dataset folder first.*"
    try:
        from ui.annotation_tools import AUDIO_SUFFIXES, list_folder_labels, list_folder_samples

        samples = list_folder_samples(dataset_dir, AUDIO_SUFFIXES)
        labels = list_folder_labels(dataset_dir, AUDIO_SUFFIXES)
        if not samples:
            return gr.update(choices=[], value=None), gr.update(choices=labels, value=None), "*No audio files were found in class folders.*"
        first_value = samples[0][1]
        return gr.update(choices=samples, value=first_value), gr.update(choices=labels, value=None), f"Loaded `{len(samples)}` audio samples across `{len(labels)}` labels."
    except Exception as exc:
        return gr.update(choices=[], value=None), gr.update(choices=[], value=None), f"❌ {exc}"


def preview_audio_annotation_sample(dataset_dir, sample_rel):
    if not dataset_dir or not sample_rel:
        return None, "*Choose an audio sample to inspect.*", gr.update(value=None)
    try:
        from ui.annotation_tools import AUDIO_SUFFIXES, list_folder_labels, preview_folder_sample

        preview_path, current_label = preview_folder_sample(dataset_dir, sample_rel)
        labels = list_folder_labels(dataset_dir, AUDIO_SUFFIXES)
        meta = (
            "### Audio sample\n\n"
            f"- **Current label**: `{current_label}`\n"
            f"- **File**: `{sample_rel}`"
        )
        return preview_path, meta, gr.update(choices=labels, value=current_label)
    except Exception as exc:
        return None, f"❌ {exc}", gr.update()


def save_audio_relabel(dataset_dir, sample_rel, new_label):
    if not dataset_dir or not sample_rel:
        return gr.update(), gr.update(), "*Choose an audio sample first.*", None, "*Choose an audio sample first.*"
    try:
        from ui.annotation_tools import AUDIO_SUFFIXES, list_folder_labels, list_folder_samples, preview_folder_sample, relabel_folder_sample

        new_rel, current_label = relabel_folder_sample(dataset_dir, sample_rel, new_label)
        samples = list_folder_samples(dataset_dir, AUDIO_SUFFIXES)
        labels = list_folder_labels(dataset_dir, AUDIO_SUFFIXES)
        preview_path, _ = preview_folder_sample(dataset_dir, new_rel)
        meta = (
            "### Audio sample\n\n"
            f"- **Current label**: `{current_label}`\n"
            f"- **File**: `{new_rel}`"
        )
        return (
            gr.update(choices=samples, value=new_rel),
            gr.update(choices=labels, value=current_label),
            f"✅ Moved clip to label `{current_label}`.",
            preview_path,
            meta,
        )
    except Exception as exc:
        return gr.update(), gr.update(), f"❌ {exc}", None, "*Save failed.*"


def load_text_annotation_samples(data_path_val, text_col_val, label_col_val):
    if not data_path_val:
        return gr.update(choices=[], value=None), gr.update(choices=[], value=None), "*Set a text dataset path first.*"
    try:
        from ui.annotation_tools import list_text_annotation_rows

        choices, labels = list_text_annotation_rows(data_path_val, text_col_val, label_col_val)
        if not choices:
            return gr.update(choices=[], value=None), gr.update(choices=labels, value=None), "*No text rows were available for annotation.*"
        first_value = choices[0][1]
        return gr.update(choices=choices, value=first_value), gr.update(choices=labels, value=None), f"Loaded `{len(choices)}` text rows for review."
    except Exception as exc:
        return gr.update(choices=[], value=None), gr.update(choices=[], value=None), f"❌ {exc}"


def preview_text_annotation_sample(data_path_val, row_index_val, text_col_val, label_col_val):
    if not data_path_val or row_index_val in (None, ""):
        return "*Choose a text row to inspect.*", gr.update(value=None)
    try:
        from ui.annotation_tools import format_text_annotation_preview, list_text_annotation_rows, preview_text_row

        text, current_label = preview_text_row(data_path_val, int(row_index_val), text_col_val, label_col_val)
        _, labels = list_text_annotation_rows(data_path_val, text_col_val, label_col_val)
        return format_text_annotation_preview(text, current_label, int(row_index_val)), gr.update(choices=labels, value=current_label)
    except Exception as exc:
        return f"❌ {exc}", gr.update()


def save_text_annotation_sample(data_path_val, row_index_val, text_col_val, label_col_val, new_label_val):
    if not data_path_val or row_index_val in (None, ""):
        return gr.update(), gr.update(), "*Choose a text row first.*", "*Choose a text row first.*"
    try:
        from ui.annotation_tools import (
            backup_path_for, format_text_annotation_preview, list_text_annotation_rows,
            preview_text_row, save_text_relabel,
        )

        saved_label = save_text_relabel(data_path_val, int(row_index_val), label_col_val, new_label_val)
        backup_name = backup_path_for(Path(data_path_val).expanduser()).name
        choices, labels = list_text_annotation_rows(data_path_val, text_col_val, label_col_val)
        text, current_label = preview_text_row(data_path_val, int(row_index_val), text_col_val, label_col_val)
        return (
            gr.update(choices=choices, value=int(row_index_val)),
            gr.update(choices=labels, value=saved_label),
            f"✅ Updated row `{int(row_index_val)}` to label `{saved_label}`. "
            f"The file as it was before your first edit is kept as `{backup_name}`.",
            format_text_annotation_preview(text, current_label, int(row_index_val)),
        )
    except Exception as exc:
        return gr.update(), gr.update(), f"❌ {exc}", "*Save failed.*"


def review_detection_boxes(image_path_val, model_key_val, conf_val, iou_val):
    if not image_path_val:
        return None, gr.update(value=[]), "*Upload an image first.*"
    try:
        from ui.annotation_tools import review_object_boxes

        annotated_rgb, rows, markdown, _ = review_object_boxes(image_path_val, model_key_val, conf_val, iou_val)
        return annotated_rgb, rows, markdown
    except Exception as exc:
        return None, gr.update(value=[]), f"❌ {exc}"


def save_detection_review(image_path_val, rows_val):
    if not image_path_val:
        return "*Upload and review an image first.*"
    try:
        from ui.annotation_tools import save_object_box_review

        txt_path = save_object_box_review(image_path_val, rows_val)
        return f"✅ Saved reviewed YOLO annotations to `{txt_path}`"
    except Exception as exc:
        return f"❌ {exc}"


def register(c: Components) -> None:
    """Wire this area's events to the components in ``c``."""
    c.annotation_image_refresh.click(
        load_image_annotation_samples,
        inputs=[c.annotation_image_dir],
        outputs=[c.annotation_image_sample, c.annotation_image_new_label, c.annotation_image_status],
    )
    c.annotation_image_sample.change(
        preview_image_annotation_sample,
        inputs=[c.annotation_image_dir, c.annotation_image_sample],
        outputs=[c.annotation_image_preview, c.annotation_image_meta, c.annotation_image_new_label],
    )
    c.annotation_image_save.click(
        save_image_relabel,
        inputs=[c.annotation_image_dir, c.annotation_image_sample, c.annotation_image_new_label],
        outputs=[c.annotation_image_sample, c.annotation_image_new_label, c.annotation_image_status, c.annotation_image_preview, c.annotation_image_meta],
    )

    c.annotation_audio_refresh.click(
        load_audio_annotation_samples,
        inputs=[c.annotation_audio_dir],
        outputs=[c.annotation_audio_sample, c.annotation_audio_new_label, c.annotation_audio_status],
    )
    c.annotation_audio_sample.change(
        preview_audio_annotation_sample,
        inputs=[c.annotation_audio_dir, c.annotation_audio_sample],
        outputs=[c.annotation_audio_preview, c.annotation_audio_meta, c.annotation_audio_new_label],
    )
    c.annotation_audio_save.click(
        save_audio_relabel,
        inputs=[c.annotation_audio_dir, c.annotation_audio_sample, c.annotation_audio_new_label],
        outputs=[c.annotation_audio_sample, c.annotation_audio_new_label, c.annotation_audio_status, c.annotation_audio_preview, c.annotation_audio_meta],
    )

    c.annotation_text_refresh.click(
        load_text_annotation_samples,
        inputs=[c.annotation_text_path, c.annotation_text_col, c.annotation_text_label_col],
        outputs=[c.annotation_text_sample, c.annotation_text_new_label, c.annotation_text_status],
    )
    c.annotation_text_sample.change(
        preview_text_annotation_sample,
        inputs=[c.annotation_text_path, c.annotation_text_sample, c.annotation_text_col, c.annotation_text_label_col],
        outputs=[c.annotation_text_preview, c.annotation_text_new_label],
    )
    c.annotation_text_save.click(
        save_text_annotation_sample,
        inputs=[c.annotation_text_path, c.annotation_text_sample, c.annotation_text_col, c.annotation_text_label_col, c.annotation_text_new_label],
        outputs=[c.annotation_text_sample, c.annotation_text_new_label, c.annotation_text_status, c.annotation_text_preview],
    )

    c.annotation_det_review_btn.click(
        review_detection_boxes,
        inputs=[c.annotation_det_image, c.annotation_det_model, c.annotation_det_conf, c.annotation_det_iou],
        outputs=[c.annotation_det_output, c.annotation_det_table, c.annotation_det_status],
    )
    c.annotation_det_save_btn.click(
        save_detection_review,
        inputs=[c.annotation_det_image, c.annotation_det_table],
        outputs=[c.annotation_det_status],
    )

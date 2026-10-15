"""Data and Model tabs: dataset upload and preview, column choices, guided
recommendations, the built-in walkthrough, and controls that follow the
chosen modality, model, task and project mode."""
from __future__ import annotations

import atexit, os, shutil, tempfile, zipfile
from functools import lru_cache

import gradio as gr

from config import DEVICE
from models.registry import get_compatible_tasks, get_models, get_modes, is_vit
from ui.helpers import (
    CLASSIFICATION_METRIC_OPTIONS,
    REGRESSION_METRIC_OPTIONS,
    TASKS,
    _BUILTIN_AUDIO_TUTORIAL_PATH,
    _BUILTIN_IMAGE_TUTORIAL_PATH,
    _BUILTIN_TABULAR_TUTORIAL_PATH,
    _BUILTIN_TEXT_TUTORIAL_PATH,
    _BUILTIN_TIMESERIES_TUTORIAL_PATH,
    _BUILTIN_VIDEO_TUTORIAL_PATH,
    _apply_recommendation_payload,
    _build_image_quality_lab_html,
    _build_text_quality_lab_html,
    _cleaning_preview_empty_html,
    _path_signature,
    _read_structured_preview,
    _suggest_tabular_example,
    _tutorial_markdown,
    _why_this_matters_markdown,
    _workflow_mode_updates,
    _workspace_status_markdown,
    suggest_bundle_name,
)
from ui.layout import Components
from ui.project_state import load_project_state, save_project_state


def update_modality_controls(mod):
    is_structured = mod in {"graph", "tabular", "text", "timeseries"}
    return (
        gr.update(visible=(mod == "text")),
        gr.update(visible=(mod in {"graph", "tabular", "timeseries"})),
        gr.update(visible=(mod == "timeseries")),
        gr.update(visible=(mod == "timeseries")),
        gr.update(visible=(mod == "video")),
        gr.update(visible=(mod == "image")),
        gr.update(visible=(mod == "image")),
        gr.update(visible=(mod == "audio")),
        gr.update(visible=(mod == "audio")),
        gr.update(visible=True),
        gr.update(visible=(mod in {"graph", "tabular", "text", "timeseries"})),
        gr.update(visible=(mod in {"graph", "tabular", "text", "timeseries"})),
        gr.update(visible=(mod in {"graph", "tabular", "text", "timeseries"})),
        gr.update(visible=(mod == "tabular")),
        gr.update(visible=(mod == "tabular")),
        gr.update(visible=(mod == "tabular")),
        gr.update(visible=(mod == "text")),
        gr.update(visible=(mod == "text")),
        gr.update(visible=(mod == "text")),
        gr.update(visible=(mod == "text")),
        gr.update(visible=(mod == "text")),
        gr.update(visible=(mod == "text")),
        gr.update(visible=(mod == "text")),
        gr.update(visible=(mod == "text")),
        gr.update(visible=(mod == "timeseries")),
        gr.update(visible=(mod == "timeseries")),
        gr.update(visible=(mod == "timeseries")),
        gr.update(visible=(mod == "timeseries")),
        gr.update(visible=(mod == "timeseries")),
        gr.update(visible=(mod == "timeseries")),
        gr.update(visible=(mod == "image")),
        gr.update(visible=(mod == "image")),
        gr.update(visible=(mod == "image")),
        gr.update(visible=(mod == "image")),
        gr.update(visible=(mod == "image")),
        gr.update(visible=(mod == "image")),
        gr.update(visible=(mod == "image")),
        gr.update(visible=(mod == "image")),
        gr.update(visible=(mod == "image")),
        gr.update(visible=(mod == "audio")),
        gr.update(visible=(mod == "audio")),
        gr.update(visible=(mod == "audio")),
        gr.update(visible=(mod == "audio")),
        gr.update(visible=(mod == "audio")),
        gr.update(visible=(mod == "audio")),
        gr.update(visible=(mod == "audio")),
        gr.update(visible=(mod == "audio")),
        gr.update(visible=(mod == "video")),
        gr.update(visible=(mod == "tabular")),
    )


@lru_cache(maxsize=32)
def _compute_stats_cached(path_sig, modality_name, label_name, subset_enabled, subset_percent, subset_seed):
    from data_pipeline.stats import compute_stats
    return compute_stats(
        path_sig[0],
        modality_name,
        label_name,
        subset_enabled=bool(subset_enabled),
        subset_percent=float(subset_percent),
        subset_seed=int(subset_seed),
    )


@lru_cache(maxsize=32)
def _validate_dataset_cached(path_sig, modality_name, label_name, subset_enabled, subset_percent, subset_seed):
    from data_pipeline.validation import validate_dataset
    return validate_dataset(
        path_sig[0],
        modality_name,
        label_name,
        subset_enabled=bool(subset_enabled),
        subset_percent=float(subset_percent),
        subset_seed=int(subset_seed),
    )


def _get_dataset_stats(dpath, mod, lcol, subset_enabled=False, subset_percent=100.0, subset_seed=42):
    return _compute_stats_cached(_path_signature(dpath), mod, lcol, bool(subset_enabled), float(subset_percent), int(subset_seed))


def _get_dataset_validation(dpath, mod, lcol, subset_enabled=False, subset_percent=100.0, subset_seed=42):
    return _validate_dataset_cached(_path_signature(dpath), mod, lcol, bool(subset_enabled), float(subset_percent), int(subset_seed))


def _column_choice_updates(dpath, mod, current_label, current_text, current_features, current_time):
    structured_modalities = {"graph", "tabular", "text", "timeseries"}
    if mod not in structured_modalities:
        return (
            gr.update(choices=[], value=current_label or "label"),
            gr.update(choices=[], value=current_text or "text"),
            gr.update(choices=[], value=current_features or []),
            gr.update(choices=[], value=current_time or ""),
        )

    try:
        df = _read_structured_preview(dpath)
    except Exception:
        df = None

    if df is None or df.empty:
        return (
            gr.update(choices=[], value=current_label or "label"),
            gr.update(choices=[], value=current_text or "text"),
            gr.update(choices=[], value=current_features or []),
            gr.update(choices=[], value=current_time or ""),
        )

    from data_pipeline.type_inference import infer_column_types, suggest_features_and_label

    choices = df.columns.tolist()
    suggestions = suggest_features_and_label(df)
    col_types = infer_column_types(df)
    datetime_cols = [col for col, inferred in col_types.items() if inferred == "datetime"]

    label_value = current_label if current_label in choices else None
    if not label_value:
        label_value = suggestions.get("suggested_label") or (choices[0] if choices else "label")

    text_value = current_text if current_text in choices else None
    if not text_value:
        text_value = suggestions.get("text_col") or ("text" if "text" in choices else (choices[0] if choices else "text"))

    time_value = current_time if current_time in choices else None
    if not time_value:
        time_value = datetime_cols[0] if datetime_cols else ""

    blocked_feature_cols = {label_value, time_value, text_value}
    if mod == "graph":
        blocked_feature_cols.add("node_id")
    allowed_feature_choices = [c for c in choices if c not in blocked_feature_cols]
    if current_features:
        feature_value = [col for col in current_features if col in allowed_feature_choices]
    else:
        suggested_feature_cols = suggestions.get("feature_cols", []) or []
        feature_value = [col for col in suggested_feature_cols if col in allowed_feature_choices]
    if mod == "timeseries":
        feature_value = [col for col in feature_value if col != time_value]

    return (
        gr.update(choices=choices, value=label_value),
        gr.update(choices=choices, value=text_value),
        gr.update(choices=allowed_feature_choices, value=feature_value),
        gr.update(choices=[""] + choices, value=time_value),
    )


def update_model_choices(mod, mode):
    choices = get_models(mod, mode)
    return gr.update(choices=choices, value=choices[0] if choices else None)


def update_task_choices(mod):
    choices = get_compatible_tasks(mod)
    if mod in ("tabular", "timeseries") and "regression" not in choices:
        choices = choices + ["regression"]
    return gr.update(choices=choices, value=choices[0] if choices else None)


def update_training_mode(mod, current_mode):
    available = get_modes(mod)
    if not available:
        return gr.update(choices=["fine-tune", "from_scratch"], value=current_mode)
    if current_mode in available and mod not in {"tabular", "timeseries"}:
        return gr.update(choices=available, value=current_mode)
    preferred = None
    if mod in {"tabular", "timeseries"} and "from_scratch" in available:
        preferred = "from_scratch"
    elif "fine-tune" in available:
        preferred = "fine-tune"
    else:
        preferred = available[0]
    return gr.update(choices=available, value=preferred)


def update_clusters_visible(tsk):
    return gr.update(visible=(tsk == "clustering"))


def update_class_weights_visible(tsk):
    """Class weights only apply to classification — hide for other tasks."""
    return gr.update(visible=(tsk == "classification"))


def update_cv_significance_choices(mod, mode, current_model, task_name):
    if mod not in {"tabular", "timeseries"} or task_name not in {"classification", "regression"}:
        return gr.update(choices=[], value=[]), gr.update(
            choices=CLASSIFICATION_METRIC_OPTIONS,
            value="accuracy",
        )
    available = [
        name for name in get_models(mod, mode)
        if name not in {current_model, "Autoencoder"}
    ]
    metric_choices = (
        CLASSIFICATION_METRIC_OPTIONS
        if task_name == "classification"
        else REGRESSION_METRIC_OPTIONS
    )
    return (
        gr.update(choices=available, value=[]),
        gr.update(choices=metric_choices, value=metric_choices[0]),
    )


def update_model_sweep_controls(mod, mode, current_model, task_name):
    if mod == "graph" or task_name not in {"classification", "regression"}:
        return (
            gr.update(choices=[], value=[]),
            gr.update(choices=[], value=[]),
            gr.update(choices=[], value=None),
        )
    available_models = [
        name for name in get_models(mod, mode)
        if name != "Autoencoder"
    ]
    metric_choices = (
        CLASSIFICATION_METRIC_OPTIONS if task_name == "classification" else REGRESSION_METRIC_OPTIONS
    )
    default_metrics = (
        ["accuracy", "f1_macro", "auc_ovr"] if task_name == "classification" else ["mae", "rmse", "r2"]
    )
    sort_metric = default_metrics[0]
    return (
        gr.update(choices=available_models, value=[]),
        gr.update(choices=metric_choices, value=default_metrics),
        gr.update(choices=metric_choices, value=sort_metric),
    )


def update_image_size_guidance(mod, model, current_size):
    if mod != "image":
        return gr.update(visible=False), gr.update(value="", visible=False)

    current = int(current_size or 160)
    if is_vit(model):
        return (
            gr.update(value=224, minimum=224, maximum=224, step=16, visible=True),
            gr.update(
                value=(
                    "### Image size guidance\n\n"
                    "`ViT` models in this app require **224 × 224** inputs, so the size is locked automatically."
                ),
                visible=True,
            ),
        )

    suggested = current if current not in {224} else 160
    return (
        gr.update(value=suggested, minimum=64, maximum=384, step=16, visible=True),
        gr.update(
            value=(
                "### Image size guidance\n\n"
                "Smaller image sizes like **96–160** train faster and use less memory. "
                "Larger sizes like **224–320** preserve more visual detail. "
                "The selected size is saved into preprocessing and reused during inference and export."
            ),
            visible=True,
        ),
    )


def update_modality_augmentation_defaults(mod, level):
    image_defaults = {
        "none":   (False, False, False, False, False, False),
        "light":  (True,  False, False, False, False, False),
        "medium": (True,  False, True,  True,  False, False),
        "heavy":  (True,  True,  True,  True,  True,  True),
    }
    audio_defaults = {
        "none":   (False, False, False, False, False),
        "light":  (True,  False, False, False, False),
        "medium": (True,  True,  True,  True,  False),
        "heavy":  (True,  True,  True,  True,  True),
    }
    img_vals = image_defaults.get(level, image_defaults["light"]) if mod == "image" else image_defaults["none"]
    aud_vals = audio_defaults.get(level, audio_defaults["light"]) if mod == "audio" else audio_defaults["none"]
    return tuple(gr.update(value=value) for value in (*img_vals, *aud_vals))


def update_workspace_cockpit(dpath, mod, lcol, model, tsk, bname, latest_bundle):
    return _workspace_status_markdown(dpath, mod, lcol, model, tsk, bname, latest_bundle)


def load_builtin_walkthrough(tutorial_modality: str):
    from ui.tutorials import (
        prepare_mnist_tutorial, prepare_iris_tutorial, prepare_newsgroups_tutorial,
        prepare_audio_tutorial, prepare_timeseries_tutorial, prepare_video_tutorial,
    )

    mod = tutorial_modality or "image"

    _PREPARE = {
        "image":      (prepare_mnist_tutorial,      _BUILTIN_IMAGE_TUTORIAL_PATH),
        "tabular":    (prepare_iris_tutorial,        _BUILTIN_TABULAR_TUTORIAL_PATH),
        "text":       (prepare_newsgroups_tutorial,  _BUILTIN_TEXT_TUTORIAL_PATH),
        "audio":      (prepare_audio_tutorial,       _BUILTIN_AUDIO_TUTORIAL_PATH),
        "timeseries": (prepare_timeseries_tutorial,  _BUILTIN_TIMESERIES_TUTORIAL_PATH),
        "video":      (prepare_video_tutorial,       _BUILTIN_VIDEO_TUTORIAL_PATH),
    }

    prepare_fn, dest = _PREPARE[mod]
    if not dest.exists() or not any(dest.rglob("*")):
        print(f"[tutorial] Preparing {mod} tutorial dataset …", flush=True)
        try:
            prepare_fn(dest)
            print(f"[tutorial] {mod} tutorial ready.", flush=True)
        except Exception as exc:
            raise gr.Error(f"Could not prepare {mod} tutorial: {exc}") from exc

    # Modality-specific configs: (data_path, image_size, augmentation, val_split,
    #   training_mode, lr, batch_size, epochs, dropout, optimizer, scheduler, class_weights, bundle_name)
    _CONFIGS: dict[str, tuple] = {
        "image":      (str(dest.resolve()),                          64, "none",  0.15, "from_scratch", 1e-3, 16, 5,  0.2, "adam", "none", False, "mnist_tutorial"),
        "tabular":    (str((dest / "iris.csv").resolve()),           64, "none",  0.20, "from_scratch", 1e-3, 32, 20, 0.0, "adam", "none", False, "iris_tutorial"),
        "text":       (str((dest / "newsgroups.csv").resolve()),     64, "none",  0.15, "from_scratch", 2e-4, 32, 5,  0.1, "adam", "none", False, "newsgroups_tutorial"),
        "audio":      (str(dest.resolve()),                          64, "none",  0.15, "from_scratch", 1e-3, 16, 10, 0.2, "adam", "none", False, "speechcommands_tutorial"),
        "timeseries": (str((dest / "timeseries.csv").resolve()),     64, "none",  0.20, "from_scratch", 1e-3, 32, 15, 0.1, "adam", "none", False, "timeseries_tutorial"),
        "video":      (str(dest.resolve()),                          64, "none",  0.20, "from_scratch", 1e-3, 8,  10, 0.2, "adam", "none", False, "video_tutorial"),
    }

    cfg = _CONFIGS[mod]
    return (mod, *cfg, _tutorial_markdown(mod, loaded=True))


def apply_builtin_walkthrough_model(mod, mode):
    model_choices = get_models(mod, mode)
    chosen_model = "TinyCNN" if "TinyCNN" in model_choices else (model_choices[0] if model_choices else None)
    task_choices = get_compatible_tasks(mod) if chosen_model else TASKS
    chosen_task = "classification" if "classification" in task_choices else (task_choices[0] if task_choices else None)
    return (
        gr.update(choices=model_choices, value=chosen_model),
        gr.update(choices=task_choices, value=chosen_task),
    )


def update_evaluate_visibility(task_name, model_choice):
    is_classification = task_name in {"classification", "multi-label"}
    is_regression = task_name == "regression"
    is_clustering = task_name == "clustering"
    is_anomaly = task_name == "anomaly" or model_choice == "Autoencoder"
    return (
        gr.update(visible=is_classification),
        gr.update(visible=is_classification),
        gr.update(visible=is_regression or is_anomaly),
        gr.update(visible=is_classification),
        gr.update(visible=is_clustering),
    )


def persist_project_mode(project_mode_val):
    save_project_state({
        **load_project_state(),
        "project_mode": project_mode_val,
    })
    return project_mode_val

# ── File upload → auto-fill data path ────────────────────────────────────


_UPLOAD_DIRS: list[str] = []
_MAX_UPLOAD_DIRS = 3


def handle_upload(uploaded_file, mod, current_label, current_text, current_features, current_time):
    """Extract zip → temp dir or use CSV path directly."""
    if uploaded_file is None:
        updates = _column_choice_updates("", mod, current_label, current_text, current_features, current_time)
        return "", *updates
    path = uploaded_file if isinstance(uploaded_file, str) else uploaded_file.name
    if path.lower().endswith(".zip"):
        tmp_dir = tempfile.mkdtemp(prefix="nocode_dl_")
        # Remove on exit, and keep only the most recent extractions while
        # running: every upload used to add a full copy until shutdown.
        atexit.register(shutil.rmtree, tmp_dir, True)
        _UPLOAD_DIRS.append(tmp_dir)
        while len(_UPLOAD_DIRS) > _MAX_UPLOAD_DIRS:
            shutil.rmtree(_UPLOAD_DIRS.pop(0), ignore_errors=True)
        from data_pipeline.io_utils import safe_extract_zip
        try:
            safe_extract_zip(path, tmp_dir)
        except zipfile.BadZipFile:
            shutil.rmtree(tmp_dir, ignore_errors=True)
            updates = _column_choice_updates("", mod, current_label, current_text, current_features, current_time)
            return "", *updates
        except ValueError as exc:  # too large, zip bomb, or paths escaping the folder
            shutil.rmtree(tmp_dir, ignore_errors=True)
            raise gr.Error(f"Can't use this zip: {exc}")
        structured = []
        for root, _dirs, files in os.walk(tmp_dir):
            for name in files:
                if name.lower().endswith((".csv", ".tsv", ".json")):
                    structured.append(os.path.join(root, name))
        if len(structured) == 1:
            resolved_path = structured[0]
            updates = _column_choice_updates(resolved_path, mod, current_label, current_text, current_features, current_time)
            return resolved_path, *updates
        # If zip contained a single top-level folder, use that
        entries = os.listdir(tmp_dir)
        if len(entries) == 1 and os.path.isdir(os.path.join(tmp_dir, entries[0])):
            resolved_path = os.path.join(tmp_dir, entries[0])
            updates = _column_choice_updates(resolved_path, mod, current_label, current_text, current_features, current_time)
            return resolved_path, *updates
        updates = _column_choice_updates(tmp_dir, mod, current_label, current_text, current_features, current_time)
        return tmp_dir, *updates
    updates = _column_choice_updates(path, mod, current_label, current_text, current_features, current_time)
    return path, *updates  # CSV / TSV / JSON: use the path directly

# ── Data preview ──────────────────────────────────────────────────────────


def preview_dataset(
    dpath, mod, lcol, txt_col,
    subset_enabled, subset_pct, subset_seed_val,
    text_lowercase_val, text_strip_urls_val, text_strip_punctuation_val, text_remove_stopwords_val,
    text_deduplicate_val, text_apply_stemming_val, text_apply_lemmatization_val,
    image_size_val, image_aug_flip_val, image_aug_vertical_val, image_aug_rotation_val,
    image_aug_color_val, image_aug_gray_val, image_aug_perspective_val,
    image_normalization_val, image_force_grayscale_val,
    project_mode_val, model_val, task_val
):
    if not dpath:
        empty_report = {
            "status": "blocked",
            "blocking_issues": ["No data path provided."],
            "warnings": [],
            "info": [],
            "suggested_actions": ["Upload a dataset or paste a local path before previewing."],
        }
        return "No path provided.", None, "Enter a data path above.", "### Data Quality Report\n\n- No data path provided.", _cleaning_preview_empty_html("No data path provided."), empty_report, _why_this_matters_markdown(project_mode_val, mod, model_val, task_val, {})
    try:
        path = os.path.abspath(str(dpath))

        structured_suffixes = (".csv", ".tsv", ".json")
        is_structured_file = os.path.isfile(path) and path.lower().endswith(structured_suffixes)
        is_directory = os.path.isdir(path)

        if mod in {"image", "audio", "video"} and is_structured_file:
            pretty_mod = mod.capitalize()
            return (
                "Dataset preview unavailable for the current modality.",
                None,
                (
                    f"### {pretty_mod} dataset mismatch\n\n"
                    f"You selected **{mod}** mode, but the path points to a structured file:\n\n"
                    f"`{path}`\n\n"
                    "Switch the modality to **tabular**, **text**, or **timeseries**, "
                        "or provide a folder of class-organised media files instead."
                ),
                "### Data Quality Report\n\n- Preview is blocked until the modality and data source match.",
                _cleaning_preview_empty_html("The data source does not match the selected modality."),
                {"status": "blocked", "blocking_issues": ["Modality and data source do not match."], "warnings": [], "info": [], "suggested_actions": ["Switch modality or point to the expected dataset format."]},
                _why_this_matters_markdown(project_mode_val, mod, model_val, task_val, {}),
            )

        if mod == "graph" and is_structured_file:
            return (
                "Graph dataset preview unavailable for the current source.",
                None,
                (
                    "### Graph dataset mismatch\n\n"
                    "Graph mode expects a **folder** containing `nodes.csv` and `edges.csv`, "
                    f"but the selected path points to a single file:\n\n`{path}`"
                ),
                "### Data Quality Report\n\n- Preview is blocked until the graph folder structure is available.",
                _cleaning_preview_empty_html("Graph datasets must be folders containing nodes.csv and edges.csv."),
                {"status": "blocked", "blocking_issues": ["Graph mode expects a folder dataset."], "warnings": [], "info": [], "suggested_actions": ["Point to a folder with nodes.csv and edges.csv or upload a ZIP containing that folder structure."]},
                _why_this_matters_markdown(project_mode_val, mod, model_val, task_val, {}),
            )

        if mod in {"graph", "tabular", "text", "timeseries"} and is_directory:
            structured_preview = _read_structured_preview(path)
            if structured_preview is None:
                return (
                    "Structured dataset preview unavailable.",
                    None,
                    (
                        "### Structured data not found\n\n"
                        "The selected path is a folder, but I could not find a single CSV, TSV, "
                        "or JSON file to inspect inside it.\n\n"
                        "Point to a structured data file directly, or upload a ZIP containing one."
                    ),
                    "### Data Quality Report\n\n- A structured file could not be located inside this folder.",
                    _cleaning_preview_empty_html("A single structured file could not be located inside this folder."),
                    {"status": "blocked", "blocking_issues": ["No single structured data file was found."], "warnings": [], "info": [], "suggested_actions": ["Point to a CSV, TSV, or JSON file directly."]},
                    _why_this_matters_markdown(project_mode_val, mod, model_val, task_val, {}),
                )

        from data_pipeline.stats import plot_class_distribution
        from data_pipeline.quality_report import build_quality_report, quality_report_markdown
        stats    = _get_dataset_stats(dpath, mod, lcol, subset_enabled, subset_pct, subset_seed_val)
        warnings = _get_dataset_validation(dpath, mod, lcol, subset_enabled, subset_pct, subset_seed_val)
        fig      = plot_class_distribution(stats)
        warn_txt = "\n".join(warnings) if warnings else "✅  No issues detected."
        preview_df = _read_structured_preview(dpath) if mod in {"graph", "tabular", "text", "timeseries"} else None
        if preview_df is not None and bool(subset_enabled) and float(subset_pct) < 100:
            from data_pipeline.io_utils import apply_random_subset
            preview_df, _ = apply_random_subset(
                preview_df,
                enabled=True,
                subset_percent=float(subset_pct),
                subset_seed=int(subset_seed_val),
            )
        quality_report = build_quality_report(path, mod, lcol, stats, warnings, preview_df)
        quality_lab_html = _cleaning_preview_empty_html(
            "This modality currently uses the report and validation panels for quality review."
        )
        if mod == "text":
            quality_lab_html = _build_text_quality_lab_html(
                preview_df,
                txt_col,
                lowercase=bool(text_lowercase_val),
                strip_urls=bool(text_strip_urls_val),
                strip_punctuation=bool(text_strip_punctuation_val),
                remove_stopwords=bool(text_remove_stopwords_val),
                deduplicate=bool(text_deduplicate_val),
                apply_stemming=bool(text_apply_stemming_val),
                apply_lemmatization=bool(text_apply_lemmatization_val),
            )
        elif mod == "image":
            quality_lab_html = _build_image_quality_lab_html(
                path,
                image_size_value=int(image_size_val),
                horizontal_flip=bool(image_aug_flip_val),
                vertical_flip=bool(image_aug_vertical_val),
                rotation=bool(image_aug_rotation_val),
                color_jitter=bool(image_aug_color_val),
                grayscale=bool(image_aug_gray_val),
                perspective=bool(image_aug_perspective_val),
                normalization_preset=str(image_normalization_val),
                force_grayscale=bool(image_force_grayscale_val),
            )

        # Append column type inference for tabular
        if mod == "tabular" and not stats.get("error"):
            try:
                from data_pipeline.type_inference import (
                    infer_column_types, suggest_features_and_label,
                    type_inference_markdown)
                from data_pipeline.io_utils import read_structured_file
                df         = read_structured_file(dpath, nrows=500)
                col_types  = infer_column_types(df)
                suggestions = suggest_features_and_label(df)
                warn_txt   += "\n\n" + type_inference_markdown(col_types, suggestions)
            except Exception:
                pass
        save_project_state({
            "project_mode": project_mode_val,
            "dataset_path": path,
            "modality": mod,
            "label_col": lcol,
            "random_subset": {
                "enabled": bool(subset_enabled),
                "percent": float(subset_pct),
                "seed": int(subset_seed_val),
            },
            "model_name": model_val,
            "task": task_val,
            "quality_report": quality_report,
        })
        return (
            stats.get("summary", str(stats.get("error", ""))),
            fig,
            warn_txt,
            quality_report_markdown(quality_report),
            quality_lab_html,
            quality_report,
            _why_this_matters_markdown(project_mode_val, mod, model_val, task_val, {}),
        )
    except Exception as e:
        pretty_mod = mod.capitalize() if isinstance(mod, str) else "Dataset"
        return (
            f"{pretty_mod} preview could not be generated.",
            None,
            (
                f"### Preview error\n\n"
                f"`{e}`\n\n"
                "Check that the path exists, matches the selected modality, and contains "
                "the expected files."
            ),
            f"### Data Quality Report\n\n- Preview failed: `{e}`",
            _cleaning_preview_empty_html(f"Preview failed: {e}"),
            {"status": "blocked", "blocking_issues": [str(e)], "warnings": [], "info": [], "suggested_actions": ["Fix the preview error before training."]},
            _why_this_matters_markdown(project_mode_val, mod, model_val, task_val, {}),
        )

# ── Guided recommendations ────────────────────────────────────────────────


def get_recommendations(dpath, mod, lcol, subset_enabled, subset_pct, subset_seed_val, project_mode_val, model_val, task_val, quality_report_val):
    if not dpath:
        return "*Provide a data path first.*", "*Preview a dataset to explain the recommendation.*", {}, _why_this_matters_markdown(project_mode_val, mod, model_val, task_val, {})
    try:
        from ui.guided_mode import (
            recommend,
            format_recommendation_summary,
            format_recommendation_rationale,
        )
        stats = _get_dataset_stats(dpath, mod, lcol, subset_enabled, subset_pct, subset_seed_val)
        if "error" in stats:
            return f"Dataset error: {stats['error']}", "*Recommendation unavailable while dataset preview has errors.*", {}, _why_this_matters_markdown(project_mode_val, mod, model_val, task_val, {})
        rec = recommend(mod, stats, DEVICE)
        if isinstance(quality_report_val, dict) and quality_report_val.get("warnings"):
            rec.setdefault("warnings", [])
            rec["warnings"].extend(quality_report_val.get("warnings", [])[:2])
        save_project_state({
            **load_project_state(),
            "project_mode": project_mode_val,
            "recommendation": rec,
        })
        return (
            format_recommendation_summary(rec),
            format_recommendation_rationale(rec),
            rec,
            _why_this_matters_markdown(project_mode_val, mod, model_val or rec.get("model_name", ""), task_val or rec.get("task", "classification"), rec),
        )
    except Exception as e:
        return f"Error: {e}", "*Recommendation engine failed unexpectedly.*", {}, _why_this_matters_markdown(project_mode_val, mod, model_val, task_val, {})


def apply_recommendations(modality_val, rec, project_mode_val, data_path_val, label_col_val, bundle_name_val, bundle_path_val):
    updates = _apply_recommendation_payload(modality_val, rec)
    selected_model = rec.get("model_name", "")
    selected_task = rec.get("task", "classification")
    save_project_state({
        **load_project_state(),
        "project_mode": project_mode_val,
        "recommendation": rec,
        "recommendation_applied": True,
    })
    snapshot, next_step = _workspace_status_markdown(
        data_path_val,
        modality_val,
        label_col_val,
        selected_model,
        selected_task,
        bundle_name_val,
        bundle_path_val,
    )
    why_text = _why_this_matters_markdown(project_mode_val, modality_val, selected_model, selected_task, rec)
    return (*updates, snapshot, next_step, why_text)


def register(c: Components) -> None:
    """Wire this area's events to the components in ``c``."""
    c.project_mode.change(
        persist_project_mode,
        inputs=c.project_mode,
        outputs=c.project_mode_state,
    )
    c.project_mode.change(
        _workflow_mode_updates,
        inputs=c.project_mode,
        outputs=[
            c.split_cleaning_group,
            c.advanced_training_group,
            c.sequence_options_group,
            c.sklearn_options_group,
            c.column_mapping_group,
            c.hyperparams_group,
            c.model_selection_group,
            c.evaluation_report_group,
            c.export_utilities_group,
            c.history_group,
            c.explanation_group_top,
        ],
    )
    c.project_mode.change(
        _why_this_matters_markdown,
        inputs=[c.project_mode, c.modality, c.model_name, c.task, c.recommendation_state],
        outputs=c.why_box,
    )

    c.modality.change(update_modality_controls, c.modality,
                    [
                        c.text_col, c.feature_cols, c.time_col, c.window_size, c.n_frames, c.image_size, c.image_size_hint, c.sample_rate, c.audio_image_size, c.label_col,
                        c.use_random_subset, c.subset_percent, c.subset_seed,
                        c.use_data_cleaning, c.tabular_missing_strategy, c.tabular_clip_outliers,
                        c.text_lowercase, c.text_strip_urls, c.text_strip_punctuation, c.text_remove_stopwords,
                        c.text_deduplicate, c.text_apply_stemming, c.text_apply_lemmatization, c.text_use_ngrams,
                        c.timeseries_sort_by_time, c.timeseries_fill_strategy, c.timeseries_forecast_mode, c.timeseries_forecast_horizon, c.timeseries_lag_steps, c.timeseries_rolling_window,
                        c.image_verify_files, c.image_aug_flip, c.image_aug_vertical, c.image_aug_rotation,
                        c.image_aug_color, c.image_aug_gray, c.image_aug_perspective, c.image_normalization, c.image_force_grayscale,
                        c.audio_verify_files, c.audio_normalize_waveform, c.audio_n_mels,
                        c.audio_aug_noise, c.audio_aug_shift, c.audio_aug_gain, c.audio_aug_time_mask, c.audio_aug_freq_mask,
                        c.video_verify_files, c.tabular_scaling,
                    ])
    c.modality.change(update_training_mode, [c.modality, c.training_mode], c.training_mode)
    c.modality.change(
        _column_choice_updates,
        [c.data_path, c.modality, c.label_col, c.text_col, c.feature_cols, c.time_col],
        [c.label_col, c.text_col, c.feature_cols, c.time_col],
    )
    c.modality.change(update_model_choices, [c.modality, c.training_mode], c.model_name)
    c.modality.change(update_task_choices, c.modality, c.task)
    c.modality.change(
        update_cv_significance_choices,
        [c.modality, c.training_mode, c.model_name, c.task],
        [c.cv_significance_models, c.cv_significance_metric],
    )
    c.modality.change(
        update_model_sweep_controls,
        [c.modality, c.training_mode, c.model_name, c.task],
        [c.sweep_models, c.sweep_metrics, c.sweep_sort_metric],
    )
    c.modality.change(
        update_image_size_guidance,
        [c.modality, c.model_name, c.image_size],
        [c.image_size, c.image_size_hint],
    )
    c.modality.change(
        update_modality_augmentation_defaults,
        [c.modality, c.augmentation],
        [
            c.image_aug_flip, c.image_aug_vertical, c.image_aug_rotation,
            c.image_aug_color, c.image_aug_gray, c.image_aug_perspective,
            c.audio_aug_noise, c.audio_aug_shift, c.audio_aug_gain, c.audio_aug_time_mask, c.audio_aug_freq_mask,
        ],
    )
    c.modality.change(
        _why_this_matters_markdown,
        [c.project_mode, c.modality, c.model_name, c.task, c.recommendation_state],
        c.why_box,
    )
    c.training_mode.change(update_model_choices, [c.modality, c.training_mode], c.model_name)
    c.training_mode.change(
        update_cv_significance_choices,
        [c.modality, c.training_mode, c.model_name, c.task],
        [c.cv_significance_models, c.cv_significance_metric],
    )
    c.training_mode.change(
        update_model_sweep_controls,
        [c.modality, c.training_mode, c.model_name, c.task],
        [c.sweep_models, c.sweep_metrics, c.sweep_sort_metric],
    )
    c.model_name.change(
        update_cv_significance_choices,
        [c.modality, c.training_mode, c.model_name, c.task],
        [c.cv_significance_models, c.cv_significance_metric],
    )
    c.model_name.change(
        update_image_size_guidance,
        [c.modality, c.model_name, c.image_size],
        [c.image_size, c.image_size_hint],
    )
    c.model_name.change(
        update_model_sweep_controls,
        [c.modality, c.training_mode, c.model_name, c.task],
        [c.sweep_models, c.sweep_metrics, c.sweep_sort_metric],
    )
    c.model_name.change(
        update_evaluate_visibility,
        [c.task, c.model_name],
        [c.classification_views_group, c.explanation_group, c.diagnostics_group, c.sample_review_group, c.clustering_group],
    )
    c.model_name.change(
        _why_this_matters_markdown,
        [c.project_mode, c.modality, c.model_name, c.task, c.recommendation_state],
        c.why_box,
    )
    c.augmentation.change(
        update_modality_augmentation_defaults,
        [c.modality, c.augmentation],
        [
            c.image_aug_flip, c.image_aug_vertical, c.image_aug_rotation,
            c.image_aug_color, c.image_aug_gray, c.image_aug_perspective,
            c.audio_aug_noise, c.audio_aug_shift, c.audio_aug_gain, c.audio_aug_time_mask, c.audio_aug_freq_mask,
        ],
    )
    c.task.change(update_clusters_visible, c.task, c.n_clusters)
    c.task.change(update_class_weights_visible, c.task, c.use_class_weights)
    c.task.change(
        update_cv_significance_choices,
        [c.modality, c.training_mode, c.model_name, c.task],
        [c.cv_significance_models, c.cv_significance_metric],
    )
    c.task.change(
        update_model_sweep_controls,
        [c.modality, c.training_mode, c.model_name, c.task],
        [c.sweep_models, c.sweep_metrics, c.sweep_sort_metric],
    )
    c.task.change(
        update_evaluate_visibility,
        [c.task, c.model_name],
        [c.classification_views_group, c.explanation_group, c.diagnostics_group, c.sample_review_group, c.clustering_group],
    )
    c.task.change(
        _why_this_matters_markdown,
        [c.project_mode, c.modality, c.model_name, c.task, c.recommendation_state],
        c.why_box,
    )

    c.demo.load(update_modality_controls, c.modality,
                    [
                        c.text_col, c.feature_cols, c.time_col, c.window_size, c.n_frames, c.image_size, c.image_size_hint, c.sample_rate, c.audio_image_size, c.label_col,
                        c.use_random_subset, c.subset_percent, c.subset_seed,
                        c.use_data_cleaning, c.tabular_missing_strategy, c.tabular_clip_outliers,
                        c.text_lowercase, c.text_strip_urls, c.text_strip_punctuation, c.text_remove_stopwords,
                        c.text_deduplicate, c.text_apply_stemming, c.text_apply_lemmatization, c.text_use_ngrams,
                        c.timeseries_sort_by_time, c.timeseries_fill_strategy, c.timeseries_forecast_mode, c.timeseries_forecast_horizon, c.timeseries_lag_steps, c.timeseries_rolling_window,
                        c.image_verify_files, c.image_aug_flip, c.image_aug_vertical, c.image_aug_rotation,
                        c.image_aug_color, c.image_aug_gray, c.image_aug_perspective, c.image_normalization, c.image_force_grayscale,
                        c.audio_verify_files, c.audio_normalize_waveform, c.audio_n_mels,
                        c.audio_aug_noise, c.audio_aug_shift, c.audio_aug_gain, c.audio_aug_time_mask, c.audio_aug_freq_mask,
                        c.video_verify_files, c.tabular_scaling,
                    ])
    c.demo.load(update_training_mode, [c.modality, c.training_mode], c.training_mode)
    c.demo.load(
        update_cv_significance_choices,
        [c.modality, c.training_mode, c.model_name, c.task],
        [c.cv_significance_models, c.cv_significance_metric],
    )
    c.demo.load(
        update_model_sweep_controls,
        [c.modality, c.training_mode, c.model_name, c.task],
        [c.sweep_models, c.sweep_metrics, c.sweep_sort_metric],
    )
    c.demo.load(
        update_image_size_guidance,
        [c.modality, c.model_name, c.image_size],
        [c.image_size, c.image_size_hint],
    )
    c.demo.load(
        update_modality_augmentation_defaults,
        [c.modality, c.augmentation],
        [
            c.image_aug_flip, c.image_aug_vertical, c.image_aug_rotation,
            c.image_aug_color, c.image_aug_gray, c.image_aug_perspective,
            c.audio_aug_noise, c.audio_aug_shift, c.audio_aug_gain, c.audio_aug_time_mask, c.audio_aug_freq_mask,
        ],
    )
    c.demo.load(update_model_choices, [c.modality, c.training_mode], c.model_name)
    c.demo.load(update_task_choices, c.modality, c.task)
    c.demo.load(
        update_workspace_cockpit,
        [c.data_path, c.modality, c.label_col, c.model_name, c.task, c.bundle_name, c.bundle_out],
        [c.workspace_snapshot, c.next_step_box],
    )
    c.demo.load(
        update_evaluate_visibility,
        [c.task, c.model_name],
        [c.classification_views_group, c.explanation_group, c.diagnostics_group, c.sample_review_group, c.clustering_group],
    )
    c.demo.load(
        _workflow_mode_updates,
        [c.project_mode],
        [
            c.split_cleaning_group,
            c.advanced_training_group,
            c.sequence_options_group,
            c.sklearn_options_group,
            c.column_mapping_group,
            c.hyperparams_group,
            c.model_selection_group,
            c.evaluation_report_group,
            c.export_utilities_group,
            c.history_group,
            c.explanation_group_top,
        ],
    )
    c.demo.load(
        _why_this_matters_markdown,
        [c.project_mode, c.modality, c.model_name, c.task, c.recommendation_state],
        c.why_box,
    )

    c.data_upload.change(
        handle_upload,
        inputs=[c.data_upload, c.modality, c.label_col, c.text_col, c.feature_cols, c.time_col],
        outputs=[c.data_path, c.label_col, c.text_col, c.feature_cols, c.time_col],
    )
    c.data_path.change(
        _column_choice_updates,
        inputs=[c.data_path, c.modality, c.label_col, c.text_col, c.feature_cols, c.time_col],
        outputs=[c.label_col, c.text_col, c.feature_cols, c.time_col],
    )
    c.data_path.change(
        suggest_bundle_name,
        inputs=[c.data_path, c.modality, c.model_name, c.task, c.bundle_name],
        outputs=c.bundle_name,
    )
    c.data_path.change(
        update_workspace_cockpit,
        inputs=[c.data_path, c.modality, c.label_col, c.model_name, c.task, c.bundle_name, c.bundle_out],
        outputs=[c.workspace_snapshot, c.next_step_box],
    )
    c.label_col.change(
        _column_choice_updates,
        inputs=[c.data_path, c.modality, c.label_col, c.text_col, c.feature_cols, c.time_col],
        outputs=[c.label_col, c.text_col, c.feature_cols, c.time_col],
    )
    c.label_col.change(
        update_workspace_cockpit,
        inputs=[c.data_path, c.modality, c.label_col, c.model_name, c.task, c.bundle_name, c.bundle_out],
        outputs=[c.workspace_snapshot, c.next_step_box],
    )
    c.text_col.change(
        _column_choice_updates,
        inputs=[c.data_path, c.modality, c.label_col, c.text_col, c.feature_cols, c.time_col],
        outputs=[c.label_col, c.text_col, c.feature_cols, c.time_col],
    )
    c.time_col.change(
        _column_choice_updates,
        inputs=[c.data_path, c.modality, c.label_col, c.text_col, c.feature_cols, c.time_col],
        outputs=[c.label_col, c.text_col, c.feature_cols, c.time_col],
    )
    c.model_name.change(
        suggest_bundle_name,
        inputs=[c.data_path, c.modality, c.model_name, c.task, c.bundle_name],
        outputs=c.bundle_name,
    )
    c.model_name.change(
        update_workspace_cockpit,
        inputs=[c.data_path, c.modality, c.label_col, c.model_name, c.task, c.bundle_name, c.bundle_out],
        outputs=[c.workspace_snapshot, c.next_step_box],
    )
    c.task.change(
        suggest_bundle_name,
        inputs=[c.data_path, c.modality, c.model_name, c.task, c.bundle_name],
        outputs=c.bundle_name,
    )
    c.task.change(
        update_workspace_cockpit,
        inputs=[c.data_path, c.modality, c.label_col, c.model_name, c.task, c.bundle_name, c.bundle_out],
        outputs=[c.workspace_snapshot, c.next_step_box],
    )
    c.bundle_name.change(
        update_workspace_cockpit,
        inputs=[c.data_path, c.modality, c.label_col, c.model_name, c.task, c.bundle_name, c.bundle_out],
        outputs=[c.workspace_snapshot, c.next_step_box],
    )

    preview_event = c.preview_btn.click(preview_dataset,
                                      inputs=[
                                          c.data_path, c.modality, c.label_col, c.text_col,
                                          c.use_random_subset, c.subset_percent, c.subset_seed,
                                          c.text_lowercase, c.text_strip_urls, c.text_strip_punctuation, c.text_remove_stopwords,
                                          c.text_deduplicate, c.text_apply_stemming, c.text_apply_lemmatization,
                                          c.image_size, c.image_aug_flip, c.image_aug_vertical, c.image_aug_rotation,
                                          c.image_aug_color, c.image_aug_gray, c.image_aug_perspective,
                                          c.image_normalization, c.image_force_grayscale,
                                          c.project_mode, c.model_name, c.task
                                      ],
                                      outputs=[c.stats_summary_box, c.dist_plot, c.validation_box, c.quality_report_box, c.quality_examples_box, c.quality_report_state, c.why_box])

    c.recommend_btn.click(get_recommendations,
                        inputs=[c.data_path, c.modality, c.label_col, c.use_random_subset, c.subset_percent, c.subset_seed, c.project_mode, c.model_name, c.task, c.quality_report_state],
                        outputs=[c.recommendations_box, c.recommendation_rationale_box, c.recommendation_state, c.why_box])
    preview_event.then(
        get_recommendations,
        inputs=[c.data_path, c.modality, c.label_col, c.use_random_subset, c.subset_percent, c.subset_seed, c.project_mode, c.model_name, c.task, c.quality_report_state],
        outputs=[c.recommendations_box, c.recommendation_rationale_box, c.recommendation_state, c.why_box],
    )
    preview_event.then(
        _suggest_tabular_example,
        inputs=[c.data_path, c.modality, c.label_col, c.infer_tab_input],
        outputs=c.infer_tab_input,
    )
    preview_event.then(
        suggest_bundle_name,
        inputs=[c.data_path, c.modality, c.model_name, c.task, c.bundle_name],
        outputs=c.bundle_name,
    )
    preview_event.then(
        update_workspace_cockpit,
        inputs=[c.data_path, c.modality, c.label_col, c.model_name, c.task, c.bundle_name, c.bundle_out],
        outputs=[c.workspace_snapshot, c.next_step_box],
    )

    c.apply_recommendations_btn.click(
        apply_recommendations,
        inputs=[c.modality, c.recommendation_state, c.project_mode, c.data_path, c.label_col, c.bundle_name, c.bundle_out],
        outputs=[
            c.training_mode, c.model_name, c.task, c.augmentation, c.image_size, c.batch_size, c.epochs, c.dropout, c.scheduler_name, c.use_class_weights,
            c.workspace_snapshot, c.next_step_box, c.why_box,
        ],
    )

    c.tutorial_modality_dropdown.change(
        lambda m: _tutorial_markdown(m),
        inputs=[c.tutorial_modality_dropdown],
        outputs=[c.tutorial_box],
    )

    tutorial_event = c.load_tutorial_btn.click(
        load_builtin_walkthrough,
        inputs=[c.tutorial_modality_dropdown],
        outputs=[
            c.modality,
            c.data_path,
            c.image_size,
            c.augmentation,
            c.val_split,
            c.training_mode,
            c.lr,
            c.batch_size,
            c.epochs,
            c.dropout,
            c.optimizer,
            c.scheduler_name,
            c.use_class_weights,
            c.bundle_name,
            c.tutorial_box,
        ],
    )
    tutorial_event.then(
        apply_builtin_walkthrough_model,
        inputs=[c.modality, c.training_mode],
        outputs=[c.model_name, c.task],
    )
    tutorial_event.then(
        preview_dataset,
        inputs=[
            c.data_path, c.modality, c.label_col, c.text_col,
            c.use_random_subset, c.subset_percent, c.subset_seed,
            c.text_lowercase, c.text_strip_urls, c.text_strip_punctuation, c.text_remove_stopwords,
            c.text_deduplicate, c.text_apply_stemming, c.text_apply_lemmatization,
            c.image_size, c.image_aug_flip, c.image_aug_vertical, c.image_aug_rotation,
            c.image_aug_color, c.image_aug_gray, c.image_aug_perspective,
            c.image_normalization, c.image_force_grayscale,
            c.project_mode, c.model_name, c.task
        ],
        outputs=[c.stats_summary_box, c.dist_plot, c.validation_box, c.quality_report_box, c.quality_examples_box, c.quality_report_state, c.why_box],
    )
    tutorial_event.then(
        get_recommendations,
        inputs=[c.data_path, c.modality, c.label_col, c.use_random_subset, c.subset_percent, c.subset_seed, c.project_mode, c.model_name, c.task, c.quality_report_state],
        outputs=[c.recommendations_box, c.recommendation_rationale_box, c.recommendation_state, c.why_box],
    )
    tutorial_event.then(
        update_workspace_cockpit,
        inputs=[c.data_path, c.modality, c.label_col, c.model_name, c.task, c.bundle_name, c.bundle_out],
        outputs=[c.workspace_snapshot, c.next_step_box],
    )

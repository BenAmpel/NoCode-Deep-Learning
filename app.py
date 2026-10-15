"""
NoCode-DL — No-Code Deep Learning Platform
Run with:  python run_local.py
"""
from __future__ import annotations
import atexit, logging, os, re, traceback, tempfile, shutil, zipfile
from functools import lru_cache
from pathlib import Path

# Must be set before gradio is imported. Gradio's usage analytics contact
# api.gradio.app and checkip.amazonaws.com, which contradicts the app's
# "your data never leaves this machine" promise.
os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")

import gradio as gr
import numpy as np

from config import DEVICE
from training import cancel as _cancel
from models.registry import get_models, get_compatible_tasks, get_modes, is_sklearn, is_vit
from ui.project_state import load_project_state, save_project_state
from pipeline import RUN_CONFIG_FIELDS, RunConfig
from pipeline.context import RunResult
from pipeline.progress import RunLog, Update
from pipeline.runner import PreflightError, cleanup as cleanup_run, execute as execute_run, preflight_errors, prepare_run, run as run_stages
from ui.helpers import (
    CLASSIFICATION_METRIC_OPTIONS,
    HISTORY_COLUMNS,
    REGRESSION_METRIC_OPTIONS,
    TASKS,
    _BUILTIN_AUDIO_TUTORIAL_PATH,
    _BUILTIN_IMAGE_TUTORIAL_PATH,
    _BUILTIN_TABULAR_TUTORIAL_PATH,
    _BUILTIN_TEXT_TUTORIAL_PATH,
    _BUILTIN_TIMESERIES_TUTORIAL_PATH,
    _BUILTIN_VIDEO_TUTORIAL_PATH,
    _DEFAULT_CLASSIFICATION_NOTE,
    _DEFAULT_DIAGNOSTICS_NOTE,
    _DEFAULT_EXPLANATION_NOTE,
    _DEFAULT_SAMPLE_REVIEW_NOTE,
    _OUTPUTS_ROOT,
    _PROJECT_ROOT,
    _apply_recommendation_payload,
    _build_action_items,
    _build_dashboard_kpi,
    _build_evaluation_notes,
    _build_image_quality_lab_html,
    _build_text_quality_lab_html,
    _cleaning_preview_empty_html,
    _format_eval_kpis,
    _format_eval_summary,
    _format_live_training_metrics_card,
    _format_status_block,
    _get_pyplot,
    _list_bundles,
    _loss_plot,
    _metric_display_value,
    _path_signature,
    _pick_primary_chart,
    _read_structured_preview,
    _slug_part,
    _suggest_tabular_example,
    _suggest_timeseries_example,
    _tutorial_markdown,
    _why_this_matters_markdown,
    _workflow_mode_updates,
    _workspace_status_markdown,
    suggest_bundle_name,
)
from ui.layout import build_layout

# ── Logging setup ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("nocode_dl")

# The component tree is built in ui/layout.py; the callbacks below wire it up.
APP_THEME = gr.themes.Soft(
    primary_hue="emerald",
    secondary_hue="cyan",
    neutral_hue="slate",
).set(
    body_background_fill="#f6f8f4",
    background_fill_primary="#ffffff",
    background_fill_secondary="#edf3ef",
    block_background_fill="#ffffff",
    block_border_color="#d6e2d8",
    block_label_background_fill="#ffffff",
    block_title_text_color="#16312a",
    body_text_color="#2f4b43",
    color_accent="#16856f",
    color_accent_soft="#d9f2ea",
    input_background_fill="#fbfdfb",
    input_border_color="#bfd4c8",
    button_primary_background_fill="#115d52",
    button_primary_background_fill_hover="#0f7263",
    button_primary_text_color="#f8fffd",
    button_secondary_background_fill="#ecf5f0",
    button_secondary_background_fill_hover="#dceee6",
    button_secondary_text_color="#115d52",
)

# The app stylesheet; launch_app() passes it to demo.launch(css=...).
_APP_CSS_PATH = _PROJECT_ROOT / "styles" / "app.css"




# ─────────────────────────────────────────────────────────────────────────────
# Core pipeline
# ─────────────────────────────────────────────────────────────────────────────

_STOPPED_MESSAGE = "⏹ Training stopped. Nothing was exported for this run."


def _format_run_error(exc: Exception) -> str:
    """Status text for a failed run: a friendly message, then the traceback.

    Call it from the ``except`` block that caught ``exc``.
    """
    try:
        from ui.error_formatter import format_error
        friendly = format_error(exc)
    except Exception:
        friendly = f"❌ **Error:** {exc}"
    tb = traceback.format_exc()
    return (
        f"{friendly}\n\n"
        f"{'─' * 60}\n"
        f"Traceback (most recent call last):\n{tb}"
    )


def run_pipeline(
    config: RunConfig | None = None,
    /,
    progress=gr.Progress(track_tqdm=False),
    **fields,
):
    """Train, evaluate and export one model, yielding UI updates as it goes.

    Pass a ``RunConfig``, or its fields as keyword arguments. The work happens
    in the pipeline package's stages; this generator turns their progress
    into Train-tab output tuples and their exceptions into status text.
    """
    cfg = config if config is not None else RunConfig(**fields)
    _NONE     = gr.update(value=None)

    run_log = RunLog()
    telemetry = run_log.telemetry

    def _emit(status=None, loss_fig=_NONE, eta=None, telemetry_card=None, eval_sum="", cm=_NONE, roc=_NONE,
              tsne=_NONE, gcam=_NONE, shap=_NONE, mis_img=_NONE, mis_df=_NONE,
              bundle="", arch=_NONE,
              train_summary="", reg_plot=_NONE, anomaly_plot=_NONE,
              eval_kpis="",
              class_note=_DEFAULT_CLASSIFICATION_NOTE,
              explanation_note_val=_DEFAULT_EXPLANATION_NOTE,
              diagnostics_note_val=_DEFAULT_DIAGNOSTICS_NOTE,
              sample_review_note_val=_DEFAULT_SAMPLE_REVIEW_NOTE,
              text_shap=_NONE,
              dash_kpi=_NONE, dash_primary=_NONE, dash_explain_plot=_NONE,
              dash_explain_html=_NONE, dash_loss=_NONE, dash_actions=""):
        status_val = status if status is not None else telemetry["status_log"]
        eta_val = eta if eta is not None else telemetry["eta_text"]
        telemetry_val = telemetry_card if telemetry_card is not None else _format_live_training_metrics_card({
            "phase": telemetry.get("phase", "Waiting"),
            "epoch_text": telemetry.get("epoch_text", "—"),
            "train_loss": telemetry.get("latest_metrics", {}).get("train_loss", "—"),
            "val_loss": telemetry.get("latest_metrics", {}).get("val_loss", "—"),
            "metric": telemetry.get("latest_metrics", {}).get("metric", "—"),
            "lr": telemetry.get("latest_metrics", {}).get("lr", "—"),
        })
        return (status_val, loss_fig, eta_val, telemetry_val, eval_sum, cm, roc, tsne, gcam, shap,
                mis_img, mis_df, bundle, arch,
                train_summary, reg_plot, anomaly_plot, eval_kpis,
                class_note, explanation_note_val, diagnostics_note_val,
                sample_review_note_val, text_shap,
                dash_kpi, dash_primary, dash_explain_plot, dash_explain_html,
                dash_loss, dash_actions)

    def _emit_update(update: Update):
        return _emit(
            loss_fig=_loss_plot(update.loss_history) if update.loss_history is not None else _NONE,
            eta=update.eta,
            tsne=gr.update(value=update.tsne_fig) if update.tsne_fig is not None else _NONE,
            train_summary=update.train_summary,
        )

    def _emit_result(result: RunResult):
        history = result.history
        class_note_text, explanation_note_text, diagnostics_note_text, sample_review_note_text = (
            _build_evaluation_notes(
                task=cfg.task,
                modality=cfg.modality,
                model_name=cfg.model_name,
                sklearn_model=result.sklearn_model,
                cm_fig=result.cm_fig,
                roc_fig=result.roc_fig,
                gcam_fig=result.gcam_fig,
                shap_fig=result.shap_fig,
                reg_fig=result.reg_fig,
                anomaly_fig=result.anomaly_fig,
                mis_img_fig=result.mis_img_fig,
                mis_df_val=result.mis_df,
                tsne_fig=result.tsne_fig,
            )
        )
        return _emit(
            loss_fig=gr.update(value=_loss_plot(history) if history else None),
            eta="ETA: Done",
            eval_sum=_format_eval_summary(result.eval_summary),
            cm=gr.update(value=result.cm_fig),
            roc=gr.update(value=result.roc_fig),
            tsne=gr.update(value=result.tsne_fig),
            gcam=gr.update(value=result.gcam_fig),
            shap=gr.update(value=result.shap_fig),
            mis_img=gr.update(value=result.mis_img_fig),
            mis_df=gr.update(value=result.mis_df),
            bundle=result.bundle_path,
            arch=gr.update(value=result.arch_fig),
            train_summary=result.training_summary,
            reg_plot=gr.update(value=result.reg_fig),
            anomaly_plot=gr.update(value=result.anomaly_fig),
            eval_kpis=_format_eval_kpis(result.eval_summary, history, cfg.task),
            class_note=class_note_text,
            explanation_note_val=explanation_note_text,
            diagnostics_note_val=diagnostics_note_text,
            sample_review_note_val=sample_review_note_text,
            text_shap=gr.update(value=result.text_shap_html or ""),
            dash_kpi=_build_dashboard_kpi(cfg.task, cfg.modality, history, result.metrics,
                                          cfg.model_name, cfg.training_mode),
            dash_primary=gr.update(value=_pick_primary_chart(
                cfg.task, result.cm_fig, result.reg_fig, result.tsne_fig, result.anomaly_fig)),
            dash_explain_plot=gr.update(value=result.shap_fig or result.gcam_fig),
            dash_explain_html=gr.update(value=result.text_shap_html or ""),
            dash_loss=gr.update(value=_loss_plot(history) if history else None),
            dash_actions=_build_action_items(cfg.task, history, result.metrics, cfg.model_name),
        )

    preflight = preflight_errors(cfg)
    if preflight:
        yield _emit("\n".join(preflight))
        return

    status = run_log.log("Initialising…")
    run_log.set_phase("initialising", eta_text="ETA: estimating…")
    yield _emit(status)

    ctx = prepare_run(cfg, run_log, progress)
    try:
        for item in execute_run(ctx):
            yield _emit_result(item) if isinstance(item, RunResult) else _emit_update(item)
    except _cancel.TrainingStopped:
        run_log.set_phase("stopped", eta_text="ETA: stopped")
        yield _emit(run_log.log(_STOPPED_MESSAGE))
    except Exception as exc:
        run_log.set_phase("error", eta_text="ETA: halted")
        yield _emit(run_log.log(_format_run_error(exc)))
    finally:
        cleanup_run(ctx)

# ─────────────────────────────────────────────────────────────────────────────
# Gradio UI
# ─────────────────────────────────────────────────────────────────────────────

# Gradio 6 takes theme and css in launch(); see launch_app().
demo, c = build_layout()

with demo:

    # ── Dynamic callbacks ─────────────────────────────────────────────────────

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

    demo.load(update_modality_controls, c.modality,
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
    demo.load(update_training_mode, [c.modality, c.training_mode], c.training_mode)
    demo.load(
        update_cv_significance_choices,
        [c.modality, c.training_mode, c.model_name, c.task],
        [c.cv_significance_models, c.cv_significance_metric],
    )
    demo.load(
        update_model_sweep_controls,
        [c.modality, c.training_mode, c.model_name, c.task],
        [c.sweep_models, c.sweep_metrics, c.sweep_sort_metric],
    )
    demo.load(
        update_image_size_guidance,
        [c.modality, c.model_name, c.image_size],
        [c.image_size, c.image_size_hint],
    )
    demo.load(
        update_modality_augmentation_defaults,
        [c.modality, c.augmentation],
        [
            c.image_aug_flip, c.image_aug_vertical, c.image_aug_rotation,
            c.image_aug_color, c.image_aug_gray, c.image_aug_perspective,
            c.audio_aug_noise, c.audio_aug_shift, c.audio_aug_gain, c.audio_aug_time_mask, c.audio_aug_freq_mask,
        ],
    )
    demo.load(update_model_choices, [c.modality, c.training_mode], c.model_name)
    demo.load(update_task_choices, c.modality, c.task)
    demo.load(
        update_workspace_cockpit,
        [c.data_path, c.modality, c.label_col, c.model_name, c.task, c.bundle_name, c.bundle_out],
        [c.workspace_snapshot, c.next_step_box],
    )
    demo.load(
        update_evaluate_visibility,
        [c.task, c.model_name],
        [c.classification_views_group, c.explanation_group, c.diagnostics_group, c.sample_review_group, c.clustering_group],
    )
    demo.load(
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
    demo.load(
        _why_this_matters_markdown,
        [c.project_mode, c.modality, c.model_name, c.task, c.recommendation_state],
        c.why_box,
    )

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

    # ── Train button ──────────────────────────────────────────────────────────

    # Every RunConfig field is fed by the component of the same name.
    run_config_components = {name: c[name] for name in RUN_CONFIG_FIELDS}

    def train_from_ui(values, progress=gr.Progress(track_tqdm=False)):
        yield from run_pipeline(
            RunConfig.from_component_values(run_config_components, values),
            progress=progress,
        )

    train_event = c.train_btn.click(
        train_from_ui,
        # A set of inputs makes Gradio pass one {component: value} dict.
        inputs=set(run_config_components.values()),
        outputs=[
            c.status_box, c.loss_plot, c.eta_box, c.live_metrics_card,
            c.eval_summary_box, c.cm_plot, c.roc_plot, c.tsne_plot_out,
            c.gradcam_out, c.shap_out, c.misclass_img, c.misclass_df,
            c.bundle_out, c.arch_viz_out,
            c.training_summary_out, c.regression_plot_out, c.anomaly_plot_out, c.eval_kpi_bar,
            c.classification_note, c.explanation_note, c.diagnostics_note, c.sample_review_note,
            c.text_shap_out,
            c.dashboard_kpi, c.dashboard_primary_chart, c.dashboard_explain_plot,
            c.dashboard_explain_html, c.dashboard_loss_chart, c.dashboard_actions,
        ],
        concurrency_id="heavy_jobs",
        concurrency_limit=1,
    )

    # Stop button cancels the running training generator

    # Auto-fill inference bundle path from training result
    c.bundle_out.change(
        lambda p: gr.update(choices=_list_bundles(), value=p),
        inputs=c.bundle_out, outputs=c.infer_bundle_path)
    c.infer_refresh_btn.click(
        lambda: gr.update(choices=_list_bundles()),
        outputs=c.infer_bundle_path)
    c.bundle_out.change(
        update_workspace_cockpit,
        inputs=[c.data_path, c.modality, c.label_col, c.model_name, c.task, c.bundle_name, c.bundle_out],
        outputs=[c.workspace_snapshot, c.next_step_box],
    )
    c.bundle_out.change(
        _suggest_tabular_example,
        inputs=[c.data_path, c.modality, c.label_col, c.infer_tab_input],
        outputs=c.infer_tab_input,
    )

    # ── Cross-validation callback ─────────────────────────────────────────────

    def run_cv(modality_val, data_path_val, text_col_val, label_col_val,
               time_col_val, window_size_val, n_frames_val, sample_rate_val,
               augmentation_val, val_split_val, subset_enabled_val, subset_percent_val, subset_seed_val, training_mode_val, model_name_val,
               task_val, epochs_val, lr_val, batch_size_val, dropout_val,
               optimizer_val, scheduler_val, use_amp_val,
               timeseries_forecast_mode_val, timeseries_forecast_horizon_val, timeseries_lag_steps_val, timeseries_rolling_window_val,
               hidden_size_val, num_layers_val, cv_k_val):
        if not data_path_val or not model_name_val:
            yield "❌ Set Data path and Model before running cross-validation."
            return

        yield f"Starting {int(cv_k_val)}-fold cross-validation…\n"
        lines = []

        try:
            from training.cross_val import cross_validate, format_cv_results

            # Build a model factory that captures the current UI settings
            def _model_factory(num_classes: int, input_size: int = 1):
                if is_sklearn(model_name_val):
                    from models.tabular_models import get_tabular_model
                    return get_tabular_model(model_name_val, num_classes=num_classes,
                                             input_size=input_size,
                                             task=task_val,
                                             dropout=float(dropout_val))
                elif modality_val == "text":
                    from models.text_models import get_text_model
                    return get_text_model(model_name_val, num_classes=num_classes,
                                          mode=training_mode_val,
                                          hidden_size=int(hidden_size_val),
                                          num_layers=int(num_layers_val),
                                          dropout=float(dropout_val))
                elif modality_val == "tabular":
                    from models.tabular_models import get_tabular_model
                    return get_tabular_model(model_name_val, num_classes=num_classes,
                                             input_size=input_size,
                                             dropout=float(dropout_val))
                elif modality_val == "timeseries":
                    from models.timeseries_models import get_timeseries_model
                    return get_timeseries_model(
                        model_name_val,
                        num_classes=num_classes,
                        input_size=input_size,
                        hidden_size=int(hidden_size_val),
                        num_layers=int(num_layers_val),
                        dropout=float(dropout_val),
                    )
                else:
                    from models.image_models import get_image_model
                    return get_image_model(model_name_val, num_classes=num_classes,
                                           mode=training_mode_val,
                                           dropout=float(dropout_val))

            # Load the full dataset for cross-validation
            if modality_val == "tabular":
                import pandas as pd, numpy as np
                import torch
                from torch.utils.data import TensorDataset
                from modalities.tabular import load_tabular_data
                # We need the full dataset; load with a tiny val_split and combine
                # augmentation="none": noise must not be applied before the data
                # is split into folds, or validation folds see augmented rows.
                tl, vl, classes, prep, input_size, Xtr, ytr, Xv, yv = load_tabular_data(
                    data_path_val, label_col=label_col_val, batch_size=int(batch_size_val),
                    val_split=0.01, augmentation="none",
                    subset_percent=float(subset_percent_val) if bool(subset_enabled_val) else 100.0,
                    subset_seed=int(subset_seed_val),
                    task=task_val)
                X_all = np.concatenate([Xtr, Xv])
                y_all = np.concatenate([ytr, yv])
                label_dtype = "float32" if task_val == "regression" else "int64"
                full_dataset = TensorDataset(
                    torch.tensor(X_all), torch.tensor(y_all.astype(label_dtype)))
                num_classes  = len(classes)

                for prog in cross_validate(
                    full_dataset,
                    model_factory=lambda: _model_factory(num_classes, input_size),
                    k=int(cv_k_val),
                    epochs=int(epochs_val), lr=float(lr_val),
                    optimizer_name=optimizer_val, task=task_val,
                    batch_size=int(batch_size_val),
                ):
                    if prog.get("done"):
                        result = format_cv_results(
                            prog["fold_accs"],
                            prog["mean_acc"],
                            prog["std_acc"],
                            fold_metrics=prog.get("fold_metrics"),
                            aggregate_metrics=prog.get("aggregate_metrics"),
                            task=task_val,
                        )
                        if prog.get("warning"):
                            result = f"> ⚠️ {prog['warning']}\n\n" + result
                        lines.append(result)
                    else:
                        fold_line = (
                            f"Fold {prog['fold']}/{prog['total_folds']}  "
                            f"epoch {prog['epoch']}  val_metric={prog.get('val_acc', 0):.2f}"
                            + ("%" if task_val == "classification" else "")
                        )
                        lines.append(fold_line)
                    yield "\n".join(lines)
            elif modality_val == "timeseries":
                import numpy as np
                import torch
                from torch.utils.data import TensorDataset
                from modalities.timeseries import prepare_timeseries_windows

                windows, labels, classes, prep, input_size = prepare_timeseries_windows(
                    data_path=data_path_val,
                    label_col=label_col_val,
                    time_col=time_col_val if time_col_val else None,
                    window_size=int(window_size_val),
                    task=task_val,
                    subset_percent=float(subset_percent_val) if bool(subset_enabled_val) else 100.0,
                    subset_seed=int(subset_seed_val),
                    forecast_mode=bool(timeseries_forecast_mode_val),
                    forecast_horizon=int(timeseries_forecast_horizon_val),
                    lag_steps=int(timeseries_lag_steps_val),
                    rolling_window=int(timeseries_rolling_window_val),
                )
                num_classes = len(classes) if task_val == "classification" else 1
                from sklearn.model_selection import TimeSeriesSplit
                temporal_splits = list(TimeSeriesSplit(n_splits=int(cv_k_val)).split(windows))

                if is_sklearn(model_name_val):
                    X_all = windows.reshape(len(windows), -1).astype(np.float32)
                    y_all = labels
                    from eval.metrics import classification_metric_summary, regression_metric_summary

                    fold_scores = []
                    fold_metrics = []
                    for fold_idx, (train_idx, val_idx) in enumerate(temporal_splits, start=1):
                        model = _model_factory(num_classes=num_classes, input_size=X_all.shape[1])
                        X_train = X_all[train_idx]
                        y_train = y_all[train_idx]
                        X_val = X_all[val_idx]
                        y_val = y_all[val_idx]
                        lines.append(f"Fold {fold_idx}/{int(cv_k_val)}  fitting classical baseline…")
                        yield "\n".join(lines)
                        model.fit(X_train, y_train)
                        y_pred = model.predict(X_val)
                        y_prob = model.predict_proba(X_val) if hasattr(model, "predict_proba") else None
                        if task_val == "classification":
                            metric_summary = classification_metric_summary(y_val, y_pred, y_prob, n_classes=num_classes)
                            score = float(metric_summary["accuracy"]) * 100.0
                        else:
                            metric_summary = regression_metric_summary(y_val, y_pred)
                            score = float(metric_summary["mae"])
                        fold_scores.append(round(score, 4))
                        fold_metrics.append(metric_summary)
                    from training.cross_val import _aggregate_metric_dicts, format_cv_results
                    result = format_cv_results(
                        fold_scores,
                        round(float(np.mean(fold_scores)), 4),
                        round(float(np.std(fold_scores)), 4),
                        fold_metrics=fold_metrics,
                        aggregate_metrics=_aggregate_metric_dicts(fold_metrics),
                        task=task_val,
                    )
                    result = "> ℹ️ Time-series cross-validation uses walk-forward splits to preserve chronology.\n\n" + result
                    lines.append(result)
                    yield "\n".join(lines)
                else:
                    label_dtype = torch.int64 if task_val == "classification" else torch.float32
                    full_dataset = TensorDataset(
                        torch.tensor(windows, dtype=torch.float32),
                        torch.tensor(labels, dtype=label_dtype),
                    )
                    for prog in cross_validate(
                        full_dataset,
                        model_factory=lambda: _model_factory(num_classes, input_size),
                        k=int(cv_k_val),
                        epochs=int(epochs_val), lr=float(lr_val),
                        optimizer_name=optimizer_val, task=task_val,
                        batch_size=int(batch_size_val),
                        splits=temporal_splits,
                        split_strategy_name="timeseries_split",
                    ):
                        if prog.get("done"):
                            result = format_cv_results(
                                prog["fold_accs"],
                                prog["mean_acc"],
                                prog["std_acc"],
                                fold_metrics=prog.get("fold_metrics"),
                                aggregate_metrics=prog.get("aggregate_metrics"),
                                task=task_val,
                            )
                            result = "> ℹ️ Time-series cross-validation uses walk-forward splits to preserve chronology.\n\n" + result
                            lines.append(result)
                        else:
                            fold_line = (
                                f"Fold {prog['fold']}/{prog['total_folds']}  "
                                f"epoch {prog['epoch']}  val_metric={prog.get('val_acc', 0):.2f}"
                                + ("%" if task_val == "classification" else "")
                            )
                            lines.append(fold_line)
                        yield "\n".join(lines)
            else:
                yield "ℹ️ Cross-validation is currently supported for **tabular** modality only."
                return

        except Exception as exc:
            yield f"❌ Cross-validation failed: {exc}"

    cv_event = c.cv_run_btn.click(
        run_cv,
        inputs=[
            c.modality, c.data_path, c.text_col, c.label_col, c.time_col,
            c.window_size, c.n_frames, c.sample_rate, c.augmentation,
            c.val_split, c.use_random_subset, c.subset_percent, c.subset_seed, c.training_mode, c.model_name, c.task,
            c.epochs, c.lr, c.batch_size, c.dropout, c.optimizer, c.scheduler_name, c.use_amp,
            c.timeseries_forecast_mode, c.timeseries_forecast_horizon, c.timeseries_lag_steps, c.timeseries_rolling_window,
            c.hidden_size, c.num_layers, c.cv_k,
        ],
        outputs=c.cv_results_box,
        concurrency_id="heavy_jobs",
        concurrency_limit=1,
    )

    def run_cv_significance(modality_val, data_path_val, label_col_val, time_col_val, window_size_val, subset_enabled_val, subset_percent_val, subset_seed_val, training_mode_val,
                            timeseries_forecast_mode_val, timeseries_forecast_horizon_val, timeseries_lag_steps_val, timeseries_rolling_window_val,
                            base_model_name, task_val, batch_size_val, dropout_val,
                            optimizer_val, hidden_size_val, num_layers_val,
                            epochs_val, lr_val, n_estimators_val, max_depth_val,
                            c_param_val, max_iter_val, lr_xgb_val, cv_k_val,
                            metric_name, compare_models):
        if modality_val not in {"tabular", "timeseries"}:
            return "### ⚠️ Significance testing is currently available for tabular and time-series models only.", None
        if task_val not in {"classification", "regression"}:
            return "### ⚠️ Significance testing currently supports classification or regression tasks.", None
        if int(cv_k_val) != 10:
            return "### ⚠️ Set cross-validation to **10 folds** before running the significance test.", None
        selected_models = [m for m in (compare_models or []) if m and m != base_model_name]
        if not selected_models:
            return "### ⚠️ Choose at least one additional model family to compare against the active model.", None

        import numpy as np
        import torch
        from scipy.stats import ttest_rel, wilcoxon
        from sklearn.model_selection import KFold, StratifiedKFold, TimeSeriesSplit
        from torch.utils.data import DataLoader, TensorDataset

        from eval.metrics import classification_metric_summary, regression_metric_summary
        from models.registry import is_sklearn
        from models.tabular_models import get_tabular_model
        from modalities.tabular import load_tabular_data
        from modalities.timeseries import prepare_timeseries_windows
        from training.cross_val import _collect_fold_predictions, _train_one_fold

        if modality_val == "tabular":
            _tl, _vl, classes, _prep, input_size, Xtr, ytr, Xv, yv = load_tabular_data(
                data_path_val,
                label_col=label_col_val,
                batch_size=int(batch_size_val),
                val_split=0.01,
                augmentation="none",
                subset_percent=float(subset_percent_val) if bool(subset_enabled_val) else 100.0,
                subset_seed=int(subset_seed_val),
                task=task_val,
            )
            X_all = np.concatenate([Xtr, Xv])
            y_all = np.concatenate([ytr, yv])
            ts_input_size = None
            windows_all = None
        else:
            windows_all, y_all, classes, prep_ts, ts_input_size = prepare_timeseries_windows(
                data_path=data_path_val,
                label_col=label_col_val,
                time_col=time_col_val if time_col_val else None,
                window_size=int(window_size_val),
                task=task_val,
                subset_percent=float(subset_percent_val) if bool(subset_enabled_val) else 100.0,
                subset_seed=int(subset_seed_val),
                forecast_mode=bool(timeseries_forecast_mode_val),
                forecast_horizon=int(timeseries_forecast_horizon_val),
                lag_steps=int(timeseries_lag_steps_val),
                rolling_window=int(timeseries_rolling_window_val),
            )
            X_all = windows_all.reshape(len(windows_all), -1).astype(np.float32)
            input_size = int(prep_ts.get("sklearn_input_size", X_all.shape[1]))

        split_warning = None
        if modality_val == "timeseries":
            splitter = TimeSeriesSplit(n_splits=10)
            splits = list(splitter.split(X_all))
            higher_is_better = metric_name not in {"mae", "rmse"}
        elif task_val == "classification":
            _, counts = np.unique(y_all, return_counts=True)
            min_class_size = int(counts.min()) if len(counts) else 0
            if min_class_size >= 10:
                splitter = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
                splits = list(splitter.split(X_all, y_all))
            else:
                splitter = KFold(n_splits=10, shuffle=True, random_state=42)
                splits = list(splitter.split(X_all))
                split_warning = (
                    f"Used plain K-Fold because the smallest class has only {min_class_size} samples."
                )
            higher_is_better = metric_name not in {"mae", "rmse"}
        else:
            splitter = KFold(n_splits=10, shuffle=True, random_state=42)
            splits = list(splitter.split(X_all))
            higher_is_better = metric_name == "r2"

        def build_family(model_name_local):
            if modality_val == "timeseries" and not is_sklearn(model_name_local):
                from models.timeseries_models import get_timeseries_model

                return get_timeseries_model(
                    model_name_local,
                    num_classes=(len(classes) if task_val == "classification" else 1),
                    input_size=ts_input_size,
                    hidden_size=int(hidden_size_val),
                    num_layers=int(num_layers_val),
                    dropout=float(dropout_val),
                )
            return get_tabular_model(
                model_name_local,
                num_classes=len(classes) if task_val == "classification" else 1,
                input_size=input_size,
                task=task_val,
                dropout=float(dropout_val),
                n_estimators=int(n_estimators_val),
                max_depth=int(max_depth_val),
                C=float(c_param_val),
                max_iter=int(max_iter_val),
                learning_rate=float(lr_xgb_val),
            )

        def score_family(model_name_local):
            fold_scores = []
            for train_idx, val_idx in splits:
                if is_sklearn(model_name_local):
                    X_train = X_all[train_idx]
                    y_train = y_all[train_idx]
                    X_val = X_all[val_idx]
                    y_val = y_all[val_idx]
                    model = build_family(model_name_local)
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_val)
                    y_prob = model.predict_proba(X_val) if hasattr(model, "predict_proba") else None
                else:
                    model = build_family(model_name_local)
                    if modality_val == "timeseries":
                        label_dtype = torch.int64 if task_val == "classification" else torch.float32
                        train_loader = DataLoader(
                            TensorDataset(
                                torch.tensor(windows_all[train_idx], dtype=torch.float32),
                                torch.tensor(y_all[train_idx], dtype=label_dtype),
                            ),
                            batch_size=int(batch_size_val),
                            shuffle=True,
                        )
                        val_loader = DataLoader(
                            TensorDataset(
                                torch.tensor(windows_all[val_idx], dtype=torch.float32),
                                torch.tensor(y_all[val_idx], dtype=label_dtype),
                            ),
                            batch_size=int(batch_size_val),
                            shuffle=False,
                        )
                    else:
                        X_train = X_all[train_idx]
                        y_train = y_all[train_idx]
                        X_val = X_all[val_idx]
                        y_val = y_all[val_idx]
                        label_dtype = torch.int64 if task_val == "classification" else torch.float32
                        train_loader = DataLoader(
                            TensorDataset(torch.tensor(X_train), torch.tensor(y_train, dtype=label_dtype)),
                            batch_size=int(batch_size_val),
                            shuffle=True,
                        )
                        val_loader = DataLoader(
                            TensorDataset(torch.tensor(X_val), torch.tensor(y_val, dtype=label_dtype)),
                            batch_size=int(batch_size_val),
                            shuffle=False,
                        )
                    for _ in _train_one_fold(
                        model,
                        train_loader,
                        val_loader,
                        epochs=int(epochs_val),
                        lr=float(lr_val),
                        optimizer_name=optimizer_val,
                        task=task_val,
                    ):
                        pass
                    y_true, y_pred, y_prob = _collect_fold_predictions(model, val_loader, task_val)
                    y_val = y_true

                if task_val == "classification":
                    metric_summary = classification_metric_summary(y_val, y_pred, y_prob, n_classes=len(classes))
                else:
                    metric_summary = regression_metric_summary(y_val, y_pred)
                score = metric_summary.get(metric_name)
                if score is None:
                    raise ValueError(f"Metric '{metric_name}' is unavailable for model '{model_name_local}'.")
                fold_scores.append(float(score))
            return np.asarray(fold_scores, dtype=float)

        baseline_scores = score_family(base_model_name)
        challenger_curves = {}
        lines = [
            "### Significance testing",
            "",
            f"- **Baseline family**: `{base_model_name}`",
            f"- **Metric**: `{metric_name}`",
            f"- **Folds**: `10` paired folds",
            f"- **Split strategy**: `{'TimeSeriesSplit' if modality_val == 'timeseries' else ('StratifiedKFold' if split_warning is None and task_val == 'classification' else 'KFold')}`",
            "",
            "| Challenger | Baseline mean | Challenger mean | Delta | Paired t-test p | Wilcoxon p | Result |",
            "|---|---:|---:|---:|---:|---:|---|",
        ]
        if split_warning:
            lines.insert(5, f"- **Note**: {split_warning}")

        for challenger in selected_models:
            challenger_scores = score_family(challenger)
            challenger_curves[challenger] = challenger_scores
            if higher_is_better:
                delta = challenger_scores.mean() - baseline_scores.mean()
            else:
                delta = baseline_scores.mean() - challenger_scores.mean()
            ttest_p = float(ttest_rel(challenger_scores, baseline_scores).pvalue)
            try:
                wilcoxon_p = float(wilcoxon(challenger_scores, baseline_scores).pvalue)
            except ValueError:
                wilcoxon_p = float("nan")
            result = "Different" if (ttest_p < 0.05 or (not np.isnan(wilcoxon_p) and wilcoxon_p < 0.05)) else "No clear difference"
            lines.append(
                f"| {challenger} | {baseline_scores.mean():.4f} | {challenger_scores.mean():.4f} | {delta:+.4f} | {ttest_p:.4f} | "
                f"{('n/a' if np.isnan(wilcoxon_p) else f'{wilcoxon_p:.4f}')} | {result} |"
            )

        lines.extend([
            "",
            "Paired tests use the same 10 folds for each family. Positive delta means the challenger performed better on the selected metric.",
        ])
        plot = _plot_cv_significance_scores(metric_name, base_model_name, baseline_scores, challenger_curves)
        return "\n".join(lines), plot

    cv_sig_event = c.cv_significance_btn.click(
        run_cv_significance,
        inputs=[
            c.modality, c.data_path, c.label_col, c.time_col, c.window_size, c.use_random_subset, c.subset_percent, c.subset_seed, c.training_mode,
            c.timeseries_forecast_mode, c.timeseries_forecast_horizon, c.timeseries_lag_steps, c.timeseries_rolling_window,
            c.model_name, c.task, c.batch_size, c.dropout,
            c.optimizer, c.hidden_size, c.num_layers,
            c.epochs, c.lr, c.n_estimators, c.max_depth,
            c.C_param, c.max_iter, c.lr_xgb, c.cv_k,
            c.cv_significance_metric, c.cv_significance_models,
        ],
        outputs=[c.cv_significance_box, c.cv_significance_plot],
        concurrency_id="heavy_jobs",
        concurrency_limit=1,
    )

    # ── Hyperparameter search callback ────────────────────────────────────────

    def run_hparam_search(modality_val, data_path_val, text_col_val, label_col_val,
                          time_col_val, window_size_val, n_frames_val, sample_rate_val,
                          augmentation_val, val_split_val, subset_enabled_val, subset_percent_val, subset_seed_val, training_mode_val,
                          model_name_val, task_val, batch_size_val, dropout_val,
                          optimizer_val, hidden_size_val, num_layers_val,
                          lr_vals_str, dropout_vals_str, n_trials_val, hparam_epochs_val):
        if not data_path_val or not model_name_val:
            yield "❌ Set Data path and Model before running hyperparameter search."
            return

        # Parse candidate values
        try:
            lr_candidates      = [float(v.strip()) for v in lr_vals_str.split(",") if v.strip()]
            dropout_candidates = [float(v.strip()) for v in dropout_vals_str.split(",") if v.strip()]
        except ValueError as e:
            yield f"❌ Could not parse candidates: {e}"
            return

        param_space = {"lr": lr_candidates, "dropout": dropout_candidates}
        yield f"Starting random hyperparameter search over {len(lr_candidates)}×{len(dropout_candidates)} grid…\n"
        lines = []

        try:
            if modality_val != "tabular":
                yield "ℹ️ Hyperparameter search is currently supported for **tabular** modality only."
                return

            import pandas as pd, numpy as np, torch
            from torch.utils.data import TensorDataset, DataLoader
            from modalities.tabular import load_tabular_data
            from training.hparam_search import random_search, format_search_results

            tl, vl, classes, prep, input_size, Xtr, ytr, Xv, yv = load_tabular_data(
                data_path_val, label_col=label_col_val, batch_size=int(batch_size_val),
                val_split=float(val_split_val),
                subset_percent=float(subset_percent_val) if bool(subset_enabled_val) else 100.0,
                subset_seed=int(subset_seed_val),
                task=task_val)
            num_classes = len(classes)

            def _factory(params):
                from models.tabular_models import get_tabular_model
                return get_tabular_model(
                    model_name_val, num_classes=num_classes,
                    input_size=input_size,
                    dropout=float(params.get("dropout", dropout_val)))

            all_results = []
            for prog in random_search(
                _factory, tl, vl,
                param_space=param_space,
                n_trials=int(n_trials_val),
                epochs_per_trial=int(hparam_epochs_val),
                task=task_val,
            ):
                if prog.get("done"):
                    lines.append("\n" + format_search_results(prog["all_results"]))
                    lines.append(f"\n✅ Best params: {prog['best_params']}  "
                                 f"val_acc={prog['best_val_acc']:.2f}%")
                else:
                    lines.append(f"Trial {prog['trial']}/{prog['total_trials']}  "
                                 f"params={prog['params']}  val_acc={prog['val_acc']:.2f}%")
                yield "\n".join(lines)

        except Exception as exc:
            yield f"❌ Hyperparameter search failed: {exc}"

    hparam_event = c.hparam_run_btn.click(
        run_hparam_search,
        inputs=[
            c.modality, c.data_path, c.text_col, c.label_col, c.time_col,
            c.window_size, c.n_frames, c.sample_rate, c.augmentation,
            c.val_split, c.use_random_subset, c.subset_percent, c.subset_seed, c.training_mode, c.model_name, c.task,
            c.batch_size, c.dropout, c.optimizer,
            c.hidden_size, c.num_layers,
            c.hparam_lr_vals, c.hparam_dropout_vals, c.hparam_n_trials, c.hparam_epochs,
        ],
        outputs=c.hparam_results_box,
        concurrency_id="heavy_jobs",
        concurrency_limit=1,
    )

    def _train_sweep_model(run_config):
        """Train one sweep model; return its bundle path or raise RuntimeError(reason).

        The reason is the first ❌ line of the status text the Train tab would
        have shown for the same failure.
        """
        result = None
        try:
            for item in run_stages(run_config):
                if isinstance(item, RunResult):
                    result = item
        except PreflightError as exc:
            failure_text = "\n".join(exc.errors)
        except _cancel.TrainingStopped:
            failure_text = _STOPPED_MESSAGE
        except Exception as exc:
            failure_text = _format_run_error(exc)
        else:
            if result is not None and result.bundle_path:
                return result.bundle_path
            failure_text = ""
        reason = next(
            (line.strip().lstrip("❌").replace("**", "").strip()
             for line in failure_text.splitlines() if line.strip().startswith("❌")),
            "the run saved no model bundle",
        )
        raise RuntimeError(reason)

    def run_model_sweep(config, sweep_models_val, sweep_metrics_val, sweep_sort_metric_val, sweep_sort_order_val):
        """Train ``config`` once per model in ``sweep_models_val`` and tabulate the metrics."""
        modality_val, task_val, bundle_name_val = config.modality, config.task, config.bundle_name
        import pandas as pd
        from training.run_comparison import load_history

        selected_models = [m for m in (sweep_models_val or []) if m]
        selected_metrics = [m for m in (sweep_metrics_val or []) if m]
        if len(selected_models) < 2:
            yield (
                "### ⚠️ Model sweep\n\nChoose at least two models to compare.",
                pd.DataFrame(),
            )
            return
        if task_val not in {"classification", "regression"}:
            yield (
                "### ⚠️ Model sweep\n\nModel sweep currently supports classification and regression tasks.",
                pd.DataFrame(),
            )
            return
        if not selected_metrics:
            yield (
                "### ⚠️ Model sweep\n\nChoose at least one metric to show in the table.",
                pd.DataFrame(),
            )
            return

        status_lines = ["### Model sweep", "", f"Preparing {len(selected_models)} runs for `{modality_val}` / `{task_val}`."]
        rows = []

        def _sorted_frame(items):
            if not items:
                return pd.DataFrame(columns=["model", *selected_metrics, "status", "bundle"])
            df = pd.DataFrame(items)
            sort_metric = sweep_sort_metric_val if sweep_sort_metric_val in df.columns else None
            if sort_metric:
                df = df.sort_values(
                    by=sort_metric,
                    ascending=(str(sweep_sort_order_val) == "ascending"),
                    na_position="last",
                    kind="stable",
                )
            visible_cols = ["model", *selected_metrics, "status", "bundle"]
            return df[[c for c in visible_cols if c in df.columns]]

        _cancel.reset()
        for idx, model_choice in enumerate(selected_models, start=1):
            if _cancel.stop_requested():
                status_lines.append("⏹ Sweep stopped.")
                break
            status_lines.append(f"- Running `{model_choice}` ({idx}/{len(selected_models)})…")
            yield "\n".join(status_lines), _sorted_frame(rows)

            local_bundle = f"{bundle_name_val}_{_slug_part(model_choice)}"
            try:
                bundle_path_val = _train_sweep_model(
                    config.replace(model_name=model_choice, bundle_name=local_bundle))
                run_record = next((r for r in reversed(load_history()) if r.get("bundle_path") == bundle_path_val), None)
                metric_payload = (run_record or {}).get("metrics", {})
                row = {
                    "model": model_choice,
                    "status": "complete",
                    "bundle": bundle_path_val,
                }
                for metric_name in selected_metrics:
                    row[metric_name] = _metric_display_value(metric_name, metric_payload.get(metric_name))
                rows.append(row)
                status_lines.append(f"  ✓ `{model_choice}` complete")
            except Exception as exc:
                row = {
                    "model": model_choice,
                    "status": f"failed: {exc}",
                    "bundle": "",
                }
                for metric_name in selected_metrics:
                    row[metric_name] = None
                rows.append(row)
                status_lines.append(f"  ⚠️ `{model_choice}` failed")

            yield "\n".join(status_lines), _sorted_frame(rows)

        status_lines.extend([
            "",
            f"Sorted by `{sweep_sort_metric_val}` in **{sweep_sort_order_val}** order.",
        ])
        yield "\n".join(status_lines), _sorted_frame(rows)

    # The sweep picks the model itself, so it takes every RunConfig input but model_name.
    sweep_config_components = {
        name: comp for name, comp in run_config_components.items() if name != "model_name"
    }

    def sweep_from_ui(values):
        yield from run_model_sweep(
            RunConfig.from_component_values(sweep_config_components, values, model_name=""),
            values[c.sweep_models], values[c.sweep_metrics],
            values[c.sweep_sort_metric], values[c.sweep_sort_order],
        )

    sweep_event = c.sweep_run_btn.click(
        sweep_from_ui,
        inputs={*sweep_config_components.values(),
                c.sweep_models, c.sweep_metrics, c.sweep_sort_metric, c.sweep_sort_order},
        outputs=[c.sweep_status_box, c.sweep_table],
        concurrency_id="heavy_jobs",
        concurrency_limit=1,
    )

    def _plot_cv_significance_scores(metric_name, baseline_name, baseline_scores, challenger_scores_by_name):
        plt = _get_pyplot()
        fig, ax = plt.subplots(figsize=(8.5, 4.8), dpi=130)
        folds = np.arange(1, len(baseline_scores) + 1)
        ax.plot(folds, baseline_scores, marker="o", linewidth=2.2, label=baseline_name, color="#115d52")
        palette = ["#d97706", "#2563eb", "#7c3aed", "#dc2626", "#0f766e"]
        for idx, (name, scores) in enumerate(challenger_scores_by_name.items()):
            ax.plot(
                folds,
                scores,
                marker="o",
                linewidth=1.8,
                linestyle="--",
                label=name,
                color=palette[idx % len(palette)],
                alpha=0.95,
            )
        ax.set_title(f"Fold-by-fold comparison for {metric_name}")
        ax.set_xlabel("Fold")
        ax.set_ylabel(metric_name.replace("_", " ").title())
        ax.set_xticks(folds)
        ax.grid(alpha=0.22, linestyle=":")
        ax.legend(frameon=False, fontsize=8)
        fig.tight_layout()
        plt.close(fig)  # rendered by Gradio after return; frees pyplot's reference
        return fig

    # ── FastAPI generator ─────────────────────────────────────────────────────

    def gen_fastapi(bundle_path, mod):
        if not bundle_path:
            return _format_status_block("FastAPI export", "No bundle available. Train a model first.", success=False)
        try:
            from export.fastapi_generator import generate_fastapi_app
            path = generate_fastapi_app(bundle_path, mod)
            return _format_status_block(
                "FastAPI export",
                (f"FastAPI app written to:\n  {path}\n\n"
                 f"Start server (reachable only from this computer):\n"
                 f'  cd "{bundle_path}"\n'
                 f"  pip install -r requirements_serve.txt\n"
                 f"  python main.py\n\n"
                 f"Then open http://127.0.0.1:8000/docs"),
            )
        except Exception as e:
            return _format_status_block("FastAPI export", f"Error: {e}", success=False)

    c.fastapi_btn.click(gen_fastapi, inputs=[c.bundle_out, c.modality], outputs=c.fastapi_status_box)

    # ── Docker generator ──────────────────────────────────────────────────────

    def gen_docker(bundle_path, mod):
        if not bundle_path:
            return _format_status_block("Docker export", "No bundle available. Train a model first.", success=False)
        try:
            from export.docker_generator import generate_docker_bundle
            paths = generate_docker_bundle(bundle_path, mod)
            files = "\n  ".join(paths)
            return _format_status_block(
                "Docker export",
                (f"Docker files written:\n  {files}\n\n"
                 f"Build & run (reachable only from this computer):\n"
                 f'  cd "{bundle_path}"\n'
                 f"  docker compose up --build\n\n"
                 f"See README_deploy.md for exposing it with an API key."),
            )
        except Exception as e:
            return _format_status_block("Docker export", f"Error: {e}", success=False)

    c.docker_btn.click(gen_docker, inputs=[c.bundle_out, c.modality], outputs=c.docker_status_box)

    # ── Model card generator ──────────────────────────────────────────────────

    def gen_model_card(bundle_path, mod, mname, tmode, tsk, eval_sum):
        if not bundle_path:
            return _format_status_block("Model card", "No bundle available. Train a model first.", success=False)
        try:
            from export.model_card import generate_model_card
            import json
            with open(f"{bundle_path}/labels.json") as f:
                labels_dict = json.load(f)
            classes = list(labels_dict.values())
            path = generate_model_card(
                bundle_path=bundle_path, modality=mod, model_name=mname,
                training_mode=tmode, task=tsk, classes=classes,
                hyperparams={}, history=[], eval_summary=eval_sum)
            return _format_status_block("Model card", f"Model card saved:\n  {path}")
        except Exception as e:
            return _format_status_block("Model card", f"Error: {e}", success=False)

    c.model_card_btn.click(gen_model_card,
                         inputs=[c.bundle_out, c.modality, c.model_name,
                                 c.training_mode, c.task, c.eval_summary_box],
                         outputs=c.model_card_status_box)

    # ── Streamlit dashboard generator ─────────────────────────────────────────

    def gen_streamlit(bundle_path_val, mod, mname, tmode, tsk, eval_summary_val):
        if not bundle_path_val:
            return "Set the bundle path first (train a model or select one)."
        try:
            from export.streamlit_generator import generate_streamlit_dashboard
            # Try to load history and metrics from run_history.json
            run_history = []
            run_metrics = {}
            run_classes = []
            try:
                from training.run_comparison import load_history
                all_runs = load_history()
                for run in reversed(all_runs):
                    if run.get("bundle_path") == bundle_path_val:
                        run_history = run.get("history", [])
                        run_metrics = run.get("metrics") or {}
                        break
            except Exception:
                pass
            # Try to load classes from labels.json
            try:
                import json as _json
                labels_file = os.path.join(bundle_path_val, "labels.json")
                if os.path.isfile(labels_file):
                    with open(labels_file, encoding="utf-8") as fh:
                        run_classes = list(_json.load(fh).values())
            except Exception:
                pass

            out_path = generate_streamlit_dashboard(
                bundle_path=bundle_path_val,
                modality=mod,
                model_name=mname,
                task=tsk,
                training_mode=tmode,
                history=run_history,
                metrics=run_metrics,
                classes=run_classes,
                eval_summary=eval_summary_val,
            )
            return (
                f"### Streamlit dashboard exported\n\n"
                f"```\n{out_path}\n```\n\n"
                f"Run it with:\n\n"
                f"```bash\npip install streamlit pandas matplotlib numpy\n"
                f"streamlit run \"{out_path}\"\n```"
            )
        except Exception as e:
            return f"### Export failed\n\n```\n{e}\n```"

    c.streamlit_btn.click(
        gen_streamlit,
        inputs=[c.bundle_out, c.modality, c.model_name, c.training_mode, c.task, c.eval_summary_box],
        outputs=c.streamlit_status,
    )

    # ── Run history ───────────────────────────────────────────────────────────

    def refresh_history():
        try:
            from training.run_comparison import history_dataframe, HISTORY_FILE
            if not HISTORY_FILE.exists():
                return gr.update(value=[], headers=HISTORY_COLUMNS)
            df = history_dataframe()
            if not len(df):
                return gr.update(value=[], headers=HISTORY_COLUMNS)
            ordered = df.reindex(columns=HISTORY_COLUMNS)
            return gr.update(value=ordered.values.tolist(), headers=HISTORY_COLUMNS)
        except Exception as e:
            logger.warning("refresh_history failed: %s", e)
            return gr.update(value=[], headers=HISTORY_COLUMNS)

    def compare_runs(ids_str):
        try:
            ids = [int(x.strip()) for x in ids_str.split(",") if x.strip()]
            if not ids:
                return (
                    "### ⚠️ Run comparison\n\nEnter one or more run IDs from the history table, for example `0` or `0,1`.",
                    None,
                )
            from training.run_comparison import compare_runs_plot
            return (
                f"### ✅ Run comparison\n\nComparing run IDs: `{', '.join(str(i) for i in ids)}`",
                compare_runs_plot(ids),
            )
        except Exception as e:
            logger.warning("compare_runs failed: %s", e)
            return (
                f"### ⚠️ Run comparison\n\nCould not compare runs.\n\n```text\n{e}\n```",
                None,
            )

    c.refresh_btn.click(refresh_history, outputs=c.history_df)
    c.compare_btn.click(compare_runs, inputs=c.compare_ids, outputs=[c.compare_status_box, c.compare_plot])
    # Only auto-load history on page load if the history file already exists —
    # avoids a redundant disk read (and silent failure) on a fresh install.
    from training.run_comparison import HISTORY_FILE as _HIST_FILE
    if _HIST_FILE.exists():
        demo.load(refresh_history, outputs=c.history_df)

    demo.load(
        lambda: gr.update(choices=_list_bundles()),
        outputs=c.infer_bundle_path,
    )

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

    c.det_img_btn.click(
        detect_image,
        inputs=[c.det_img_input, c.det_model, c.det_conf, c.det_iou],
        outputs=[c.det_img_output, c.det_img_stats_md, c.det_img_chart],
    )

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

    det_vid_event = c.det_vid_btn.click(
        detect_video,
        inputs=[c.det_vid_input, c.det_model, c.det_conf, c.det_iou],
        outputs=[c.det_vid_status, c.det_vid_output, c.det_vid_stats_md, c.det_vid_chart],
        concurrency_id="heavy_jobs",
        concurrency_limit=1,
    )

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

    yolo_event = c.yolo_train_btn.click(
        train_yolo_custom,
        inputs=[c.yolo_data_path, c.yolo_model_size, c.yolo_epochs, c.yolo_batch],
        outputs=[c.yolo_status_box, c.yolo_result_md],
        concurrency_id="heavy_jobs",
        concurrency_limit=1,
    )

    # Stop sets the cooperative flag checked between training batches (the
    # cancel alone only lands at the next yield) and cancels every heavy job.
    c.stop_btn.click(
        _cancel.request_stop,
        cancels=[train_event, cv_event, cv_sig_event, hparam_event, sweep_event, det_vid_event, yolo_event],
    )

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

    # ── Try Your Model — inference callbacks ──────────────────────────────────

    def _predict(bundle_path, predict_fn_name, inp):
        if not bundle_path:
            return "*Set the bundle path above first.*", None
        try:
            from ui.inference_helpers import (
                predict_image, predict_text, predict_tabular, predict_audio, predict_timeseries,
                predictions_to_markdown, predictions_to_chart)
            fn = {"image": predict_image, "text": predict_text,
                  "tabular": predict_tabular, "audio": predict_audio, "timeseries": predict_timeseries}[predict_fn_name]
            if predict_fn_name == "tabular":
                if not inp or not str(inp).strip():
                    return (
                        "### ⚠️ Missing tabular input\n\nPaste a JSON object with feature values before running prediction.",
                        None,
                    )
                import json
                try:
                    inp = json.loads(inp) if isinstance(inp, str) else inp
                except json.JSONDecodeError as exc:
                    return (
                        f"### ⚠️ Invalid JSON\n\n```text\n{exc}\n```\n\n"
                        "Use a JSON object like "
                        "`{\"study_hours\": 12, \"attendance_pct\": 96}`.",
                        None,
                    )
            elif predict_fn_name == "timeseries":
                if not inp or not str(inp).strip():
                    return (
                        "### ⚠️ Missing time-series input\n\nPaste a JSON array of sequential rows before running prediction.",
                        None,
                    )
                import json
                try:
                    inp = json.loads(inp) if isinstance(inp, str) else inp
                except json.JSONDecodeError as exc:
                    return (
                        f"### ⚠️ Invalid JSON\n\n```text\n{exc}\n```\n\n"
                        "Use a JSON array like "
                        "`[{\"sensor_a\": 0.12, \"sensor_b\": 1.4}, {\"sensor_a\": 0.18, \"sensor_b\": 1.3}]`.",
                        None,
                    )
            elif predict_fn_name == "text" and not str(inp or "").strip():
                return "### ⚠️ Missing text input\n\nEnter some text before running prediction.", None
            elif predict_fn_name in {"image", "audio"} and inp is None:
                return f"### ⚠️ Missing {predict_fn_name} input\n\nUpload a file before running prediction.", None
            preds = fn(bundle_path, inp)
            return predictions_to_markdown(preds), predictions_to_chart(preds)
        except Exception as e:
            return f"### ⚠️ Prediction error\n\n```text\n{e}\n```", None

    c.infer_img_btn.click(
        lambda bp, img: _predict(bp, "image", img),
        inputs=[c.infer_bundle_path, c.infer_img_input],
        outputs=[c.infer_img_md, c.infer_img_chart],
    )
    def _predict_text_with_explanation(bp, txt):
        md, chart = _predict(bp, "text", txt)
        explain_html = ""
        if bp and txt and txt.strip():
            try:
                from ui.inference_helpers import explain_text_prediction
                explain_html = explain_text_prediction(bp, txt)
            except Exception:
                pass
        return md, chart, explain_html

    c.infer_text_btn.click(
        _predict_text_with_explanation,
        inputs=[c.infer_bundle_path, c.infer_text_input],
        outputs=[c.infer_text_md, c.infer_text_chart, c.infer_text_explain],
    )
    c.infer_tab_btn.click(
        lambda bp, fv: _predict(bp, "tabular", fv),
        inputs=[c.infer_bundle_path, c.infer_tab_input],
        outputs=[c.infer_tab_md, c.infer_tab_chart],
    )
    c.infer_tab_fill_btn.click(
        _suggest_tabular_example,
        inputs=[c.data_path, c.modality, c.label_col, c.infer_tab_input],
        outputs=c.infer_tab_input,
    )
    c.infer_ts_fill_btn.click(
        _suggest_timeseries_example,
        inputs=[c.data_path, c.modality, c.label_col, c.time_col, c.infer_ts_input],
        outputs=c.infer_ts_input,
    )
    c.infer_aud_btn.click(
        lambda bp, aud: _predict(bp, "audio", aud),
        inputs=[c.infer_bundle_path, c.infer_aud_input],
        outputs=[c.infer_aud_md, c.infer_aud_chart],
    )
    c.infer_ts_btn.click(
        lambda bp, ts: _predict(bp, "timeseries", ts),
        inputs=[c.infer_bundle_path, c.infer_ts_input],
        outputs=[c.infer_ts_md, c.infer_ts_chart],
    )

    # ── Batch prediction ──────────────────────────────────────────────────────

    def _batch_predict(bundle_path: str, folder_path: str):
        if not bundle_path:
            return None, "⚠️ Select a bundle first."
        folder = Path(folder_path.strip()) if folder_path else None
        if not folder or not folder.is_dir():
            return None, f"⚠️ Folder not found: `{folder_path}`"

        try:
            from ui.inference_helpers import (
                _cached_bundle, predict_image, predict_audio, predict_text,
            )
            bundle = _cached_bundle(bundle_path)
            mod = bundle.get("preprocessing", {}).get("modality", "image")
        except Exception as exc:
            return None, f"⚠️ Could not load bundle: {exc}"

        _EXT_MAP = {
            "image":     {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"},
            "audio":     {".wav", ".mp3", ".flac", ".ogg", ".m4a"},
            "text":      {".txt"},
            "video":     {".mp4", ".avi", ".mov", ".mkv", ".webm"},
        }
        _FN_MAP = {
            "image": predict_image,
            "audio": predict_audio,
            "text":  predict_text,
        }
        fn = _FN_MAP.get(mod)
        if fn is None:
            return None, f"⚠️ Batch prediction is not yet supported for **{mod}** modality."

        exts = _EXT_MAP.get(mod, set())
        files = sorted(f for f in folder.rglob("*") if f.is_file() and f.suffix.lower() in exts)
        if not files:
            return None, f"⚠️ No {mod} files found in `{folder_path}`."

        rows = []
        for fpath in files:
            try:
                inp = fpath.read_text(encoding="utf-8", errors="replace") if mod == "text" else str(fpath)
                preds = fn(bundle_path, inp)
                top = preds[0] if preds else {}
                if top.get("error"):
                    rows.append([fpath.name, "ERROR", str(top["error"])])
                else:
                    rows.append([fpath.name, top.get("label", "?"), f"{top.get('confidence', 0)*100:.1f}%"])
            except Exception as exc:
                rows.append([fpath.name, "ERROR", str(exc)])

        import pandas as pd
        df = pd.DataFrame(rows, columns=["file", "prediction", "confidence"])
        status = f"✅ Scored **{len(rows)}** files."
        return df, status

    def _batch_save_csv(df):
        if df is None or (hasattr(df, "empty") and df.empty):
            return "⚠️ No results to save."
        import pandas as pd
        out_dir = _OUTPUTS_ROOT
        out_dir.mkdir(parents=True, exist_ok=True)
        import time as _time
        ts = _time.strftime("%Y%m%d_%H%M%S")
        path = out_dir / f"batch_predictions_{ts}.csv"
        pd.DataFrame(df).to_csv(path, index=False)
        return f"✅ Saved to `{path}`"

    c.batch_run_btn.click(
        _batch_predict,
        inputs=[c.infer_bundle_path, c.batch_folder],
        outputs=[c.batch_df, c.batch_status],
    )
    c.batch_save_btn.click(
        _batch_save_csv,
        inputs=[c.batch_df],
        outputs=[c.batch_save_status],
    )


# Gradio 4 marked dark mode with a `dark` class on <html>; Gradio 6 puts it on
# <body>. styles/app.css defines its dark tokens and ~20 overrides under `html.dark`,
# so mirror the body class onto <html> (and keep it in sync if it changes).
_SYNC_DARK_CLASS_JS = """
() => {
  const sync = () => document.documentElement.classList.toggle(
    "dark", document.body.classList.contains("dark"));
  sync();
  new MutationObserver(sync).observe(document.body, {attributes: true, attributeFilter: ["class"]});
}
"""


def launch_app(
    *,
    inbrowser: bool = True,
    server_port: int = 7860,
    prevent_thread_lock: bool = False,
    frontend: bool = True,
    show_api: bool = True,
):
    logger.info("Device: %s", DEVICE)
    return demo.launch(
        share=False,
        inbrowser=inbrowser,
        server_name="127.0.0.1",
        server_port=server_port,
        prevent_thread_lock=prevent_thread_lock,
        theme=APP_THEME,
        css=_APP_CSS_PATH.read_text(encoding="utf-8"),
        js=_SYNC_DARK_CLASS_JS,
        # Gradio 6 replaced show_api with footer_links.
        footer_links=["api", "gradio", "settings"] if show_api else ["gradio", "settings"],
        # Gradio 6 only serves files from the working directory, the temp dir
        # and allowed_paths; outputs/ holds trained bundles, previews and plots.
        allowed_paths=[str(_OUTPUTS_ROOT)],
        _frontend=frontend,
    )


if __name__ == "__main__":
    launch_app()

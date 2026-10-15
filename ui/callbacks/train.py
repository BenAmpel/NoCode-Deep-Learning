"""Train tab: the Train button, what follows a finished run, and Stop."""
from __future__ import annotations

import traceback

import gradio as gr

from pipeline import RUN_CONFIG_FIELDS, RunConfig
from pipeline.context import RunResult
from pipeline.progress import RunLog, Update
from pipeline.runner import (
    cleanup as cleanup_run,
    execute as execute_run,
    preflight_errors,
    prepare_run,
)
from training import cancel as _cancel
from ui.callbacks.data import update_workspace_cockpit
from ui.helpers import (
    _DEFAULT_CLASSIFICATION_NOTE,
    _DEFAULT_DIAGNOSTICS_NOTE,
    _DEFAULT_EXPLANATION_NOTE,
    _DEFAULT_SAMPLE_REVIEW_NOTE,
    _build_action_items,
    _build_dashboard_kpi,
    _build_evaluation_notes,
    _format_eval_kpis,
    _format_eval_summary,
    _format_live_training_metrics_card,
    _list_bundles,
    _loss_plot,
    _pick_primary_chart,
    _suggest_tabular_example,
)
from ui.layout import Components


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


def register(c: Components) -> dict:
    """Wire this area's events to the components in ``c``.

    Returns its long-running events by name, for the Stop button.
    """
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

    return {"train_event": train_event}


def register_stop_button(c: Components, heavy_events: list) -> None:
    """Stop sets the cooperative stop flag and cancels every heavy job."""
    # Stop sets the cooperative flag checked between training batches (the
    # cancel alone only lands at the next yield) and cancels every heavy job.
    c.stop_btn.click(
        _cancel.request_stop,
        cancels=heavy_events,
    )

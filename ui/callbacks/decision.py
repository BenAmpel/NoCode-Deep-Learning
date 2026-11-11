"""Evaluate tab: threshold and cost panel, and group fairness panel.

Both panels read the saved bundle (``c.bundle_out``), so they refresh after
every training run and never re-run the model.
"""
from __future__ import annotations

import logging

import gradio as gr

from eval.decision import (
    DecisionError,
    _num,
    best_threshold,
    decision_report,
    default_positive_class,
    fairness_report,
    load_predictions,
    positive_scores,
)
from ui.layout import Components

logger = logging.getLogger("nocode_dl")

_EMPTY_DECISION = "*Train a classification model to tune its threshold.*"
_EMPTY_FAIRNESS = "*Train a tabular classification model to check group fairness.*"


def _prefer_group(cols: list[str]) -> str | None:
    for hint in ("gender", "sex", "race", "ethnicity", "age_group", "group"):
        for col in cols:
            if col.lower() == hint:
                return col
    return cols[0] if cols else None


def on_bundle_change(bundle_path):
    """Reset both panels for a new bundle: class choices and group columns."""
    try:
        pred = load_predictions(bundle_path)
    except DecisionError as exc:
        msg = f"*{exc}*" if bundle_path else _EMPTY_DECISION
        return (gr.update(choices=[], value=None), gr.update(value=0.5), msg, None,
                gr.update(choices=[], value=None), _EMPTY_FAIRNESS if not bundle_path else f"*{exc}*", None)
    except Exception as exc:  # never break the Evaluate tab
        logger.warning("Decision panel failed to load bundle: %s", exc)
        return (gr.update(choices=[], value=None), gr.update(value=0.5), f"❌ {exc}", None,
                gr.update(choices=[], value=None), _EMPTY_FAIRNESS, None)
    pos = default_positive_class(pred.classes)
    group_cols = sorted(pred.groups)
    group = _prefer_group(group_cols)
    dec_md, dec_fig = _decision(bundle_path, pos, 0.5, 0, 0, 0)
    if group_cols:
        fair_md, fair_df = _fairness(bundle_path, group, pos, 0.5)
    else:
        fair_md, fair_df = ("*Group fairness needs a tabular classifier. "
                            "This run saved no group columns.*", None)
    return (gr.update(choices=pred.classes, value=pos), gr.update(value=0.5), dec_md, dec_fig,
            gr.update(choices=group_cols, value=group), fair_md, fair_df)


def _decision(bundle_path, pos, threshold, tp_value, fp_cost, fn_cost):
    if not bundle_path:
        return _EMPTY_DECISION, None
    try:
        return decision_report(bundle_path, pos, threshold, tp_value, fp_cost, fn_cost)
    except DecisionError as exc:
        return f"*{exc}*", None
    except Exception as exc:
        logger.warning("Threshold panel failed: %s", exc)
        return f"❌ Could not score this threshold: {exc}", None


def _fairness(bundle_path, group_col, pos, threshold):
    if not bundle_path:
        return _EMPTY_FAIRNESS, None
    try:
        return fairness_report(bundle_path, group_col, pos, threshold)
    except DecisionError as exc:
        return f"*{exc}*", None
    except Exception as exc:
        logger.warning("Fairness panel failed: %s", exc)
        return f"❌ Could not compute group rates: {exc}", None


def update_panels(bundle_path, pos, threshold, tp_value, fp_cost, fn_cost, group_col):
    dec_md, dec_fig = _decision(bundle_path, pos, threshold, tp_value, fp_cost, fn_cost)
    fair_md, fair_df = _fairness(bundle_path, group_col, pos, threshold) if group_col else (
        (_EMPTY_FAIRNESS if not bundle_path else "*Pick a group column.*"), None)
    return dec_md, dec_fig, fair_md, fair_df


def use_best_threshold(bundle_path, pos, tp_value, fp_cost, fn_cost):
    try:
        pred = load_predictions(bundle_path)
        if pos not in pred.classes:
            pos = default_positive_class(pred.classes)
        idx = pred.classes.index(pos)
        if all(abs(_num(v, 0.0)) == 0 for v in (tp_value, fp_cost, fn_cost)):
            return gr.update()          # no costs entered: nothing to optimize
        t, _, _ = best_threshold(pred.y_true == idx, positive_scores(pred, idx),
                                 tp_value, fp_cost, fn_cost)
        return gr.update(value=round(min(max(t, 0.0), 1.0), 4))
    except Exception:
        return gr.update()


def register(c: Components) -> None:
    c.bundle_out.change(
        on_bundle_change, inputs=[c.bundle_out],
        outputs=[c.decision_pos_class, c.decision_threshold, c.decision_md, c.decision_plot,
                 c.fairness_group_col, c.fairness_md, c.fairness_df],
        show_progress="hidden",
    )
    inputs = [c.bundle_out, c.decision_pos_class, c.decision_threshold,
              c.decision_tp_value, c.decision_fp_cost, c.decision_fn_cost, c.fairness_group_col]
    outputs = [c.decision_md, c.decision_plot, c.fairness_md, c.fairness_df]
    c.decision_threshold.release(update_panels, inputs=inputs, outputs=outputs, show_progress="hidden")
    for comp in (c.decision_pos_class, c.decision_tp_value, c.decision_fp_cost,
                 c.decision_fn_cost, c.fairness_group_col):
        comp.input(update_panels, inputs=inputs, outputs=outputs, show_progress="hidden")
    c.decision_best_btn.click(
        use_best_threshold,
        inputs=[c.bundle_out, c.decision_pos_class, c.decision_tp_value, c.decision_fp_cost, c.decision_fn_cost],
        outputs=[c.decision_threshold],
    ).then(update_panels, inputs=inputs, outputs=outputs, show_progress="hidden")


__all__ = ["register", "on_bundle_change", "update_panels", "use_best_threshold"]

"""Pitfall coach: common machine-learning mistakes, spotted from a finished run.

Pure functions over what evaluation already produced (validation labels and
predictions, the metrics dict, split sizes). No torch, no gradio, no training.

Each finding is a plain dict, so it can go straight into ``run_record.json``:

    {"id", "severity", "title", "evidence", "why", "try"}

``find_pitfalls`` returns ``None`` when the task has no checks (clustering,
anomaly detection, multi-label, Autoencoder runs) and ``[]`` when the checks
ran and found nothing.
"""
from __future__ import annotations

import math
from collections import Counter
from typing import Any, Sequence

SEVERITY_ORDER = {"critical": 0, "warning": 1, "info": 2}
SEVERITY_ICON = {"critical": "🔴", "warning": "🟠", "info": "🔵"}

# Thresholds (each has a fires / doesn't-fire test pair in tests/test_pitfalls.py).
SMALL_VAL_N = 30            # fewer validation samples than this: scores are noisy
SMALL_VAL_PER_CLASS = 5     # fewer validation samples of some class than this
TINY_VAL_N = 20             # below this, only the small-validation finding is shown
BASELINE_MARGIN = 0.02      # accuracy within 2 points of always guessing the majority
COLLAPSE_SHARE = 0.95       # 95%+ of predictions are one class
IMBALANCE_RATIO = 3.0       # largest class at least 3x the smallest
PERFECT_SCORE = 0.99        # accuracy or R² this high: check for leakage
FEW_TRAIN_PER_CLASS = 20    # from-scratch deep models with fewer per class than this

MAX_SHOWN = 3


def _finding(fid: str, severity: str, title: str, evidence: str, why: str, try_: str) -> dict:
    return {"id": fid, "severity": severity, "title": title,
            "evidence": evidence, "why": why, "try": try_}


def wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval for a proportion (well behaved at small n and at 0 or 1)."""
    if n <= 0:
        return (0.0, 1.0)
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def _pct(x: float) -> str:
    return f"{x * 100:.1f}%"


def _name(label: Any, classes: Sequence[Any] | None) -> str:
    try:
        idx = int(label)
        if classes and 0 <= idx < len(classes):
            return f"'{classes[idx]}'"
    except (TypeError, ValueError):
        pass
    return f"'{label}'"


def _flat_labels(values: Any) -> list | None:
    """1-D list of hashable labels, or None for multi-label / empty / odd shapes."""
    if values is None:
        return None
    out = []
    for v in values:
        if isinstance(v, (list, tuple)):
            return None
        out.append(v)
    return out or None


# ── classification ───────────────────────────────────────────────────────────

def _classification_findings(
    y_true: list, y_pred: list, metrics: dict, classes: Sequence[Any] | None,
    n_train: int | None, from_scratch_deep: bool,
) -> list[dict]:
    n = len(y_true)
    correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)
    acc = correct / n
    true_counts = Counter(y_true)
    pred_counts = Counter(y_pred)
    majority_label, majority_n = true_counts.most_common(1)[0]
    majority_share = majority_n / n
    n_classes = max(len(classes or []), len(true_counts))

    findings: list[dict] = []

    # Small validation set: how much one sample is worth, and the interval.
    smallest_present = min(true_counts.values())
    index_labels = bool(classes) and all(
        isinstance(k, int) and 0 <= k < len(classes) for k in true_counts)
    missing = [c for c in range(len(classes)) if c not in true_counts] if index_labels else []
    small = n < SMALL_VAL_N or smallest_present < SMALL_VAL_PER_CLASS or bool(missing)
    if small:
        lo, hi = wilson_interval(correct, n)
        parts = [f"The validation set has {n} samples, so one sample moves accuracy by "
                 f"{100 / n:.1f} points. The 95% interval for your {_pct(acc)} accuracy is "
                 f"{_pct(lo)} to {_pct(hi)}."]
        if missing:
            parts.append(f"{len(missing)} class(es) never appear in the validation data.")
        elif smallest_present < SMALL_VAL_PER_CLASS:
            parts.append(f"The rarest class has only {smallest_present} validation sample(s).")
        findings.append(_finding(
            "small_validation", "warning", "The validation set is too small to trust the score",
            " ".join(parts),
            "With few validation samples, a lucky or unlucky split can swing the score a lot, "
            "and two models can look different when they are not.",
            "Use more data or a larger validation fraction, and compare models on the interval, "
            "not on the single number.",
        ))
        if n < TINY_VAL_N:
            return findings

    # Predictions collapse to one class.
    top_pred, top_pred_n = pred_counts.most_common(1)[0]
    collapsed = (len(true_counts) >= 2 and top_pred_n / n >= COLLAPSE_SHARE
                 and majority_share < COLLAPSE_SHARE)
    if collapsed:
        findings.append(_finding(
            "collapsed_predictions", "critical", "The model predicts almost only one class",
            f"{_pct(top_pred_n / n)} of validation predictions are {_name(top_pred, classes)}, "
            f"but that class is {_pct(true_counts.get(top_pred, 0) / n)} of the true labels.",
            "A model that always gives the same answer has not learned the classes; its accuracy "
            "is just the share of that class.",
            "Turn on class weights or balance the data, check that labels match their inputs, and "
            "try a lower learning rate or more epochs.",
        ))
    # Accuracy no better than always guessing the majority class.
    elif len(true_counts) >= 2 and acc - majority_share < BASELINE_MARGIN:
        findings.append(_finding(
            "below_baseline", "critical", "Accuracy is no better than always guessing the most common class",
            f"Accuracy is {_pct(acc)}. Always answering {_name(majority_label, classes)} would score "
            f"{_pct(majority_share)} on this validation set.",
            "Accuracy only means something next to this baseline. Here the model adds little or "
            "nothing over a constant guess.",
            "Report balanced accuracy and macro F1, use class weights, and check the features "
            "actually carry signal about the label.",
        ))

    # Imbalanced classes with accuracy as the headline.
    if len(true_counts) >= 2 and majority_n / smallest_present >= IMBALANCE_RATIO:
        minority_label = min(true_counts, key=lambda k: true_counts[k])
        bal = metrics.get("balanced_accuracy")
        f1m = metrics.get("f1_macro")
        extra = ""
        if isinstance(bal, (int, float)) and math.isfinite(bal):
            extra = f" Balanced accuracy is {_pct(float(bal))}"
            if isinstance(f1m, (int, float)) and math.isfinite(f1m):
                extra += f" and macro F1 is {float(f1m):.2f}"
            extra += f", against {_pct(acc)} plain accuracy."
        findings.append(_finding(
            "class_imbalance", "info", "The classes are imbalanced, so accuracy flatters the model",
            f"In the validation data, {_name(majority_label, classes)} has {majority_n} samples and "
            f"{_name(minority_label, classes)} has {true_counts[minority_label]} "
            f"({majority_n / smallest_present:.1f} to 1).{extra}",
            "When one class dominates, a model can score high accuracy while missing the rare class "
            "entirely, and the rare class is often the one that matters.",
            "Lead with balanced accuracy or macro F1, read the per-class recall in the confusion "
            "matrix, and consider class weights.",
        ))

    # Suspiciously perfect.
    if acc >= PERFECT_SCORE and n >= TINY_VAL_N:
        findings.append(_finding(
            "suspiciously_perfect", "warning", "A near-perfect score: check for leakage",
            f"Validation accuracy is {_pct(acc)} on {n} samples.",
            "Real tasks are rarely this easy. Perfect scores often mean the answer leaked into the "
            "inputs: an ID or date column, a feature computed from the label, or the same rows in "
            "both training and validation.",
            "Remove ID-like and label-derived columns, check for duplicate rows across the split, "
            "and test on data collected separately.",
        ))

    # Too few training samples per class for a deep model trained from scratch.
    if from_scratch_deep and n_train and n_classes:
        per_class = n_train / n_classes
        if per_class < FEW_TRAIN_PER_CLASS:
            findings.append(_finding(
                "few_training_samples", "info", "Very few training samples per class for a model trained from scratch",
                f"{n_train} training samples across {n_classes} classes is about {per_class:.0f} per class.",
                "Deep networks trained from random weights need many examples per class; with few, "
                "they memorise the training set instead of learning the pattern.",
                "Use a pretrained model in fine-tune mode, add data augmentation, or try a classical "
                "model (Random Forest, Logistic Regression) as a baseline.",
            ))

    return findings


# ── regression ───────────────────────────────────────────────────────────────

def _regression_findings(metrics: dict, n: int) -> list[dict]:
    r2 = metrics.get("r2")
    if not isinstance(r2, (int, float)) or not math.isfinite(r2):
        return []
    r2 = float(r2)
    findings: list[dict] = []
    if n < SMALL_VAL_N:
        findings.append(_finding(
            "small_validation", "warning", "The validation set is too small to trust the score",
            f"The validation set has {n} samples. R² and error measures from so few points can "
            "change a lot with a different split.",
            "With few validation samples, a lucky or unlucky split can swing the score a lot.",
            "Use more data or a larger validation fraction before comparing models.",
        ))
        if n < TINY_VAL_N:
            return findings
    if r2 <= 0:
        findings.append(_finding(
            "below_baseline", "critical", "Worse than always predicting the average",
            f"R² is {r2:.3f}. An R² of 0 is what you get by predicting the mean of the validation "
            "targets for every row; below 0 is worse than that.",
            "The model's predictions add no information over a constant guess.",
            "Check the target column and scaling, try a simpler model (Linear Regression) as a "
            "baseline, and train longer or with a lower learning rate.",
        ))
    elif r2 >= PERFECT_SCORE:
        findings.append(_finding(
            "suspiciously_perfect", "warning", "A near-perfect score: check for leakage",
            f"R² is {r2:.4f} on {n} validation samples.",
            "Real targets are rarely predicted this well. A feature may be the target in disguise "
            "(computed from it, or measured after it).",
            "Look at the most important features for anything derived from the target, and check "
            "for duplicate rows across the split.",
        ))
    return findings


# ── entry points ─────────────────────────────────────────────────────────────

def find_pitfalls(
    *,
    task: str,
    model_name: str = "",
    metrics: dict | None = None,
    y_true: Any = None,
    y_pred: Any = None,
    classes: Sequence[Any] | None = None,
    n_train: int | None = None,
    from_scratch_deep: bool = False,
) -> list[dict] | None:
    """All findings for one run, most severe first; None when the task has no checks."""
    metrics = metrics or {}
    if model_name == "Autoencoder":
        return None
    if task == "classification":
        yt, yp = _flat_labels(y_true), _flat_labels(y_pred)
        if yt is None or yp is None or len(yt) != len(yp):
            return None
        findings = _classification_findings(yt, yp, metrics, classes, n_train, from_scratch_deep)
    elif task == "regression":
        n = len(y_true) if y_true is not None else 0
        if n == 0:
            return None
        findings = _regression_findings(metrics, n)
    else:
        return None
    return sorted(findings, key=lambda f: SEVERITY_ORDER.get(f["severity"], 9))


def pitfalls_for_run(ctx: Any) -> list[dict] | None:
    """Gather the inputs from a finished RunContext and run the checks."""
    cfg = ctx.cfg
    art = ctx.evaluation_artifacts or {}
    n_train = None
    try:
        n_train = len(ctx.train_loader.dataset)
    except (AttributeError, TypeError):
        pass
    if n_train is None and cfg.modality == "graph":
        try:
            n_train = len((ctx.extra_data or {})["train_idx"])
        except (KeyError, TypeError):
            pass
    return find_pitfalls(
        task=art.get("task", cfg.task),
        model_name=cfg.model_name,
        metrics=ctx.result.metrics,
        y_true=art.get("y_true"),
        y_pred=art.get("y_pred"),
        classes=ctx.classes,
        n_train=n_train,
        from_scratch_deep=(not ctx.sklearn_model and cfg.training_mode != "fine-tune"),
    )


def render_markdown(findings: list[dict] | None, *, max_shown: int = MAX_SHOWN) -> str:
    """The 'Pitfall check' section for the Run summary card ('' when not applicable)."""
    if findings is None:
        return ""
    if not findings:
        return "### Pitfall check\n✅ Nothing flagged: no common mistakes spotted in this run's validation results."
    lines = ["### Pitfall check"]
    for f in findings[:max_shown]:
        lines += [
            f"**{SEVERITY_ICON.get(f['severity'], '•')} {f['title']}**  ",
            f"{f['evidence']}  ",
            f"*Why it matters:* {f['why']}  ",
            f"*Try:* {f['try']}",
            "",
        ]
    hidden = len(findings) - max_shown
    if hidden > 0:
        lines.append(f"_{hidden} more in the run record (RUN_RECORD.md) inside the bundle._")
    return "\n".join(lines).rstrip()

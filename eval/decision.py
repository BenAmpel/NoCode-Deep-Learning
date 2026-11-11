"""Business decision tools on a trained classifier's validation predictions.

Everything here reads ``evaluation_artifacts.json`` (validation ``y_true`` and
``y_prob``) and ``labels.json`` from a saved bundle, so it works for any
classification bundle without re-running the model.

* Threshold and cost: pick a positive class, move the decision threshold, and
  score each threshold with a dollar value per true positive and a cost per
  false positive (and optionally per false negative).
* Group fairness: compare actual and predicted positive rates across the
  values of one column (which may be a column left out of the model), and
  report the impact ratio (lowest rate / highest rate) against the
  four-fifths rule.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

FOUR_FIFTHS = 0.8
MAX_GROUPS = 50          # a group column with more distinct values is not a "group"
MISSING_GROUP = "(missing)"


class DecisionError(ValueError):
    """A readable reason the decision tools cannot run on this bundle."""


@dataclass
class Predictions:
    classes: list[str]
    y_true: np.ndarray            # (N,) class indices
    y_prob: np.ndarray | None     # (N, C) probabilities, or None
    y_pred: np.ndarray            # (N,) class indices
    groups: dict[str, list]       # column -> per-row raw values (validation order)


def load_predictions(bundle_path: str | Path) -> Predictions:
    """Load validation predictions for a classification bundle."""
    if not bundle_path or not str(bundle_path).strip():
        raise DecisionError("Train a model first (no bundle selected).")
    path = Path(str(bundle_path).strip())
    art_path = path / "evaluation_artifacts.json"
    if not art_path.is_file():
        raise DecisionError("This bundle has no saved validation predictions. Retrain it with this version of the app.")
    try:
        art = json.loads(art_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise DecisionError(f"Could not read the validation predictions: {exc}") from exc
    if art.get("task") not in (None, "classification"):
        raise DecisionError("Threshold, cost, and fairness views need a classification model.")
    try:
        labels = json.loads((path / "labels.json").read_text())
        classes = [labels[k] for k in sorted(labels, key=lambda k: int(k))]
    except Exception:
        classes = []
    y_true = np.asarray(art.get("y_true") or [], dtype=np.int64)
    if y_true.size == 0:
        raise DecisionError("The validation split is empty, so there is nothing to score.")
    y_prob = art.get("y_prob")
    y_prob = None if y_prob is None else np.asarray(y_prob, dtype=np.float64)
    if y_prob is not None and (y_prob.ndim != 2 or y_prob.shape[0] != y_true.size):
        y_prob = None
    y_pred = np.asarray(art.get("y_pred") or [], dtype=np.int64)
    if y_pred.size != y_true.size:
        y_pred = y_prob.argmax(axis=1) if y_prob is not None else np.zeros_like(y_true)
    n_classes = max(len(classes), (y_prob.shape[1] if y_prob is not None else 0),
                    int(y_true.max()) + 1, int(y_pred.max()) + 1 if y_pred.size else 0)
    if len(classes) < n_classes:
        classes = classes + [str(i) for i in range(len(classes), n_classes)]
    groups = art.get("group_values") or {}
    groups = {k: v for k, v in groups.items() if isinstance(v, list) and len(v) == y_true.size}
    return Predictions(classes, y_true, y_prob, y_pred, groups)


def default_positive_class(classes: list[str]) -> str | None:
    """Guess the 'positive' class: 1/true/yes/fraud-like names, else the last class."""
    if not classes:
        return None
    hints = ("1", "true", "yes", "fraud", "escalate", "positive", "fake", "hired", "spam", "churn", "default")
    for hint in hints:
        for c in classes:
            if str(c).strip().lower() == hint:
                return c
    return classes[-1]


def positive_scores(pred: Predictions, pos_idx: int) -> np.ndarray:
    """Probability of the positive class per row (hard 0/1 if no probabilities)."""
    if pred.y_prob is not None and pos_idx < pred.y_prob.shape[1]:
        return np.nan_to_num(pred.y_prob[:, pos_idx], nan=0.0)
    return (pred.y_pred == pos_idx).astype(np.float64)


def confusion_at(y_pos: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, int]:
    """Counts when 'positive' means score >= threshold."""
    flag = scores >= threshold
    return {
        "tp": int(np.sum(flag & y_pos)),
        "fp": int(np.sum(flag & ~y_pos)),
        "fn": int(np.sum(~flag & y_pos)),
        "tn": int(np.sum(~flag & ~y_pos)),
    }


def _num(x, default: float = 0.0) -> float:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return default
    return v if math.isfinite(v) else default


def dollar_score(counts: dict[str, int], value_tp, cost_fp, cost_fn=0.0) -> float:
    """value per caught positive minus costs of false alarms and misses (costs are positive numbers)."""
    return (_num(value_tp) * counts["tp"] - abs(_num(cost_fp)) * counts["fp"]
            - abs(_num(cost_fn)) * counts["fn"])


def candidate_thresholds(scores: np.ndarray) -> np.ndarray:
    """Every distinct score plus the end points, so the sweep finds the exact optimum."""
    uniq = np.unique(np.clip(scores, 0.0, 1.0))
    return np.unique(np.concatenate([[0.0], uniq, [1.0 + 1e-9]]))


def best_threshold(y_pos, scores, value_tp, cost_fp, cost_fn=0.0) -> tuple[float, float, dict]:
    """Threshold with the highest dollar score (ties go to the higher threshold, fewer flags)."""
    best = None
    for t in candidate_thresholds(scores):
        counts = confusion_at(y_pos, scores, t)
        s = dollar_score(counts, value_tp, cost_fp, cost_fn)
        if best is None or s >= best[1]:
            best = (float(min(t, 1.0)), s, counts)
    return best


def decision_report(bundle_path, positive_class, threshold, value_tp, cost_fp, cost_fn=0.0):
    """Return (markdown, figure) for the threshold and cost panel."""
    pred = load_predictions(bundle_path)
    if len(pred.classes) < 2:
        raise DecisionError("The model has only one class, so there is no decision to tune.")
    if positive_class not in pred.classes:
        positive_class = default_positive_class(pred.classes)
    pos_idx = pred.classes.index(positive_class)
    y_pos = pred.y_true == pos_idx
    scores = positive_scores(pred, pos_idx)
    t = min(max(_num(threshold, 0.5), 0.0), 1.0)
    counts = confusion_at(y_pos, scores, t)
    score = dollar_score(counts, value_tp, cost_fp, cost_fn)
    bt, bs, bcounts = best_threshold(y_pos, scores, value_tp, cost_fp, cost_fn)
    caught = counts["tp"]
    total_pos = int(y_pos.sum())
    recall = caught / total_pos if total_pos else 0.0
    flagged = counts["tp"] + counts["fp"]
    precision = caught / flagged if flagged else 0.0
    lines = [
        f"**Positive class:** `{positive_class}` · **Threshold:** {t:.2f} · **Validation rows:** {len(y_pos):,}",
        "",
        "| | Predicted positive | Predicted negative |",
        "|---|---:|---:|",
        f"| **Actually `{positive_class}`** | {counts['tp']:,} caught | {counts['fn']:,} missed |",
        f"| **Actually other** | {counts['fp']:,} false alarms | {counts['tn']:,} correct |",
        "",
        f"Caught {caught:,} of {total_pos:,} ({recall:.0%}). Precision {precision:.0%}.",
        "",
    ]
    if all(abs(_num(v, 0.0)) == 0 for v in (value_tp, cost_fp, cost_fn)):
        lines.append("### Score: enter the dollar values above to find the best threshold.")
    else:
        lines += [
            f"### Score at this threshold: ${score:,.2f}",
            f"Best threshold: **{bt:.2f}** → ${bs:,.2f} "
            f"({bcounts['tp']:,} caught, {bcounts['fp']:,} false alarms, {bcounts['fn']:,} missed).",
        ]
    if pred.y_prob is None:
        lines.append("\n> ⚠️ This model saved no probabilities, so only its hard predictions are scored "
                     "and the threshold has no effect.")
    if len(pred.classes) > 2:
        lines.append(f"\n> ℹ️ Multiclass model: `{positive_class}` is scored against all other classes combined.")
    return "\n".join(lines), _threshold_figure(y_pos, scores, t, bt, value_tp, cost_fp, cost_fn)


def _threshold_figure(y_pos, scores, t, bt, value_tp, cost_fp, cost_fn):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    grid = np.linspace(0.0, 1.0, 101)
    dollars = [dollar_score(confusion_at(y_pos, scores, g), value_tp, cost_fp, cost_fn) for g in grid]
    fig, ax = plt.subplots(figsize=(6, 3.2))
    ax.plot(grid, dollars, color="#2f7d6d")
    ax.axvline(t, color="#888", linestyle="--", label=f"current {t:.2f}")
    ax.axvline(bt, color="#c0504d", linestyle=":", label=f"best {bt:.2f}")
    ax.axhline(0, color="#ccc", linewidth=0.8)
    ax.set_xlabel("Decision threshold")
    ax.set_ylabel("Score ($)")
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    plt.close(fig)
    return fig


# ── Group fairness ────────────────────────────────────────────────────────────

def _clean_group(v) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return MISSING_GROUP
    s = str(v).strip()
    return s if s and s.lower() != "nan" else MISSING_GROUP


def fairness_table(group_values: list, y_pos: np.ndarray, flag: np.ndarray) -> pd.DataFrame:
    """Per-group actual and predicted positive rates."""
    df = pd.DataFrame({"group": [_clean_group(v) for v in group_values],
                       "actual": y_pos.astype(float), "predicted": flag.astype(float)})
    out = df.groupby("group", sort=True).agg(rows=("actual", "size"),
                                             actual_rate=("actual", "mean"),
                                             predicted_rate=("predicted", "mean")).reset_index()
    return out


def impact_ratio(rates: pd.Series) -> float | None:
    """Lowest rate / highest rate; None when the highest rate is 0 (nobody selected)."""
    rates = rates.dropna()
    if len(rates) < 2:
        return None
    hi = rates.max()
    return None if hi <= 0 else float(rates.min() / hi)


def fairness_report(bundle_path, group_col, positive_class, threshold):
    """Return (markdown, table) for the group fairness panel."""
    pred = load_predictions(bundle_path)
    if not pred.groups:
        raise DecisionError("Group fairness needs a tabular classifier trained with this version of the app.")
    if group_col not in pred.groups:
        raise DecisionError("Pick a group column.")
    if positive_class not in pred.classes:
        positive_class = default_positive_class(pred.classes)
    pos_idx = pred.classes.index(positive_class)
    y_pos = pred.y_true == pos_idx
    t = min(max(_num(threshold, 0.5), 0.0), 1.0)
    flag = positive_scores(pred, pos_idx) >= t
    table = fairness_table(pred.groups[group_col], y_pos, flag)
    if len(table) > MAX_GROUPS:
        raise DecisionError(f"`{group_col}` has {len(table)} values in validation; pick a column with at most {MAX_GROUPS}.")
    actual_ir = impact_ratio(table.actual_rate)
    pred_ir = impact_ratio(table.predicted_rate)

    def verdict(ir):
        if ir is None:
            return "n/a (fewer than two groups, or nobody selected)"
        return f"**{ir:.2f}** ({'passes' if ir >= FOUR_FIFTHS else 'fails'} the four-fifths rule)"

    small = table[table.rows < 30].group.tolist()
    lines = [
        f"**Group column:** `{group_col}` · **Positive class:** `{positive_class}` · **Threshold:** {t:.2f}",
        "",
        f"- Impact ratio in the actual labels: {verdict(actual_ir)}",
        f"- Impact ratio in the model's predictions: {verdict(pred_ir)}",
        "",
        "Impact ratio = lowest group's positive rate ÷ highest group's. Below 0.80 fails the EEOC four-fifths rule.",
    ]
    if small:
        lines.append(f"\n> ⚠️ Fewer than 30 validation rows in: {', '.join(map(str, small[:10]))}. Their rates are noisy.")
    show = table.copy()
    show["actual_rate"] = (show.actual_rate * 100).round(1).astype(str) + "%"
    show["predicted_rate"] = (show.predicted_rate * 100).round(1).astype(str) + "%"
    show.columns = ["group", "validation rows", f"actual {positive_class} rate", f"predicted {positive_class} rate"]
    return "\n".join(lines), show


MAX_GROUP_COLUMNS = 20
MAX_GROUP_ROWS = 50_000   # keeps evaluation_artifacts.json small on big datasets


def _json_value(v):
    if v is None:
        return None
    if hasattr(v, "item") and not isinstance(v, (str, bytes)):
        try:
            v = v.item()
        except (ValueError, AttributeError):
            pass
    if isinstance(v, float) and math.isnan(v):
        return None
    if isinstance(v, (str, int, float, bool)):
        return v
    try:
        if pd.isna(v):
            return None
    except (TypeError, ValueError):
        pass
    return str(v)


def group_columns_for(df_val: pd.DataFrame, label_col: str,
                      max_cols: int = MAX_GROUP_COLUMNS) -> dict[str, list]:
    """Low-cardinality columns of the validation rows, saved so the fairness
    panel can group by them later (including columns left out of the model)."""
    out: dict[str, list] = {}
    if df_val is None or len(df_val) == 0 or len(df_val) > MAX_GROUP_ROWS:
        return out
    for col in df_val.columns:
        if str(col) == str(label_col) or len(out) >= max_cols:
            continue
        s = df_val[col]
        try:
            n_unique = s.nunique(dropna=True)
        except TypeError:          # unhashable cells (lists, dicts)
            continue
        if 1 <= n_unique <= MAX_GROUPS:
            out[str(col)] = [_json_value(v) for v in s.tolist()]
    return out

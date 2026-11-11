"""Threshold/cost panel and group fairness panel (eval.decision, ui.callbacks.decision)."""
import json

import numpy as np
import pandas as pd
import pytest

from eval import decision as d


def _bundle(tmp_path, y_true, y_prob=None, y_pred=None, labels=("0", "1"), groups=None,
            task="classification", name="b"):
    path = tmp_path / name
    path.mkdir()
    art = {"task": task, "y_true": list(y_true)}
    if y_prob is not None:
        art["y_prob"] = [list(r) for r in y_prob]
    if y_pred is not None:
        art["y_pred"] = list(y_pred)
    if groups is not None:
        art["group_values"] = groups
    (path / "evaluation_artifacts.json").write_text(json.dumps(art))
    if labels is not None:
        (path / "labels.json").write_text(json.dumps({str(i): c for i, c in enumerate(labels)}))
    return str(path)


def _binary(tmp_path, **kw):
    # 4 positives scored 0.9, 0.8, 0.4, 0.2; 4 negatives scored 0.7, 0.3, 0.1, 0.05
    y_true = [1, 1, 1, 1, 0, 0, 0, 0]
    p1 = [0.9, 0.8, 0.4, 0.2, 0.7, 0.3, 0.1, 0.05]
    return _bundle(tmp_path, y_true, [[1 - p, p] for p in p1], **kw)


# ── loading ────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("path", [None, "", "   "])
def test_no_bundle_is_a_readable_error(path):
    with pytest.raises(d.DecisionError, match="Train a model"):
        d.load_predictions(path)


def test_missing_artifacts_and_bad_json(tmp_path):
    (tmp_path / "empty").mkdir()
    with pytest.raises(d.DecisionError, match="no saved validation"):
        d.load_predictions(tmp_path / "empty")
    bad = tmp_path / "bad"
    bad.mkdir()
    (bad / "evaluation_artifacts.json").write_text("{not json")
    with pytest.raises(d.DecisionError, match="Could not read"):
        d.load_predictions(bad)


def test_regression_and_empty_validation_are_rejected(tmp_path):
    with pytest.raises(d.DecisionError, match="classification"):
        d.load_predictions(_bundle(tmp_path, [1.0, 2.0], task="regression", name="r"))
    with pytest.raises(d.DecisionError, match="empty"):
        d.load_predictions(_bundle(tmp_path, [], name="e"))


def test_bad_prob_shape_falls_back_and_missing_labels_are_padded(tmp_path):
    p = _bundle(tmp_path, [0, 1, 2], y_prob=[[0.5, 0.5]], y_pred=[0, 1, 2], labels=None)
    pred = d.load_predictions(p)
    assert pred.y_prob is None
    assert pred.classes == ["0", "1", "2"]


# ── threshold and cost ────────────────────────────────────────────────────

def test_confusion_and_dollar_score_by_hand(tmp_path):
    pred = d.load_predictions(_binary(tmp_path))
    y_pos = pred.y_true == 1
    s = d.positive_scores(pred, 1)
    assert d.confusion_at(y_pos, s, 0.5) == {"tp": 2, "fp": 1, "fn": 2, "tn": 3}
    c = d.confusion_at(y_pos, s, 0.5)
    # Costs are magnitudes: a negative cost entered by mistake is still a cost.
    assert d.dollar_score(c, 100, 10, 5) == 100 * 2 - 10 * 1 - 5 * 2
    assert d.dollar_score(c, 100, -10, -5) == d.dollar_score(c, 100, 10, 5)
    assert d.dollar_score(c, None, "abc", float("nan")) == 0.0


def test_threshold_extremes(tmp_path):
    pred = d.load_predictions(_binary(tmp_path))
    y_pos, s = pred.y_true == 1, d.positive_scores(pred, 1)
    assert d.confusion_at(y_pos, s, 0.0)["fn"] == 0          # flag everything
    assert d.confusion_at(y_pos, s, 1.0)["tp"] == 0          # flag nothing


def test_best_threshold_maximizes_score(tmp_path):
    pred = d.load_predictions(_binary(tmp_path))
    y_pos, s = pred.y_true == 1, d.positive_scores(pred, 1)
    t, score, counts = d.best_threshold(y_pos, s, 100, 10, 0)
    # Brute force over a fine grid must not beat it.
    grid = max(d.dollar_score(d.confusion_at(y_pos, s, g), 100, 10, 0) for g in np.linspace(0, 1, 1001))
    assert score == pytest.approx(grid)
    # Expensive false alarms push the threshold up.
    t_hi, _, _ = d.best_threshold(y_pos, s, 100, 1000, 0)
    assert t_hi >= t


def test_best_threshold_all_zero_costs_is_finite(tmp_path):
    pred = d.load_predictions(_binary(tmp_path))
    t, score, _ = d.best_threshold(pred.y_true == 1, d.positive_scores(pred, 1), 0, 0, 0)
    assert np.isfinite(t) and score == 0


def test_report_renders_and_clamps_threshold(tmp_path):
    md, fig = d.decision_report(_binary(tmp_path), "1", 7.5, 100, 10, 0)
    assert "Threshold:** 1.00" in md and fig is not None
    md, _ = d.decision_report(_binary(tmp_path, name="c"), "not-a-class", "oops", 1, 1)
    assert "Threshold:** 0.50" in md and "Positive class:** `1`" in md


def test_report_without_probabilities_warns(tmp_path):
    p = _bundle(tmp_path, [0, 1, 1, 0], y_pred=[0, 1, 0, 0])
    md, _ = d.decision_report(p, "1", 0.5, 1, 1)
    assert "no probabilities" in md and "1 caught" in md


def test_multiclass_and_single_class(tmp_path):
    p = _bundle(tmp_path, [0, 1, 2], [[.8, .1, .1], [.1, .8, .1], [.1, .1, .8]],
                labels=("low", "mid", "high"), name="m")
    md, _ = d.decision_report(p, "high", 0.5, 1, 1)
    assert "Multiclass" in md
    with pytest.raises(d.DecisionError, match="only one class"):
        d.decision_report(_bundle(tmp_path, [0, 0], [[1.0], [1.0]], labels=("only",), name="s"), "only", .5, 1, 1)


def test_nan_probabilities_do_not_crash(tmp_path):
    p = _bundle(tmp_path, [0, 1], [[float("nan"), float("nan")], [0.2, 0.8]])
    md, _ = d.decision_report(p, "1", 0.5, 1, 1)
    assert "1 caught" in md


@pytest.mark.parametrize("classes,expected", [
    (["no", "yes"], "yes"), (["legit", "fraud"], "fraud"), (["a", "b", "c"], "c"),
    ([" True ", "False"], " True "), ([], None),
])
def test_default_positive_class(classes, expected):
    assert d.default_positive_class(classes) == expected


# ── fairness ──────────────────────────────────────────────────────────────

def test_impact_ratio_matches_hand_computation(tmp_path):
    # Group A: 10 rows, 5 positive (50%). Group B: 10 rows, 2 positive... use 3 (30%) → 0.6
    y_true = [1] * 5 + [0] * 5 + [1] * 3 + [0] * 7
    probs = [[0, 1] if y else [1, 0] for y in y_true]     # perfect model
    groups = {"gender": ["A"] * 10 + ["B"] * 10}
    md, table = d.fairness_report(_bundle(tmp_path, y_true, probs, groups=groups), "gender", "1", 0.5)
    assert "**0.60** (fails" in md
    assert list(table["validation rows"]) == [10, 10]
    assert "Fewer than 30" in md


def test_missing_group_values_become_their_own_group(tmp_path):
    groups = {"region": ["x", None, float("nan"), "", "x", "nan"]}
    p = _bundle(tmp_path, [1, 0, 1, 0, 1, 0], [[0, 1]] * 6, groups=groups)
    _, table = d.fairness_report(p, "region", "1", 0.5)
    assert set(table["group"]) == {"x", d.MISSING_GROUP}


def test_impact_ratio_edge_cases():
    assert d.impact_ratio(pd.Series([0.5])) is None
    assert d.impact_ratio(pd.Series([0.0, 0.0])) is None
    assert d.impact_ratio(pd.Series([0.2, float("nan"), 0.4])) == pytest.approx(0.5)


def test_fairness_errors(tmp_path):
    p = _binary(tmp_path)
    with pytest.raises(d.DecisionError, match="tabular"):
        d.fairness_report(p, "gender", "1", 0.5)
    many = {"id": [str(i) for i in range(60)]}
    q = _bundle(tmp_path, [i % 2 for i in range(60)], [[.5, .5]] * 60, groups=many, name="many")
    with pytest.raises(d.DecisionError, match="Pick a group"):
        d.fairness_report(q, "nope", "1", 0.5)
    with pytest.raises(d.DecisionError, match="at most"):
        d.fairness_report(q, "id", "1", 0.5)


def test_group_values_with_wrong_length_are_ignored(tmp_path):
    p = _bundle(tmp_path, [0, 1], [[1, 0], [0, 1]], groups={"g": ["a"]})
    assert d.load_predictions(p).groups == {}


def test_group_columns_for_picks_low_cardinality_hashable_columns():
    df = pd.DataFrame({
        "label": [0, 1] * 30,
        "gender": ["m", "f", None] * 20,
        "id": range(60),
        "score": np.linspace(0, 1, 60),
        "tags": [["a"]] * 60,
        "when": pd.date_range("2024-01-01", periods=2).tolist() * 30,
        "allnan": [np.nan] * 60,
    })
    out = d.group_columns_for(df, "label")
    assert "label" not in out and "id" not in out and "score" not in out and "tags" not in out and "allnan" not in out
    assert out["gender"][:3] == ["m", "f", None]
    assert isinstance(out["when"][0], str)
    json.dumps(out)                                    # must serialize
    assert d.group_columns_for(None, "label") == {}
    assert d.group_columns_for(df.iloc[:0], "label") == {}
    assert len(d.group_columns_for(pd.DataFrame({f"c{i}": [0, 1] for i in range(40)}), "x")) == d.MAX_GROUP_COLUMNS


# ── UI callbacks ──────────────────────────────────────────────────────────

def test_callbacks_never_raise(tmp_path):
    from ui.callbacks import decision as cb

    for bp in (None, "", str(tmp_path / "missing")):
        out = cb.on_bundle_change(bp)
        assert len(out) == 7
        assert len(cb.update_panels(bp, None, 0.5, 0, 0, 0, None)) == 4
        cb.use_best_threshold(bp, None, 1, 1, 1)
    groups = {"gender": ["a", "b"] * 4}
    p = _binary(tmp_path, groups=groups, name="g")
    out = cb.on_bundle_change(p)
    assert out[0]["value"] == "1" and out[4]["value"] == "gender"
    assert out[6] is not None
    md, fig, fmd, fdf = cb.update_panels(p, "1", 0.3, 100, "", None, "gender")
    assert "Threshold:** 0.30" in md and fdf is not None
    upd = cb.use_best_threshold(p, "1", 100, 10, 0)
    assert 0 <= upd["value"] <= 1


def test_tabular_training_saves_group_values(tmp_path):
    """End to end: the real tabular loader + evaluate path writes group_values."""
    from modalities.tabular import load_tabular_data

    rng = np.random.default_rng(1)
    n = 80
    df = pd.DataFrame({"x": rng.normal(size=n), "gender": rng.choice(["f", "m"], n),
                       "label": rng.choice(["no", "yes"], n)})
    path = tmp_path / "t.csv"
    df.to_csv(path, index=False)
    meta: dict = {}
    _, val_loader, *_ = load_tabular_data(str(path), label_col="label", feature_cols=["x"],
                                          augmentation="none", meta_out=meta)
    val = meta["val_frame"]
    assert len(val) == len(val_loader.dataset)
    groups = d.group_columns_for(val, "label")
    assert "gender" in groups and len(groups["gender"]) == len(val)


def test_zero_costs_ask_for_values_instead_of_a_best_threshold(tmp_path):
    from ui.callbacks import decision as cb

    p = _binary(tmp_path)
    md, _ = d.decision_report(p, "1", 0.5, 0, 0, None)
    assert "enter the dollar values" in md and "Best threshold" not in md
    assert "2 caught" in md
    assert cb.use_best_threshold(p, "1", 0, "", None) == {"__type__": "update"}

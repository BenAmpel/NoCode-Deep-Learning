"""Pitfall coach: each check fires just past its threshold and stays quiet just before it."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from eval import pitfalls as P
from export import run_record
from pipeline.context import RunResult
from pipeline.runner import run
from test_pipeline import _config, _isolated_state  # noqa: F401  (autouse fixture)
from test_run_record import _REG_COLS


def ids(findings):
    return {f["id"] for f in findings or []}


def labels(counts: dict) -> list:
    """Labels with the given counts, interleaved (0, 1, 0, 1, ..., then the rest)."""
    left, out = dict(counts), []
    while any(left.values()):
        for k in counts:
            if left[k]:
                out.append(k)
                left[k] -= 1
    return out


def with_correct(y_true: list, n_correct: int) -> list:
    """Predictions that get the first n_correct right and the rest wrong (next class)."""
    k = max(y_true) + 1
    return [t if i < n_correct else (t + 1) % k for i, t in enumerate(y_true)]


def clf(y_true, y_pred, **kw):
    kw.setdefault("classes", [str(c) for c in sorted(set(y_true) | set(y_pred))])
    return P.find_pitfalls(task="classification", y_true=y_true, y_pred=y_pred, **kw)


# ── each threshold: fires / doesn't fire ─────────────────────────────────────

def test_small_validation_threshold():
    big = labels({0: 15, 1: 15})                      # 30 samples, 15 per class
    assert "small_validation" not in ids(clf(big, with_correct(big, 24)))
    small = labels({0: 15, 1: 14})                    # 29 samples
    found = clf(small, with_correct(small, 23))
    assert "small_validation" in ids(found)
    ev = next(f for f in found if f["id"] == "small_validation")["evidence"]
    assert "3.4 points" in ev and "95% interval" in ev


def test_rare_class_and_missing_class_count_as_small_validation():
    y = labels({0: 46, 1: 4})                         # n=50 but only 4 of class 1
    assert "small_validation" in ids(clf(y, with_correct(y, 45)))
    y = labels({0: 45, 1: 5})
    assert "small_validation" not in ids(clf(y, with_correct(y, 45)))
    y = labels({0: 25, 1: 25})                        # class '2' never in validation
    assert "small_validation" in ids(clf(y, with_correct(y, 40), classes=["a", "b", "c"]))


def test_tiny_validation_shows_only_the_size_warning():
    y = labels({0: 3, 1: 1})                          # the 4-row fixture case
    assert ids(clf(y, [0, 0, 0, 0])) == {"small_validation"}


def test_baseline_threshold():
    y = labels({0: 70, 1: 30})                        # majority share 70%
    assert "below_baseline" in ids(clf(y, with_correct(y, 71)))     # 71%: within 2 points
    assert "below_baseline" not in ids(clf(y, with_correct(y, 72)))  # 72%: 2 points above


def test_collapse_threshold_and_it_replaces_the_baseline_finding():
    y = labels({0: 60, 1: 40})
    pred = [0] * 95 + [1] * 5
    found = clf(y, pred)
    assert "collapsed_predictions" in ids(found) and "below_baseline" not in ids(found)
    assert "collapsed_predictions" not in ids(clf(y, [0] * 94 + [1] * 6))
    # A validation set that really is 96% one class is not a collapse.
    y = labels({0: 96, 1: 4})
    assert "collapsed_predictions" not in ids(clf(y, [0] * 100))


def test_imbalance_threshold_reports_balanced_accuracy():
    y = labels({0: 60, 1: 20})                        # 3 to 1
    found = clf(y, list(y), metrics={"balanced_accuracy": 0.81, "f1_macro": 0.77})
    imb = next(f for f in found if f["id"] == "class_imbalance")
    assert "3.0 to 1" in imb["evidence"] and "81.0%" in imb["evidence"] and "0.77" in imb["evidence"]
    y = labels({0: 59, 1: 20})
    assert "class_imbalance" not in ids(clf(y, list(y)))


def test_perfect_score_threshold():
    y = labels({0: 50, 1: 50})
    assert "suspiciously_perfect" in ids(clf(y, with_correct(y, 99)))
    assert "suspiciously_perfect" not in ids(clf(y, with_correct(y, 98)))


def test_few_training_samples_only_for_from_scratch_deep_models():
    y = labels({0: 50, 1: 50})
    pred = with_correct(y, 80)
    assert "few_training_samples" in ids(clf(y, pred, n_train=38, from_scratch_deep=True))
    assert "few_training_samples" not in ids(clf(y, pred, n_train=40, from_scratch_deep=True))
    assert "few_training_samples" not in ids(clf(y, pred, n_train=10, from_scratch_deep=False))


def test_regression_thresholds():
    def reg(r2, n=50):
        return ids(P.find_pitfalls(task="regression", metrics={"r2": r2}, y_true=[0.0] * n, y_pred=[0.0] * n))

    assert "below_baseline" in reg(0.0) and "below_baseline" not in reg(0.01)
    assert "suspiciously_perfect" in reg(0.99) and "suspiciously_perfect" not in reg(0.989)
    assert reg(0.5) == set()
    assert reg(0.5, n=29) == {"small_validation"}
    assert reg(float("nan")) == set()


def test_no_checks_for_tasks_the_coach_does_not_cover():
    assert P.find_pitfalls(task="clustering", y_true=[0, 1], y_pred=[0, 1]) is None
    assert P.find_pitfalls(task="anomaly") is None
    assert P.find_pitfalls(task="classification", model_name="Autoencoder", y_true=[0], y_pred=[0]) is None
    assert P.find_pitfalls(task="multi-label", y_true=[[0, 1]], y_pred=[[0, 1]]) is None
    assert P.find_pitfalls(task="classification", y_true=[[0, 1]], y_pred=[[0, 1]]) is None
    assert P.find_pitfalls(task="classification", y_true=[], y_pred=[]) is None
    assert P.find_pitfalls(task="regression", y_true=[], y_pred=[]) is None


def test_string_labels_do_not_trip_the_missing_class_check():
    y = ["cat"] * 25 + ["dog"] * 25
    assert "small_validation" not in ids(clf(y, list(y)[:40] + ["cat"] * 10, classes=["cat", "dog"]))


def test_findings_are_sorted_and_rendering_caps_them():
    y = labels({0: 70, 1: 10})
    found = clf(y, with_correct(y, 70), n_train=20, from_scratch_deep=True)
    sev = [P.SEVERITY_ORDER[f["severity"]] for f in found]
    assert sev == sorted(sev) and found[0]["severity"] == "critical"
    md = P.render_markdown(found, max_shown=1)
    assert md.startswith("### Pitfall check") and f"{len(found) - 1} more" in md
    assert P.render_markdown(None) == ""
    assert "Nothing flagged" in P.render_markdown([])


def test_wilson_interval():
    lo, hi = P.wilson_interval(8, 10)
    assert lo == pytest.approx(0.490, abs=1e-3) and hi == pytest.approx(0.943, abs=1e-3)
    assert P.wilson_interval(0, 10)[0] == 0.0 and P.wilson_interval(10, 10)[1] == 1.0


# ── end to end ───────────────────────────────────────────────────────────────

def _run(tmp_path, monkeypatch, **changes) -> RunResult:
    import pipeline.runner as runner

    monkeypatch.setattr(runner, "OUTPUTS_ROOT", tmp_path / "bundles")
    result = list(run(_config(**changes)))[-1]
    assert isinstance(result, RunResult)
    return result


@pytest.mark.parametrize("changes,checked", [
    (dict(model_name="MLP"), True),
    (dict(model_name="RandomForest"), True),
    (_REG_COLS, True),
    (dict(task="anomaly", model_name="Autoencoder"), False),
], ids=["mlp", "random-forest", "regression", "anomaly-autoencoder"])
def test_every_run_carries_its_pitfall_check(tmp_path, monkeypatch, changes, checked):
    result = _run(tmp_path, monkeypatch, **changes)
    record = json.loads(Path(result.bundle_path, run_record.JSON_NAME).read_text(encoding="utf-8"))
    md = Path(result.bundle_path, run_record.MARKDOWN_NAME).read_text(encoding="utf-8")
    assert record["pitfalls"] == result.pitfalls
    if checked:
        assert isinstance(result.pitfalls, list)
        assert "## Pitfall check" in md
        # The tiny fixture's validation set is the first thing a student should hear about.
        assert result.pitfalls and result.pitfalls[0]["id"] == "small_validation"
    else:
        assert result.pitfalls is None and "## Pitfall check" not in md


def test_a_failing_pitfall_check_never_fails_the_run(tmp_path, monkeypatch):
    def boom(*_a, **_k):
        raise RuntimeError("bad arrays")

    monkeypatch.setattr(P, "pitfalls_for_run", boom)
    result = _run(tmp_path, monkeypatch, model_name="RandomForest")
    assert result.pitfalls is None
    assert Path(result.bundle_path, run_record.JSON_NAME).is_file()


def test_run_summary_card_puts_the_pitfall_check_first():
    from ui.callbacks.train import _with_pitfalls

    finding = P._finding("x", "critical", "T", "E", "W", "Y")
    text = _with_pitfalls([finding], "loss summary")
    assert text.index("Pitfall check") < text.index("loss summary")
    assert _with_pitfalls(None, "loss summary") == "loss summary"
    assert _with_pitfalls(None, "") == ""

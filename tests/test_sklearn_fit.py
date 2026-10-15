import copy
import inspect
import sys
from pathlib import Path

import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.ensemble import (
    ExtraTreesRegressor,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression

from training import cancel
from training.sklearn_fit import fit_with_stop

IRIS = Path(__file__).resolve().parent.parent / "fixtures" / "iris_tutorial" / "iris.csv"


@pytest.fixture(autouse=True)
def _clear_stop_flag():
    cancel.reset()
    yield
    cancel.reset()


@pytest.fixture(scope="module")
def data():
    return make_classification(n_samples=12000, n_features=20, random_state=0)


def _same_model(a, b, X):
    pa = a.predict_proba(X) if hasattr(a, "predict_proba") else a.predict(X)
    pb = b.predict_proba(X) if hasattr(b, "predict_proba") else b.predict(X)
    return np.array_equal(pa, pb) and a.get_params() == b.get_params()


@pytest.mark.parametrize("model", [
    RandomForestClassifier(n_estimators=37, random_state=42),
    ExtraTreesRegressor(n_estimators=25, random_state=1),
    HistGradientBoostingClassifier(max_iter=60, early_stopping=False, random_state=42),
    # single fit early-stops at iteration 52; chunks are 13 iterations, so the
    # stop lands exactly on a chunk boundary
    HistGradientBoostingClassifier(max_iter=260, early_stopping=True, random_state=42),
    LogisticRegression(max_iter=200),
], ids=lambda m: type(m).__name__)
def test_chunked_fit_matches_a_single_fit(model, data):
    X, y = data
    single = copy.deepcopy(model).fit(X, y)
    chunked = fit_with_stop(copy.deepcopy(model), X, y)
    assert _same_model(single, chunked, X)
    if hasattr(single, "n_iter_"):
        assert chunked.n_iter_ == single.n_iter_


class _ForestThatStops(RandomForestClassifier):
    def fit(self, X, y, sample_weight=None):
        super().fit(X, y, sample_weight)
        cancel.request_stop()  # pressed during the first chunk
        return self


class _BoosterThatStops(HistGradientBoostingClassifier):
    def fit(self, X, y, sample_weight=None):
        super().fit(X, y, sample_weight)
        cancel.request_stop()
        return self


@pytest.mark.parametrize("model, done", [
    (_ForestThatStops(n_estimators=200, random_state=0), lambda m: len(m.estimators_)),
    (_BoosterThatStops(max_iter=200, early_stopping=False, random_state=0), lambda m: m.n_iter_),
], ids=["forest", "hist_gradient_boosting"])
def test_stop_ends_the_fit_after_the_current_chunk(model, done, data):
    X, y = data
    with pytest.raises(cancel.TrainingStopped):
        fit_with_stop(model, X, y)
    assert done(model) == 10  # one chunk of 200 / 20, not the full 200


def test_stop_pressed_before_the_fit_skips_it(data):
    cancel.request_stop()
    model = RandomForestClassifier(n_estimators=5)
    with pytest.raises(cancel.TrainingStopped):
        fit_with_stop(model, *data)
    assert not hasattr(model, "estimators_")


@pytest.mark.skipif(sys.platform == "darwin", reason="XGBoost is not used on macOS")
def test_stop_ends_an_xgboost_fit_between_rounds(data):
    xgboost = pytest.importorskip("xgboost")

    class _PressStop(xgboost.callback.TrainingCallback):
        def after_iteration(self, model, epoch, evals_log):
            if epoch == 2:
                cancel.request_stop()
            return False

    model = xgboost.XGBClassifier(n_estimators=200, callbacks=[_PressStop()])
    with pytest.raises(cancel.TrainingStopped):
        fit_with_stop(model, *data)
    assert model.get_booster().num_boosted_rounds() == 3
    assert len(model.get_params()["callbacks"]) == 1  # our callback removed again


# ---------------------------------------------------------------------------
# Evaluate-tab jobs
# ---------------------------------------------------------------------------

def _significance_args(**overrides):
    from ui.callbacks.evaluate import run_cv_significance

    values = dict(
        modality_val="tabular", data_path_val=str(IRIS), label_col_val="label", time_col_val="",
        window_size_val=10, subset_enabled_val=False, subset_percent_val=100, subset_seed_val=0,
        training_mode_val="standard", timeseries_forecast_mode_val=False,
        timeseries_forecast_horizon_val=1, timeseries_lag_steps_val=0, timeseries_rolling_window_val=0,
        base_model_name="LogisticRegression", task_val="classification", batch_size_val=16,
        dropout_val=0.1, optimizer_val="adam", hidden_size_val=16, num_layers_val=1,
        epochs_val=1, lr_val=0.01, n_estimators_val=20, max_depth_val=3, c_param_val=1.0,
        max_iter_val=200, lr_xgb_val=0.1, cv_k_val=10, metric_name="accuracy",
        compare_models=["RandomForest"],
    )
    values.update(overrides)
    params = inspect.signature(run_cv_significance).parameters
    assert list(params) == list(values)  # signature still visible through the wrapper
    return list(values.values())


def test_a_stale_stop_does_not_abort_the_next_significance_test():
    from ui.callbacks.evaluate import run_cv_significance

    cancel.request_stop()  # left over from an earlier, stopped job
    text, plot = run_cv_significance(*_significance_args())
    assert "RandomForest" in text and plot is not None


def test_a_stale_stop_does_not_abort_the_next_cross_validation():
    from ui.callbacks.evaluate import run_cv

    cancel.request_stop()  # used to make every later CV run fail with "Stopped by user."
    *_, last = run_cv(
        modality_val="tabular", data_path_val=str(IRIS), text_col_val="", label_col_val="label",
        time_col_val="", window_size_val=10, n_frames_val=8, sample_rate_val=16000, augmentation_val="none",
        val_split_val=0.2, subset_enabled_val=False, subset_percent_val=100, subset_seed_val=0,
        training_mode_val="standard", model_name_val="MLP", task_val="classification", epochs_val=1,
        lr_val=0.01, batch_size_val=16, dropout_val=0.1, optimizer_val="adam", scheduler_val="none",
        use_amp_val=False, timeseries_forecast_mode_val=False, timeseries_forecast_horizon_val=1,
        timeseries_lag_steps_val=0, timeseries_rolling_window_val=0, hidden_size_val=16, num_layers_val=1,
        cv_k_val=3,
    )
    assert "Primary fold summary" in last


def test_stop_during_a_significance_test_reports_stopped(monkeypatch):
    from ui.callbacks import evaluate

    def _stop(*_):
        raise cancel.TrainingStopped("Stopped by user.")

    monkeypatch.setattr(evaluate, "fit_with_stop", _stop)
    assert evaluate.run_cv_significance(*_significance_args()) == ("### ⏹ Significance test stopped.", None)

import builtins
import sys

import numpy as np
import pytest

import models.registry as registry
from models.registry import get_models, is_sklearn
from models.tabular_models import get_tabular_model


@pytest.fixture
def reset_xgboost_status(monkeypatch):
    monkeypatch.setattr(registry, "_xgboost_status", None)
    yield
    registry._xgboost_status = None


@pytest.fixture
def forbid_xgboost_import(monkeypatch):
    """Fail the test if anything tries to import xgboost.

    On macOS, merely loading xgboost after torch can pull in a second OpenMP
    runtime and segfault later, so the darwin path must never import it.
    """
    real_import = builtins.__import__

    def guarded(name, *args, **kwargs):
        if name == "xgboost" or name.startswith("xgboost."):
            raise AssertionError("xgboost must not be imported on macOS")
        return real_import(name, *args, **kwargs)

    monkeypatch.delitem(sys.modules, "xgboost", raising=False)
    monkeypatch.setattr(builtins, "__import__", guarded)


@pytest.mark.parametrize("modality", ["tabular", "timeseries"])
def test_macos_hides_xgboost_and_offers_hist_gradient_boosting(
    monkeypatch, reset_xgboost_status, forbid_xgboost_import, modality
):
    monkeypatch.setattr(sys, "platform", "darwin")
    models = get_models(modality, "from_scratch")
    assert "XGBoost" not in models
    assert "HistGradientBoosting" in models
    assert registry.xgboost_available() is False


def test_xgboost_hidden_when_import_fails(monkeypatch, reset_xgboost_status):
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setitem(sys.modules, "xgboost", None)  # makes `import xgboost` raise ImportError
    assert "XGBoost" not in get_models("tabular", "from_scratch")
    available, reason = registry.xgboost_status()
    assert available is False
    assert "could not be loaded" in reason


def test_choosing_xgboost_when_unavailable_raises_a_clear_error(
    monkeypatch, reset_xgboost_status, forbid_xgboost_import
):
    monkeypatch.setattr(sys, "platform", "darwin")
    with pytest.raises(ValueError, match="HistGradientBoosting"):
        get_tabular_model("XGBoost", num_classes=3, input_size=4)


@pytest.mark.parametrize("task", ["classification", "regression"])
def test_hist_gradient_boosting_trains(task):
    rng = np.random.default_rng(0)
    X = rng.random((200, 4))
    y = (X[:, 0] + X[:, 1] > 1).astype(int) if task == "classification" else X[:, 0] * 3.0
    model = get_tabular_model(
        "HistGradientBoosting", num_classes=2, input_size=4, task=task,
        n_estimators=15, max_depth=3, learning_rate=0.2,
    )
    assert is_sklearn("HistGradientBoosting")
    assert model.max_iter == 15
    assert model.max_depth == 3
    assert model.learning_rate == pytest.approx(0.2)
    model.fit(X, y)
    assert model.predict(X).shape == (200,)
    if task == "classification":
        assert model.predict_proba(X).shape == (200, 2)


def test_hist_gradient_boosting_unlimited_depth():
    model = get_tabular_model("HistGradientBoosting", num_classes=2, input_size=4, max_depth=0)
    assert model.max_depth is None


def test_guided_mode_recommends_an_offered_boosted_model(monkeypatch, reset_xgboost_status):
    from ui.guided_mode import recommend

    monkeypatch.setattr(sys, "platform", "darwin")
    rec = recommend("tabular", {"n_samples": 10_000, "n_classes": 2}, "cpu")
    assert rec["model_name"] == "HistGradientBoosting"
    assert rec["model_name"] in get_models("tabular", rec["training_mode"])


def test_shap_explains_hist_gradient_boosting():
    from eval.shap_explain import compute_shap

    rng = np.random.default_rng(0)
    X = rng.random((80, 3))
    y = (X[:, 0] > 0.5).astype(int)
    model = get_tabular_model("HistGradientBoosting", num_classes=2, input_size=3, n_estimators=10).fit(X, y)
    fig = compute_shap(model, {"X_val": X}, "tabular", {"numeric_columns": ["a", "b", "c"]})
    assert fig is not None
    assert "SHAP" in fig.axes[0].get_title()

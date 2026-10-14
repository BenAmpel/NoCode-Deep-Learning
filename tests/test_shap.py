import numpy as np
import pytest
import torch
from sklearn.ensemble import RandomForestClassifier

from config import DEVICE
from eval.shap_explain import _abs_per_feature, compute_shap
from models.tabular_models import get_tabular_model

pytest.importorskip("shap")


def _data(n=60, f=4, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, f)).astype(np.float32)
    y = (X[:, 0] + 0.5 * X[:, 1] > 0).astype(np.int64)
    return X, y


PREP = {"numeric_columns": ["a", "b", "c", "d"], "ohe_categories": {}}


@pytest.mark.parametrize(
    "shape_in",
    [
        "list_per_class",    # older shap: [ (n, f) per class ]
        "array_n_f_c",       # newer shap: (n, f, c)
        "array_n_f",         # single output
    ],
)
def test_abs_per_feature_normalizes_shapes(shape_in):
    rng = np.random.default_rng(0)
    if shape_in == "list_per_class":
        sv = [rng.normal(size=(5, 4)) for _ in range(3)]
    elif shape_in == "array_n_f_c":
        sv = rng.normal(size=(5, 4, 3))
    else:
        sv = rng.normal(size=(5, 4))
    out = _abs_per_feature(sv)
    assert out.shape == (5, 4)
    assert (out >= 0).all()


def test_mlp_shap_runs_on_training_device():
    # The MLP branch has no feature_importances_ fallback, so a None result
    # means the SHAP path itself failed (e.g. a CPU tensor sent to MPS).
    X, _ = _data()
    model = get_tabular_model("MLP", num_classes=2, input_size=4, dropout=0.0).to(DEVICE)
    fig = compute_shap(model, {"X_val": X}, "tabular", PREP)
    assert fig is not None


def test_mlp_regression_shap_is_not_all_zero():
    X, _ = _data()
    torch.manual_seed(0)
    model = get_tabular_model("MLP", num_classes=1, input_size=4, dropout=0.0).to(DEVICE)
    fig = compute_shap(model, {"X_val": X}, "tabular", PREP)
    assert fig is not None
    widths = [bar.get_width() for bar in fig.axes[0].patches]
    assert max(widths) > 0  # softmax over one output made every value 0


def test_random_forest_shap_uses_shap_not_fallback():
    X, y = _data()
    model = RandomForestClassifier(n_estimators=10, random_state=0).fit(X, y)
    fig = compute_shap(model, {"X_val": X}, "tabular", PREP)
    assert fig is not None
    assert "SHAP" in fig.axes[0].get_title()  # fallback chart has a different title

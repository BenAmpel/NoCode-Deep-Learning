from pathlib import Path

import numpy as np
import pytest
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

from export.exporter import export_bundle

ort = pytest.importorskip("onnxruntime")
pytest.importorskip("skl2onnx")


def _data(n_features=4):
    rng = np.random.default_rng(0)
    X = rng.random((300, n_features)).astype(np.float32)
    X[rng.random(X.shape) < 0.1] = np.nan  # exercise missing-value routing
    return rng, X


def _run_onnx(bundle: Path, X):
    session = ort.InferenceSession(str(bundle / "model.onnx"), providers=["CPUExecutionProvider"])
    return session.run(None, {"float_input": X})


@pytest.mark.parametrize("n_classes", [2, 3])
def test_hist_gradient_boosting_classifier_exports_onnx(tmp_path, n_classes):
    rng, X = _data()
    y = rng.integers(0, n_classes, len(X))
    model = HistGradientBoostingClassifier(max_iter=20, random_state=0).fit(X, y)

    bundle, warnings = export_bundle(
        model,
        {"modality": "tabular", "sklearn_input_size": X.shape[1]},
        [str(c) for c in range(n_classes)],
        "hgb_cls",
        output_dir=str(tmp_path),
        is_sklearn=True,
    )

    bundle = Path(bundle)
    assert warnings == []
    assert (bundle / "model.onnx").exists()
    labels, probabilities = _run_onnx(bundle, X)
    np.testing.assert_array_equal(labels, model.predict(X))
    onnx_proba = np.array([[row[c] for c in model.classes_] for row in probabilities])
    np.testing.assert_allclose(onnx_proba, model.predict_proba(X), atol=1e-5)


def test_hist_gradient_boosting_regressor_exports_onnx(tmp_path):
    rng, X = _data()
    y = np.nan_to_num(X[:, 0]) * 3 + rng.normal(0, 0.1, len(X))
    model = HistGradientBoostingRegressor(max_iter=20, random_state=0).fit(X, y)

    bundle, warnings = export_bundle(
        model,
        {"modality": "tabular", "sklearn_input_size": X.shape[1]},
        [],
        "hgb_reg",
        output_dir=str(tmp_path),
        is_sklearn=True,
    )

    assert warnings == []
    (predictions,) = _run_onnx(Path(bundle), X)
    np.testing.assert_allclose(predictions.ravel(), model.predict(X), atol=1e-4)

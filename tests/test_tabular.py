import numpy as np
import pandas as pd
import pytest

from modalities.tabular import load_tabular_data


@pytest.fixture
def regression_csv(tmp_path):
    rng = np.random.default_rng(0)
    n = 40
    df = pd.DataFrame({
        "size": rng.uniform(50, 200, n),
        "rooms": rng.integers(1, 6, n),
        # Values chosen so string order ("100000" < "99000") differs from numeric order.
        "price": rng.choice([99000.0, 100000.0, 150000.0, 250000.0], n),
    })
    path = tmp_path / "houses.csv"
    df.to_csv(path, index=False)
    return path, df


def test_regression_keeps_real_target_values(regression_csv):
    path, df = regression_csv
    _, _, classes, prep, _, _, y_train, _, y_val = load_tabular_data(
        str(path), label_col="price", augmentation="none", task="regression"
    )
    assert classes == ["target"]
    assert prep["task"] == "regression"
    y = np.concatenate([y_train, y_val])
    assert y.dtype == np.float32
    assert sorted(y.tolist()) == sorted(df["price"].astype(np.float32).tolist())


def test_regression_rejects_non_numeric_target(tmp_path):
    path = tmp_path / "bad.csv"
    pd.DataFrame({"x": [1, 2, 3, 4], "y": ["1.0", "two", "3.0", "4.0"]}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="numeric target"):
        load_tabular_data(str(path), label_col="y", task="regression")


def test_classification_still_maps_to_class_indices(regression_csv):
    path, _ = regression_csv
    _, _, classes, _, _, _, y_train, _, y_val = load_tabular_data(
        str(path), label_col="rooms", augmentation="none"
    )
    y = np.concatenate([y_train, y_val])
    assert y.dtype == np.int64
    assert set(y.tolist()) <= set(range(len(classes)))


def test_scaling_is_fit_on_training_rows_only(tmp_path):
    # Validation rows get extreme values; if they leaked into the scaler the
    # stored mean would move far away from the training rows' mean.
    n = 50
    df = pd.DataFrame({"x": np.arange(n, dtype=float), "label": ["a", "b"] * (n // 2)})
    path = tmp_path / "leak.csv"
    df.to_csv(path, index=False)
    _, _, _, prep, _, X_train, _, X_val, _ = load_tabular_data(
        str(path), label_col="label", augmentation="none", val_split=0.2
    )
    # Recover which raw values went to training via the stored scaler.
    mean, std = prep["scaler_mean"][0], prep["scaler_scale"][0]
    train_raw = X_train[:, 0] * std + mean
    assert np.isclose(mean, train_raw.mean(), atol=1e-3)
    # Standardized training column has mean ~0; validation column need not.
    assert abs(X_train[:, 0].mean()) < 1e-5

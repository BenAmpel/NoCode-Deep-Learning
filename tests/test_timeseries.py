import numpy as np
import pandas as pd
import pytest

from modalities.timeseries import load_timeseries_data, prepare_timeseries_windows


@pytest.fixture
def row_index_csv(tmp_path):
    # The only feature is the row number, so after (monotonic) scaling a
    # window's values still tell us which rows it covers.
    n = 200
    df = pd.DataFrame({"idx": np.arange(n, dtype=float), "label": (np.arange(n) // 20) % 2})
    path = tmp_path / "series.csv"
    df.to_csv(path, index=False)
    return path


def test_train_and_val_windows_do_not_overlap(row_index_csv):
    window = 10
    *_, prep, _, X_train, _, X_val, _ = load_timeseries_data(
        str(row_index_csv), label_col="label", feature_cols=["idx"],
        window_size=window, val_split=0.2, augmentation="none",
    )
    assert X_train.max() < X_val.min()


def test_normalizer_fit_on_training_rows_only(row_index_csv):
    *_, prep, _, _, _, _, _ = load_timeseries_data(
        str(row_index_csv), label_col="label", feature_cols=["idx"],
        window_size=10, val_split=0.2, augmentation="none",
    )
    train_rows = prep["train_rows"]
    assert train_rows == 160
    assert prep["mean"][0] == pytest.approx((train_rows - 1) / 2)  # mean of rows 0..159


def test_subset_keeps_windows_in_time_order(row_index_csv):
    *_, starts = prepare_timeseries_windows(
        str(row_index_csv), label_col="label", feature_cols=["idx"],
        window_size=10, subset_percent=50, subset_seed=3, return_window_starts=True,
    )
    assert len(starts) > 10
    assert np.all(np.diff(starts) > 0)


def test_default_call_shape_is_unchanged_for_cv_callers(row_index_csv):
    out = prepare_timeseries_windows(str(row_index_csv), label_col="label", feature_cols=["idx"], window_size=10)
    assert len(out) == 5
    assert out[3]["train_rows"] == 200  # default fit_fraction=1.0 keeps old behavior


def test_too_short_series_gives_clear_error(tmp_path):
    path = tmp_path / "short.csv"
    pd.DataFrame({"idx": np.arange(12.0), "label": [0, 1] * 6}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="no training window fits"):
        load_timeseries_data(str(path), label_col="label", feature_cols=["idx"],
                             window_size=10, val_split=0.2, augmentation="none")

"""
Time-series modality pipeline.

Expected data format:
    CSV file with:
        - optional timestamp column (sorted ascending, then dropped)
        - feature columns (numeric)
        - label column (one label per row; majority vote used per window)
"""
import random

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from config import DEFAULTS
from data_pipeline.io_utils import drop_missing_label_rows, read_structured_file


def build_temporal_feature_matrix(
    values: np.ndarray,
    *,
    lag_steps: int = 0,
    rolling_window: int = 0,
) -> tuple[np.ndarray, int]:
    """Expand base time-series features with lag and rolling statistics.

    Returns the engineered feature matrix and the number of leading rows that
    must be discarded to avoid NaNs caused by lagging/rolling windows.
    """
    engineered = [values]
    trim = 0

    if int(lag_steps) > 0:
        trim = max(trim, int(lag_steps))
        for lag in range(1, int(lag_steps) + 1):
            lagged = np.roll(values, shift=lag, axis=0)
            lagged[:lag, :] = np.nan
            engineered.append(lagged)

    if int(rolling_window) > 1:
        window = int(rolling_window)
        trim = max(trim, window - 1)
        means = np.full_like(values, np.nan, dtype=np.float32)
        stds = np.full_like(values, np.nan, dtype=np.float32)
        mins = np.full_like(values, np.nan, dtype=np.float32)
        maxs = np.full_like(values, np.nan, dtype=np.float32)
        for idx in range(window - 1, len(values)):
            chunk = values[idx - window + 1: idx + 1]
            means[idx] = chunk.mean(axis=0)
            stds[idx] = chunk.std(axis=0)
            mins[idx] = chunk.min(axis=0)
            maxs[idx] = chunk.max(axis=0)
        engineered.extend([means, stds, mins, maxs])

    matrix = np.concatenate(engineered, axis=1).astype(np.float32)
    return matrix, trim


def _engineered_feature_names(
    feature_cols: list[str],
    *,
    lag_steps: int = 0,
    rolling_window: int = 0,
) -> list[str]:
    names = list(feature_cols)
    if int(lag_steps) > 0:
        for lag in range(1, int(lag_steps) + 1):
            names.extend([f"{col}_lag_{lag}" for col in feature_cols])
    if int(rolling_window) > 1:
        for stat in ("roll_mean", "roll_std", "roll_min", "roll_max"):
            names.extend([f"{col}_{stat}_{int(rolling_window)}" for col in feature_cols])
    return names


def prepare_timeseries_windows(
    data_path: str,
    label_col: str,
    feature_cols: list[str] | None = None,
    time_col: str = None,
    window_size: int = 50,
    stride: int = 1,
    task: str = "classification",
    subset_percent: float = 100.0,
    subset_seed: int = 42,
    forecast_mode: bool = False,
    forecast_horizon: int = 1,
    lag_steps: int = 0,
    rolling_window: int = 0,
) -> tuple[np.ndarray, np.ndarray, list[str], dict, int]:
    import pandas as pd

    df = read_structured_file(data_path)
    df, dropped_missing = drop_missing_label_rows(df, label_col)
    if df.empty:
        raise ValueError(f"No rows remain after dropping missing labels from '{label_col}'.")

    if time_col and time_col in df.columns:
        df = df.sort_values(time_col).drop(columns=[time_col])

    selected_feature_cols = [c for c in (feature_cols or []) if c in df.columns and c not in {label_col, time_col}]
    if selected_feature_cols:
        feature_cols = selected_feature_cols
    else:
        feature_cols = [c for c in df.columns if c not in {label_col, time_col}]
    if not feature_cols:
        raise ValueError("No valid feature columns were selected for time-series training.")
    base_features = df[feature_cols].fillna(0).values.astype(np.float32)
    engineered_feature_names = _engineered_feature_names(
        feature_cols,
        lag_steps=int(lag_steps),
        rolling_window=int(rolling_window),
    )
    X, trim_rows = build_temporal_feature_matrix(
        base_features,
        lag_steps=int(lag_steps),
        rolling_window=int(rolling_window),
    )

    # Standardize per feature
    if task == "regression" or bool(forecast_mode):
        target_series = pd.to_numeric(df[label_col], errors="coerce")
        if bool(forecast_mode):
            if task != "regression":
                raise ValueError("Forecast mode currently supports regression targets only. Switch the task to regression for forecasting.")
            target_series = target_series.shift(-int(forecast_horizon))
        target_mask = target_series.notna().values
        targets = target_series.fillna(0).values.astype(np.float32)
        classes = ["target"]
    else:
        labels_raw = df[label_col].tolist()
        classes = sorted(set(str(l) for l in labels_raw))
        class_to_idx = {c: i for i, c in enumerate(classes)}
        targets = np.array([class_to_idx[str(l)] for l in labels_raw], dtype=np.int64)
        target_mask = np.ones(len(targets), dtype=bool)

    valid_mask = np.isfinite(X).all(axis=1) & target_mask
    if int(trim_rows) > 0:
        valid_mask[:int(trim_rows)] = False

    X = X[valid_mask]
    targets = targets[valid_mask]
    if X.size == 0:
        raise ValueError("No valid rows remain after applying time-series lag/rolling features. Reduce the lag steps or rolling window.")

    mean = X.mean(axis=0)
    std = X.std(axis=0) + 1e-8
    X = (X - mean) / std

    windows, window_labels = [], []
    for i in range(0, len(X) - window_size + 1, stride):
        windows.append(X[i: i + window_size])
        segment = targets[i: i + window_size]
        if task == "regression" or bool(forecast_mode):
            window_labels.append(float(segment[-1]))
        else:
            window_labels.append(int(np.bincount(segment).argmax()))

    if not windows:
        raise ValueError(
            f"Not enough rows ({len(X)}) for window_size={window_size}. "
            "Reduce window_size or use a larger dataset."
        )

    windows = np.array(windows, dtype=np.float32)
    if task == "regression":
        window_labels = np.array(window_labels, dtype=np.float32)
    else:
        window_labels = np.array(window_labels, dtype=np.int64)

    sampled_windows = len(windows)
    if float(subset_percent) < 100.0:
        rng = np.random.default_rng(int(subset_seed))
        subset_size = max(1, int(round(len(windows) * (float(subset_percent) / 100.0))))
        subset_size = min(subset_size, len(windows))
        selected_idx = rng.choice(len(windows), size=subset_size, replace=False)
        windows = windows[selected_idx]
        window_labels = window_labels[selected_idx]
        sampled_windows = subset_size

    preprocessing_config = {
        "modality": "timeseries",
        "task": "forecasting" if bool(forecast_mode) else task,
        "raw_feature_columns": feature_cols,
        "feature_columns": engineered_feature_names,
        "window_size": window_size,
        "stride": stride,
        "mean": mean.tolist(),
        "std": std.tolist(),
        "input_size": X.shape[1],
        "sklearn_input_size": int(window_size * X.shape[1]),
        "dropped_missing_labels": dropped_missing,
        "subset_rows": sampled_windows,
        "subset_percent": float(subset_percent),
        "forecast_mode": bool(forecast_mode),
        "forecast_horizon": int(forecast_horizon),
        "lag_steps": int(lag_steps),
        "rolling_window": int(rolling_window),
    }

    return windows, window_labels, classes, preprocessing_config, X.shape[1]


def load_timeseries_data(
    data_path: str,
    label_col: str,
    feature_cols: list[str] | None = None,
    time_col: str = None,
    window_size: int = 50,
    stride: int = 1,
    batch_size: int = 16,
    val_split: float = 0.2,
    augmentation: str = "light",
    task: str = "classification",
    subset_percent: float = 100.0,
    subset_seed: int = 42,
    forecast_mode: bool = False,
    forecast_horizon: int = 1,
    lag_steps: int = 0,
    rolling_window: int = 0,
) -> tuple[DataLoader, DataLoader, list[str], dict, int, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    windows, window_labels, classes, preprocessing_config, input_size = prepare_timeseries_windows(
        data_path=data_path,
        label_col=label_col,
        feature_cols=feature_cols,
        time_col=time_col,
        window_size=window_size,
        stride=stride,
        task=task,
        subset_percent=subset_percent,
        subset_seed=subset_seed,
        forecast_mode=forecast_mode,
        forecast_horizon=forecast_horizon,
        lag_steps=lag_steps,
        rolling_window=rolling_window,
    )

    n_val = max(1, int(len(windows) * val_split))
    split_at = max(1, len(windows) - n_val)
    train_idx = list(range(0, split_at))
    val_idx = list(range(split_at, len(windows)))

    # Apply timeseries augmentation to training windows only
    try:
        from data_pipeline.augmentation import get_timeseries_augmentation
        aug_fn = get_timeseries_augmentation(augmentation)
        windows_train = np.array([aug_fn(w) for w in windows[train_idx]])
    except Exception as e:
        import logging
        logging.getLogger(__name__).warning("Timeseries augmentation skipped: %s", e)
        windows_train = windows[train_idx]

    # Windows are pre-built in RAM — num_workers=0 is faster (no MP overhead).
    pm = DEFAULTS["pin_memory"]
    label_dtype = torch.float32 if task == "regression" else torch.int64
    train_loader = DataLoader(
        TensorDataset(torch.tensor(windows_train), torch.tensor(window_labels[train_idx], dtype=label_dtype)),
        batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=pm,
    )
    val_loader = DataLoader(
        TensorDataset(torch.tensor(windows[val_idx]), torch.tensor(window_labels[val_idx], dtype=label_dtype)),
        batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=pm,
    )

    X_train_flat = windows[train_idx].reshape(len(train_idx), -1).astype(np.float32)
    X_val_flat = windows[val_idx].reshape(len(val_idx), -1).astype(np.float32)
    y_train = window_labels[train_idx]
    y_val = window_labels[val_idx]

    return (
        train_loader,
        val_loader,
        classes,
        preprocessing_config,
        input_size,
        X_train_flat,
        y_train,
        X_val_flat,
        y_val,
    )

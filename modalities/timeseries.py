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
    X = df[feature_cols].fillna(0).values.astype(np.float32)

    # Standardize per feature
    mean = X.mean(axis=0)
    std = X.std(axis=0) + 1e-8
    X = (X - mean) / std

    if task == "regression":
        targets = pd.to_numeric(df[label_col], errors="coerce").fillna(0).values.astype(np.float32)
        classes = ["target"]
    else:
        labels_raw = df[label_col].tolist()
        classes = sorted(set(str(l) for l in labels_raw))
        class_to_idx = {c: i for i, c in enumerate(classes)}
        targets = np.array([class_to_idx[str(l)] for l in labels_raw], dtype=np.int64)

    windows, window_labels = [], []
    for i in range(0, len(X) - window_size + 1, stride):
        windows.append(X[i: i + window_size])
        segment = targets[i: i + window_size]
        if task == "regression":
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
        "task": task,
        "feature_columns": feature_cols,
        "window_size": window_size,
        "stride": stride,
        "mean": mean.tolist(),
        "std": std.tolist(),
        "input_size": X.shape[1],
        "sklearn_input_size": int(window_size * X.shape[1]),
        "dropped_missing_labels": dropped_missing,
        "subset_rows": sampled_windows,
        "subset_percent": float(subset_percent),
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
    )

    indices = list(range(len(windows)))
    random.seed(42)
    random.shuffle(indices)
    n_val      = max(1, int(len(windows) * val_split))
    val_idx    = indices[:n_val]
    train_idx  = indices[n_val:]

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

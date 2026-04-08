"""
Tabular modality pipeline.

Expected data format:
    CSV file where one column is the label and the rest are features.
    Numeric columns are standardized; categorical columns are one-hot encoded.
"""
import logging
import random

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from config import DEFAULTS
from data_pipeline.io_utils import drop_missing_label_rows, read_structured_file


def load_tabular_data(
    data_path: str,
    label_col: str,
    feature_cols: list[str] | None = None,
    batch_size: int = 16,
    val_split: float = 0.2,
    augmentation: str = "light",
    scaling_strategy: str = "standard",
    subset_percent: float = 100.0,
    subset_seed: int = 42,
) -> tuple[DataLoader, DataLoader, list[str], dict, int, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    df = read_structured_file(data_path)
    df, dropped_missing = drop_missing_label_rows(df, label_col)
    if df.empty:
        raise ValueError(f"No rows remain after dropping missing labels from '{label_col}'.")
    from data_pipeline.io_utils import apply_random_subset
    df, sampled_rows = apply_random_subset(
        df,
        enabled=float(subset_percent) < 100.0,
        subset_percent=subset_percent,
        subset_seed=subset_seed,
    )

    labels_raw   = df[label_col].tolist()
    classes      = sorted(set(str(l) for l in labels_raw))
    class_to_idx = {c: i for i, c in enumerate(classes)}
    labels       = np.array([class_to_idx[str(l)] for l in labels_raw], dtype=np.int64)

    if feature_cols:
        usable_feature_cols = [col for col in feature_cols if col in df.columns and col != label_col]
        if not usable_feature_cols:
            raise ValueError("No valid feature columns were selected for tabular training.")
        features_df = df[usable_feature_cols].copy()
    else:
        features_df = df.drop(columns=[label_col])
    cat_cols    = features_df.select_dtypes(include=["object", "category"]).columns.tolist()
    num_cols    = features_df.select_dtypes(include=[np.number]).columns.tolist()

    # Scale numeric features
    num_array = features_df[num_cols].fillna(0).values.astype(np.float32)
    mean = num_array.mean(axis=0)
    std = num_array.std(axis=0) + 1e-8
    min_vals = num_array.min(axis=0) if len(num_cols) else np.array([], dtype=np.float32)
    max_vals = num_array.max(axis=0) if len(num_cols) else np.array([], dtype=np.float32)
    median_vals = np.median(num_array, axis=0) if len(num_cols) else np.array([], dtype=np.float32)
    q1 = np.quantile(num_array, 0.25, axis=0) if len(num_cols) else np.array([], dtype=np.float32)
    q3 = np.quantile(num_array, 0.75, axis=0) if len(num_cols) else np.array([], dtype=np.float32)
    iqr = (q3 - q1) + 1e-8 if len(num_cols) else np.array([], dtype=np.float32)

    if scaling_strategy == "minmax" and len(num_cols):
        num_array = (num_array - min_vals) / np.where((max_vals - min_vals) == 0, 1.0, (max_vals - min_vals))
    elif scaling_strategy == "robust" and len(num_cols):
        num_array = (num_array - median_vals) / np.where(iqr == 0, 1.0, iqr)
    elif scaling_strategy == "none":
        num_array = num_array
    else:
        num_array = (num_array - mean) / std

    # One-hot encode categorical features
    ohe_categories: dict[str, list] = {}
    ohe_feature_names: list[str] = []
    parts = [num_array]
    for col in cat_cols:
        cats              = sorted(features_df[col].dropna().unique().tolist())
        ohe_categories[col] = cats
        dummies           = pd.get_dummies(features_df[col], prefix=col)
        ohe_feature_names.extend(dummies.columns.tolist())
        parts.append(dummies.fillna(0).values.astype(np.float32))

    X = np.concatenate(parts, axis=1)

    # Train/val split
    indices = list(range(len(X)))
    random.seed(42)
    random.shuffle(indices)
    n_val      = max(1, int(len(X) * val_split))
    val_idx    = indices[:n_val]
    train_idx  = indices[n_val:]

    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = labels[train_idx], labels[val_idx]

    # Apply tabular augmentation to training data only
    try:
        from data_pipeline.augmentation import get_tabular_augmentation
        aug_fn = get_tabular_augmentation(augmentation)
        X_train = aug_fn(X_train)
    except Exception as e:
        import logging
        logging.getLogger(__name__).warning("Tabular augmentation skipped: %s", e)

    # Data is already in RAM as tensors — num_workers=0 is faster (no MP overhead).
    pm = DEFAULTS["pin_memory"]
    train_loader = DataLoader(
        TensorDataset(torch.tensor(X_train), torch.tensor(y_train)),
        batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=pm,
    )
    val_loader = DataLoader(
        TensorDataset(torch.tensor(X_val), torch.tensor(y_val)),
        batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=pm,
    )

    preprocessing_config = {
        "modality": "tabular",
        "numeric_columns": num_cols,
        "categorical_columns": cat_cols,
        "feature_columns": features_df.columns.tolist(),
        "selected_feature_columns": features_df.columns.tolist(),
        "ohe_categories": ohe_categories,
        "scaler": {"mean": mean.tolist(), "std": std.tolist(), "columns": num_cols},
        "feature_order": num_cols + ohe_feature_names,
        "numeric_cols": num_cols,
        "categorical_cols": list(ohe_categories.keys()),
        "dummy_feature_names": ohe_feature_names,
        "category_maps": {col: {str(cat): f"{col}_{cat}" for cat in cats} for col, cats in ohe_categories.items()},
        "scaler_mean": mean.tolist(),
        "scaler_scale": std.tolist(),
        "scaling_strategy": scaling_strategy,
        "min_vals": min_vals.tolist() if len(num_cols) else [],
        "max_vals": max_vals.tolist() if len(num_cols) else [],
        "median_vals": median_vals.tolist() if len(num_cols) else [],
        "iqr": iqr.tolist() if len(num_cols) else [],
        "input_size": X.shape[1],
        "dropped_missing_labels": dropped_missing,
        "subset_rows": sampled_rows,
        "subset_percent": float(subset_percent),
    }

    return train_loader, val_loader, classes, preprocessing_config, X.shape[1], X_train, y_train, X_val, y_val

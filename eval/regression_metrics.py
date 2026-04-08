"""
eval/regression_metrics.py

Regression evaluation metrics and diagnostic plots for the NoCode Deep
Learning platform.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure


# ---------------------------------------------------------------------------
# Type alias
# ---------------------------------------------------------------------------
ArrayLike = Union[np.ndarray, List[float]]


# ---------------------------------------------------------------------------
# Core metric computation
# ---------------------------------------------------------------------------

def compute_regression_metrics(
    y_true: ArrayLike,
    y_pred: ArrayLike,
) -> Dict[str, float]:
    """Compute standard regression evaluation metrics.

    Parameters
    ----------
    y_true:
        Ground-truth target values.
    y_pred:
        Model predictions.

    Returns
    -------
    dict with keys:
        ``mae``   – Mean Absolute Error
        ``mse``   – Mean Squared Error
        ``rmse``  – Root Mean Squared Error
        ``r2``    – R² (coefficient of determination)
        ``mape``  – Mean Absolute Percentage Error (guarded against div-by-zero)
    """
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()

    if y_true.shape != y_pred.shape:
        raise ValueError(
            f"Shape mismatch: y_true {y_true.shape} vs y_pred {y_pred.shape}"
        )

    residuals = y_true - y_pred

    mae = float(np.mean(np.abs(residuals)))
    mse = float(np.mean(residuals ** 2))
    rmse = float(np.sqrt(mse))

    ss_res = float(np.sum(residuals ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0.0 else 0.0

    # MAPE: skip samples where y_true == 0 to avoid division-by-zero
    nonzero_mask = y_true != 0.0
    if nonzero_mask.any():
        mape = float(
            np.mean(np.abs(residuals[nonzero_mask] / y_true[nonzero_mask])) * 100.0
        )
    else:
        mape = float("nan")

    return {"mae": mae, "mse": mse, "rmse": rmse, "r2": r2, "mape": mape}


# ---------------------------------------------------------------------------
# Report formatting
# ---------------------------------------------------------------------------

def format_regression_report(metrics_dict: Dict[str, float]) -> str:
    """Return a human-readable string summary of regression metrics.

    Parameters
    ----------
    metrics_dict:
        Dictionary as returned by :func:`compute_regression_metrics`.

    Returns
    -------
    Formatted multi-line string.
    """
    mape_str = (
        f"{metrics_dict['mape']:.4f} %"
        if not np.isnan(metrics_dict["mape"])
        else "N/A (all true values are zero)"
    )

    lines = [
        "=" * 40,
        "  Regression Evaluation Report",
        "=" * 40,
        f"  MAE  (Mean Absolute Error)         : {metrics_dict['mae']:.6f}",
        f"  MSE  (Mean Squared Error)           : {metrics_dict['mse']:.6f}",
        f"  RMSE (Root Mean Squared Error)      : {metrics_dict['rmse']:.6f}",
        f"  R²   (Coefficient of Determination) : {metrics_dict['r2']:.6f}",
        f"  MAPE (Mean Abs. Percentage Error)   : {mape_str}",
        "=" * 40,
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Diagnostic plots
# ---------------------------------------------------------------------------

def residual_plot(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    class_names: Optional[List[str]] = None,  # kept for API compatibility
) -> Figure:
    """Generate a three-panel residual diagnostic figure.

    Panels
    ------
    1. Predicted vs Actual scatter with identity line.
    2. Residuals vs Predicted values with horizontal zero line.
    3. Histogram of residuals.

    Parameters
    ----------
    y_true:
        Ground-truth target values.
    y_pred:
        Model predictions.
    class_names:
        Unused for regression; accepted for a consistent API signature.

    Returns
    -------
    matplotlib Figure
    """
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    residuals = y_true - y_pred

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Regression Diagnostic Plots", fontsize=14, fontweight="bold")

    # --- Panel 1: Predicted vs Actual ---
    ax = axes[0]
    ax.scatter(y_true, y_pred, alpha=0.5, edgecolors="none", color="steelblue", s=20)
    combined_min = min(y_true.min(), y_pred.min())
    combined_max = max(y_true.max(), y_pred.max())
    ax.plot(
        [combined_min, combined_max],
        [combined_min, combined_max],
        "r--",
        linewidth=1.5,
        label="y = x (ideal)",
    )
    ax.set_xlabel("Actual Values")
    ax.set_ylabel("Predicted Values")
    ax.set_title("Predicted vs Actual")
    ax.legend(fontsize=8)

    # --- Panel 2: Residuals vs Predicted ---
    ax = axes[1]
    ax.scatter(y_pred, residuals, alpha=0.5, edgecolors="none", color="darkorange", s=20)
    ax.axhline(0, color="red", linestyle="--", linewidth=1.5, label="Zero line")
    ax.set_xlabel("Predicted Values")
    ax.set_ylabel("Residuals (Actual − Predicted)")
    ax.set_title("Residuals vs Predicted")
    ax.legend(fontsize=8)

    # --- Panel 3: Residuals histogram ---
    ax = axes[2]
    ax.hist(residuals, bins="auto", color="mediumpurple", edgecolor="white", linewidth=0.5)
    ax.axvline(0, color="red", linestyle="--", linewidth=1.5, label="Zero")
    ax.set_xlabel("Residual Value")
    ax.set_ylabel("Frequency")
    ax.set_title("Residuals Distribution")
    ax.legend(fontsize=8)

    fig.tight_layout()
    return fig


def prediction_error_plot(
    y_true: ArrayLike,
    y_pred: ArrayLike,
) -> Figure:
    """Plot sorted actual values versus predicted values.

    The samples are sorted by their true value so trends in prediction
    accuracy are visually apparent across the value range.

    Parameters
    ----------
    y_true:
        Ground-truth target values.
    y_pred:
        Model predictions.

    Returns
    -------
    matplotlib Figure
    """
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()

    sort_idx = np.argsort(y_true)
    y_true_sorted = y_true[sort_idx]
    y_pred_sorted = y_pred[sort_idx]
    x = np.arange(len(y_true_sorted))

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(x, y_true_sorted, label="Actual", color="steelblue", linewidth=1.5)
    ax.plot(
        x,
        y_pred_sorted,
        label="Predicted",
        color="darkorange",
        linewidth=1.5,
        linestyle="--",
    )
    ax.set_xlabel("Sample Index (sorted by actual value)")
    ax.set_ylabel("Value")
    ax.set_title("Prediction Error Plot — Sorted Actual vs Predicted")
    ax.legend()
    fig.tight_layout()
    return fig

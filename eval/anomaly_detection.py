"""
eval/anomaly_detection.py

Anomaly detection analysis utilities for Autoencoder models used inside the
NoCode Deep Learning platform.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from matplotlib.figure import Figure
from torch.utils.data import DataLoader


# ---------------------------------------------------------------------------
# Reconstruction error computation
# ---------------------------------------------------------------------------

def compute_reconstruction_errors(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
) -> np.ndarray:
    """Compute per-sample reconstruction errors (MSE) for an Autoencoder.

    The model is evaluated in inference mode (``model.eval()`` +
    ``torch.no_grad()``).  Each sample's reconstruction error is the mean
    squared error between its flattened input and the corresponding model
    output.

    Parameters
    ----------
    model:
        Autoencoder whose ``forward(x)`` returns the reconstruction tensor
        with the same shape as the input.
    data_loader:
        DataLoader that yields batches.  If each item is a tuple/list the
        first element is used as the input tensor; otherwise the item itself
        is used.
    device:
        Torch device to move tensors to before the forward pass.

    Returns
    -------
    np.ndarray of shape ``(N,)`` containing one MSE value per sample.
    """
    model.eval()
    errors: list[float] = []

    with torch.no_grad():
        for batch in data_loader:
            # Support (X, y) tuples as well as plain X batches
            if isinstance(batch, (list, tuple)):
                x = batch[0]
            else:
                x = batch

            x = x.to(device, non_blocking=True)
            reconstruction = model(x)

            # Flatten to (batch_size, -1) before computing MSE per sample
            x_flat = x.view(x.size(0), -1).float()
            rec_flat = reconstruction.view(reconstruction.size(0), -1).float()

            # MSE per sample: mean over feature dimension
            sample_mse = torch.mean((x_flat - rec_flat) ** 2, dim=1)
            errors.extend(sample_mse.cpu().numpy().tolist())

    return np.array(errors, dtype=float)


# ---------------------------------------------------------------------------
# Anomaly finding
# ---------------------------------------------------------------------------

def find_anomalies(
    errors: np.ndarray,
    threshold: Optional[float] = None,
    percentile: float = 95,
) -> Dict:
    """Identify anomalous samples based on reconstruction error.

    Parameters
    ----------
    errors:
        1-D array of per-sample reconstruction errors as returned by
        :func:`compute_reconstruction_errors`.
    threshold:
        Explicit anomaly threshold.  If ``None`` the threshold is derived
        automatically from ``percentile``.
    percentile:
        Percentile of the error distribution used to auto-set the threshold
        when ``threshold`` is ``None``.  Default: 95.

    Returns
    -------
    dict with keys:

    ``threshold``
        The (possibly auto-derived) anomaly threshold value.
    ``anomaly_mask``
        Boolean array of shape ``(N,)`` — ``True`` where anomalous.
    ``n_anomalies``
        Integer count of detected anomalies.
    ``anomaly_ratio``
        Fraction of samples flagged as anomalous (0–1).
    ``error_stats``
        Sub-dict with ``mean``, ``std``, ``min``, ``max``, ``p95``, ``p99``.
    """
    errors = np.asarray(errors, dtype=float).ravel()

    if threshold is None:
        threshold = float(np.percentile(errors, percentile))

    anomaly_mask = errors > threshold
    n_anomalies = int(anomaly_mask.sum())

    error_stats = {
        "mean": float(np.mean(errors)),
        "std": float(np.std(errors)),
        "min": float(np.min(errors)),
        "max": float(np.max(errors)),
        "p95": float(np.percentile(errors, 95)),
        "p99": float(np.percentile(errors, 99)),
    }

    return {
        "threshold": float(threshold),
        "anomaly_mask": anomaly_mask,
        "n_anomalies": n_anomalies,
        "anomaly_ratio": n_anomalies / len(errors) if len(errors) > 0 else 0.0,
        "error_stats": error_stats,
    }


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def reconstruction_error_plot(
    errors: np.ndarray,
    threshold: Optional[float] = None,
    percentile: float = 95,
) -> Figure:
    """Two-panel figure visualising reconstruction error distribution.

    Panel 1 — Histogram
        Distribution of reconstruction errors with the anomaly threshold
        drawn as a vertical red dashed line.

    Panel 2 — Sorted error curve
        Reconstruction errors sorted in ascending order, giving an
        anomaly-score curve.  The threshold is shown as a horizontal line.

    Parameters
    ----------
    errors:
        1-D array of per-sample reconstruction errors.
    threshold:
        Explicit threshold.  Auto-derived from ``percentile`` when ``None``.
    percentile:
        Percentile used to auto-set the threshold.  Default: 95.

    Returns
    -------
    matplotlib Figure
    """
    errors = np.asarray(errors, dtype=float).ravel()
    result = find_anomalies(errors, threshold=threshold, percentile=percentile)
    thresh = result["threshold"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Reconstruction Error Analysis", fontsize=14, fontweight="bold")

    # --- Panel 1: Histogram ---
    ax = axes[0]
    ax.hist(errors, bins="auto", color="steelblue", edgecolor="white", linewidth=0.5, alpha=0.85)
    ax.axvline(
        thresh,
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Threshold ({thresh:.4f})",
    )
    ax.set_xlabel("Reconstruction Error (MSE)")
    ax.set_ylabel("Frequency")
    ax.set_title("Error Distribution")
    ax.legend(fontsize=9)

    # --- Panel 2: Sorted error curve ---
    ax = axes[1]
    sorted_errors = np.sort(errors)
    ax.plot(sorted_errors, color="steelblue", linewidth=1.5, label="Sorted errors")
    ax.axhline(
        thresh,
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Threshold ({thresh:.4f})",
    )
    ax.set_xlabel("Sample Rank (sorted by error)")
    ax.set_ylabel("Reconstruction Error (MSE)")
    ax.set_title("Anomaly Score Curve")
    ax.legend(fontsize=9)

    fig.tight_layout()
    return fig


def anomaly_scatter(
    errors: np.ndarray,
    features_2d: np.ndarray,
    threshold: Optional[float] = None,
    percentile: float = 95,
) -> Figure:
    """Scatter plot in 2-D feature space coloured by anomaly status.

    Normal samples are plotted in blue; anomalous samples in red.  The
    ``features_2d`` array is typically obtained from dimensionality reduction
    (e.g. t-SNE or UMAP) applied to the original data or to the latent space.

    Parameters
    ----------
    errors:
        1-D array of per-sample reconstruction errors.
    features_2d:
        Array of shape ``(N, 2)`` with 2-D coordinates for each sample.
    threshold:
        Explicit threshold.  Auto-derived from ``percentile`` when ``None``.
    percentile:
        Percentile used to auto-set the threshold.  Default: 95.

    Returns
    -------
    matplotlib Figure
    """
    errors = np.asarray(errors, dtype=float).ravel()
    features_2d = np.asarray(features_2d, dtype=float)

    if features_2d.ndim != 2 or features_2d.shape[1] != 2:
        raise ValueError(
            f"features_2d must have shape (N, 2), got {features_2d.shape}"
        )
    if len(errors) != features_2d.shape[0]:
        raise ValueError(
            f"Length mismatch: errors has {len(errors)} samples but "
            f"features_2d has {features_2d.shape[0]} rows."
        )

    result = find_anomalies(errors, threshold=threshold, percentile=percentile)
    anomaly_mask = result["anomaly_mask"]
    thresh = result["threshold"]
    n_anomalies = result["n_anomalies"]
    n_normal = len(errors) - n_anomalies

    fig, ax = plt.subplots(figsize=(9, 7))

    # Normal samples
    ax.scatter(
        features_2d[~anomaly_mask, 0],
        features_2d[~anomaly_mask, 1],
        c="steelblue",
        alpha=0.5,
        s=15,
        edgecolors="none",
        label=f"Normal (n={n_normal})",
    )
    # Anomalous samples
    if n_anomalies > 0:
        ax.scatter(
            features_2d[anomaly_mask, 0],
            features_2d[anomaly_mask, 1],
            c="red",
            alpha=0.8,
            s=30,
            edgecolors="darkred",
            linewidths=0.5,
            label=f"Anomaly (n={n_anomalies})",
        )

    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")
    ax.set_title(
        f"Anomaly Detection Scatter\n"
        f"Threshold = {thresh:.4f} | "
        f"{n_anomalies}/{len(errors)} anomalies "
        f"({result['anomaly_ratio'] * 100:.1f} %)"
    )
    ax.legend(fontsize=9)
    fig.tight_layout()
    return fig

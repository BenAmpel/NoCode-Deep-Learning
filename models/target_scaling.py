"""Standardize regression targets inside the model.

A freshly initialised network outputs values near 0. When the target is, say,
a price around 250 000 or a score around 80, it spends most of training just
learning the target's scale. Wrapping the network so it predicts
``(y - mean) / std`` and un-scales its own output fixes that without touching
anything downstream: evaluation, ONNX export and inference all still see
predictions in the original units, because the mean and std are buffers of
the exported model.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn


class TargetScaler(nn.Module):
    def __init__(self, model: nn.Module, mean: float, std: float):
        super().__init__()
        self.model = model
        self.register_buffer("target_mean", torch.tensor(float(mean), dtype=torch.float32))
        self.register_buffer("target_std", torch.tensor(float(std), dtype=torch.float32))

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs) * self.target_std + self.target_mean


def wrap_for_regression(model: nn.Module, y_train) -> tuple[nn.Module, tuple[float, float]]:
    """Wrap ``model`` using the training targets' mean and std.

    Pass training targets only, so validation data doesn't shape the model.
    A constant target gets std 1.0.
    """
    y = np.asarray(y_train, dtype=np.float64).reshape(-1)
    mean = float(y.mean()) if y.size else 0.0
    std = float(y.std()) if y.size else 1.0
    if not np.isfinite(std) or std < 1e-8:
        std = 1.0
    return TargetScaler(model, mean, std), (mean, std)

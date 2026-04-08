"""
Text-based model architecture summary and parameter count visualisation.
Returns a matplotlib figure and a summary string.
"""
from __future__ import annotations
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


def summarise_model(model: nn.Module, input_shape: tuple = None) -> tuple[str, plt.Figure]:
    """
    Returns (summary_text, figure) where figure shows a layer-by-layer bar chart
    of parameter counts.
    """
    summary_text = _text_summary(model)
    fig          = _param_chart(model)
    return summary_text, fig


def _text_summary(model: nn.Module) -> str:
    lines = [f"{'Layer':<40} {'Type':<20} {'Params':>12}", "-" * 74]
    total = 0
    trainable = 0
    for name, module in model.named_modules():
        if not list(module.children()):   # leaf modules only
            n_params = sum(p.numel() for p in module.parameters())
            n_train  = sum(p.numel() for p in module.parameters() if p.requires_grad)
            total    += n_params
            trainable += n_train
            if n_params > 0:
                lines.append(f"{name:<40} {type(module).__name__:<20} {n_params:>12,}")
    lines += [
        "-" * 74,
        f"{'Total parameters':<40} {'':20} {total:>12,}",
        f"{'Trainable parameters':<40} {'':20} {trainable:>12,}",
        f"{'Frozen parameters':<40} {'':20} {total - trainable:>12,}",
    ]
    return "\n".join(lines)


def _param_chart(model: nn.Module) -> plt.Figure:
    names, param_counts, trainable_flags = [], [], []
    for name, module in model.named_modules():
        if not list(module.children()):
            n_total = sum(p.numel() for p in module.parameters())
            n_train = sum(p.numel() for p in module.parameters() if p.requires_grad)
            if n_total > 0:
                names.append(name or type(module).__name__)
                param_counts.append(n_total)
                trainable_flags.append(n_train > 0)

    if not names:
        fig, ax = plt.subplots(figsize=(6, 2))
        ax.text(0.5, 0.5, "No parameterised layers", ha="center", va="center")
        ax.axis("off")
        return fig

    n = len(names)
    colors = ["steelblue" if t else "lightcoral" for t in trainable_flags]
    fig, ax = plt.subplots(figsize=(max(7, n * 0.4), 4))
    ax.bar(range(n), param_counts, color=colors)
    ax.set_xticks(range(n))
    ax.set_xticklabels(names, rotation=60, ha="right", fontsize=6)
    ax.set_ylabel("Parameter count")
    ax.set_title("Parameters per layer")
    legend_patches = [
        mpatches.Patch(color="steelblue",   label="Trainable"),
        mpatches.Patch(color="lightcoral",  label="Frozen"),
    ]
    ax.legend(handles=legend_patches, fontsize=8)
    fig.tight_layout()
    return fig


def estimate_training_time(
    model: nn.Module,
    n_train_samples: int,
    batch_size: int,
    epochs: int,
    device: str,
) -> str:
    """Returns a human-readable estimate string."""
    n_params  = sum(p.numel() for p in model.parameters() if p.requires_grad)
    steps     = (n_train_samples / max(batch_size, 1)) * epochs
    # Rough heuristic: ~1ms per 1M trainable params per step on CPU
    speed_map = {"cpu": 1.0, "mps": 0.25, "cuda": 0.05}
    ms_per_step = (n_params / 1e6) * speed_map.get(device, 1.0)
    total_sec = ms_per_step * steps / 1000
    if total_sec < 60:
        return f"~{int(total_sec)}s"
    if total_sec < 3600:
        return f"~{int(total_sec / 60)}m"
    return f"~{total_sec / 3600:.1f}h"

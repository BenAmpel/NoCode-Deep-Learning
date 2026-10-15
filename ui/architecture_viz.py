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
        plt.close(fig)  # rendered by Gradio after return; frees pyplot's reference
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
    plt.close(fig)  # rendered by Gradio after return; frees pyplot's reference
    return fig


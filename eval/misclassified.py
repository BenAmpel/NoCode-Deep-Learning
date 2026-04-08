"""Collect and visualise misclassified validation samples."""
from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

def find_misclassified(
    model,
    val_loader,
    classes: list[str],
    modality: str,
    prep: dict,
    task: str = "classification",
    val_samples: list | None = None,
    val_texts:   list | None = None,
    X_val: np.ndarray | None = None,
    y_val: np.ndarray | None = None,
    is_sklearn: bool = False,
    n_show: int = 12,
) -> tuple[plt.Figure | None, pd.DataFrame | None]:
    if task == "clustering":
        return None, None

    from config import DEVICE

    # Collect (true, pred) for every val sample
    y_true_all, y_pred_all = [], []

    if is_sklearn and X_val is not None:
        y_pred_all = model.predict(X_val).tolist()
        y_true_all = y_val.tolist()
    else:
        model = model.to(DEVICE)
        model.eval()
        with torch.no_grad():
            for batch in val_loader:
                inputs, labels = batch
                if isinstance(inputs, dict):
                    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
                    logits = model(inputs["input_ids"], inputs.get("attention_mask")) if "attention_mask" in inputs else model(inputs["input_ids"])
                else:
                    logits = model(inputs.to(DEVICE))
                y_pred_all.extend(logits.argmax(1).cpu().numpy().tolist())
                y_true_all.extend(labels.numpy().tolist())

    y_true_all = np.array(y_true_all)
    y_pred_all = np.array(y_pred_all)
    mis_idx    = np.where(y_true_all != y_pred_all)[0]

    if len(mis_idx) == 0:
        empty_df = pd.DataFrame(columns=["true_label", "predicted_label"])
        return None, empty_df

    mis_idx = mis_idx[:n_show]

    # Text / tabular / timeseries → DataFrame
    if modality == "text" and val_texts is not None:
        rows = [{"text": val_texts[i], "true_label": classes[y_true_all[i]],
                 "predicted_label": classes[y_pred_all[i]]} for i in mis_idx]
        return None, pd.DataFrame(rows)

    if modality in ("tabular", "timeseries"):
        rows = [{"index": int(i), "true_label": classes[y_true_all[i]],
                 "predicted_label": classes[y_pred_all[i]]} for i in mis_idx]
        if modality == "tabular" and X_val is not None:
            num_cols = prep.get("numeric_columns", [])
            for row, i in zip(rows, mis_idx):
                for j, col in enumerate(num_cols):
                    row[col] = float(X_val[i, j]) if j < X_val.shape[1] else None
        return None, pd.DataFrame(rows)

    # Visual modalities (image, audio, video) → grid figure
    mean = np.array(prep.get("mean", [0.5, 0.5, 0.5])).reshape(3, 1, 1)
    std  = np.array(prep.get("std",  [0.5, 0.5, 0.5])).reshape(3, 1, 1)

    all_inputs = []
    for batch in val_loader:
        inputs, _ = batch
        if isinstance(inputs, dict):
            return None, pd.DataFrame(columns=["true_label", "predicted_label"])
        all_inputs.append(inputs.cpu())
    if not all_inputs:
        return None, None
    all_inputs = torch.cat(all_inputs, dim=0)

    n_cols = 3
    n_rows = int(np.ceil(len(mis_idx) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3, n_rows * 3))
    axes = np.array(axes).reshape(-1)
    for ax in axes:
        ax.axis("off")

    for plot_i, sample_i in enumerate(mis_idx):
        if sample_i >= len(all_inputs):
            break
        tensor = all_inputs[sample_i]
        if modality == "video":
            mid = tensor.shape[1] // 2
            frame = tensor[:, mid, :, :].numpy()
        else:
            frame = tensor.numpy()
        img = (frame * std + mean).clip(0, 1).transpose(1, 2, 0)
        axes[plot_i].imshow(img)
        axes[plot_i].set_title(
            f"T: {classes[y_true_all[sample_i]]}\nP: {classes[y_pred_all[sample_i]]}",
            fontsize=7, color="red",
        )
        axes[plot_i].axis("off")

    fig.suptitle(f"Misclassified samples ({len(mis_idx)} shown)", fontsize=10)
    fig.tight_layout()
    return fig, None

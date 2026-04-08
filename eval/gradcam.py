"""
Grad-CAM heatmaps for image, audio (mel spectrogram), and video models.
Returns None gracefully for non-convolutional models.
"""
from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.cm as cm

GRADCAM_MODALITIES = {"image", "audio", "video"}

def _get_target_layer(model, model_name: str) -> nn.Module | None:
    from models.image_models import MobileNetV3Wrapper, TinyCNN
    from models.video_models import R3DWrapper, TinyR3D
    if isinstance(model, MobileNetV3Wrapper):
        return model.features[-1]
    if isinstance(model, TinyCNN):
        return model.features[8]   # last Conv2d(64,128)
    if isinstance(model, R3DWrapper):
        return model.layer4
    if isinstance(model, TinyR3D):
        return list(model.features.children())[4]  # Conv3d(16,32)
    return None

def compute_gradcam(model, val_loader, modality: str, model_name: str, prep: dict, n_samples: int = 6) -> plt.Figure | None:
    if modality not in GRADCAM_MODALITIES:
        return None
    target_layer = _get_target_layer(model, model_name)
    if target_layer is None:
        return None

    # Always run GradCAM on CPU to avoid MPS hook issues
    model_cpu = model.to("cpu").eval()
    activations, gradients = [], []

    def fwd_hook(_, __, output):
        activations.append(output.detach())

    def bwd_hook(_, __, grad_out):
        gradients.append(grad_out[0].detach())

    fh = target_layer.register_forward_hook(fwd_hook)
    bh = target_layer.register_full_backward_hook(bwd_hook)

    samples_shown = 0
    results = []   # list of (input_img_np, cam_np, true_label, pred_label)

    for batch in val_loader:
        if samples_shown >= n_samples:
            break
        inputs, labels = batch
        if isinstance(inputs, dict):
            fh.remove(); bh.remove()
            return None  # text model — no CAM
        inputs = inputs.to("cpu")

        for i in range(min(len(inputs), n_samples - samples_shown)):
            x = inputs[i:i+1].requires_grad_(True)
            activations.clear(); gradients.clear()

            with torch.enable_grad():
                logits = model_cpu(x)
                pred_idx = logits.argmax(1).item()
                score = logits[0, pred_idx]
                model_cpu.zero_grad()
                score.backward()

            if not activations or not gradients:
                continue

            act  = activations[0].squeeze(0)   # (C, H, W) or (C, T, H, W)
            grad = gradients[0].squeeze(0)

            if act.dim() == 4:  # 3D video conv
                weights = grad.mean(dim=(1, 2, 3), keepdim=True)
                cam = (weights * act).sum(0).mean(0)  # (H, W) averaged over T
            else:               # 2D image/audio conv
                weights = grad.mean(dim=(1, 2), keepdim=True)
                cam = (weights * act).sum(0)

            cam = torch.relu(cam)
            cam = cam - cam.min()
            cam = cam / (cam.max() + 1e-8)
            cam_np = cam.numpy()

            # Denormalize original input to display
            mean = np.array(prep.get("mean", [0.5, 0.5, 0.5])).reshape(3, 1, 1)
            std  = np.array(prep.get("std",  [0.5, 0.5, 0.5])).reshape(3, 1, 1)

            if modality == "video":
                img_np = inputs[i].numpy()  # (C, T, H, W)
                mid = img_np.shape[1] // 2
                frame = img_np[:, mid, :, :]  # (C, H, W)
            else:
                frame = inputs[i].numpy()   # (C, H, W)

            frame = (frame * std + mean).clip(0, 1).transpose(1, 2, 0)  # HWC

            from PIL import Image
            cam_resized = np.array(Image.fromarray((cam_np * 255).astype(np.uint8))
                                   .resize((frame.shape[1], frame.shape[0]))) / 255.0
            heatmap = cm.jet(cam_resized)[:, :, :3]
            overlay = 0.55 * frame + 0.45 * heatmap
            overlay = overlay.clip(0, 1)

            results.append((frame, overlay, labels[i].item(), pred_idx))
            samples_shown += 1

    fh.remove(); bh.remove()
    if not results:
        return None

    n_cols = 2
    n_rows = len(results)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6, 3 * n_rows))
    if n_rows == 1:
        axes = [axes]
    for row, (orig, overlay, true_idx, pred_idx) in enumerate(results):
        axes[row][0].imshow(orig)
        axes[row][0].set_title(f"Input (true: class {true_idx})", fontsize=8)
        axes[row][0].axis("off")
        axes[row][1].imshow(overlay)
        axes[row][1].set_title(f"Grad-CAM (pred: class {pred_idx})", fontsize=8)
        axes[row][1].axis("off")
    fig.suptitle("Grad-CAM — where the model focused", fontsize=10, fontweight="bold")
    fig.tight_layout()
    return fig

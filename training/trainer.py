"""
Unified training loop — classification, multi-label, and regression,
with optional LR scheduling, AMP (CUDA only), class weights, and checkpointing.
"""
from __future__ import annotations
import time
from typing import Generator

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import (
    CosineAnnealingLR, StepLR, SequentialLR,
    LinearLR, CosineAnnealingLR as CosLR,
)
from config import DEVICE

# ---------------------------------------------------------------------------
# Batch helpers
# ---------------------------------------------------------------------------

def _forward(model: nn.Module, batch, device: str = DEVICE) -> tuple[torch.Tensor, torch.Tensor]:
    inputs, labels = batch
    labels = labels.to(device)
    if isinstance(inputs, dict):
        inputs = {k: v.to(device) for k, v in inputs.items()}
        if "attention_mask" in inputs:
            logits = model(inputs["input_ids"], inputs["attention_mask"])
        else:
            logits = model(inputs["input_ids"])
    else:
        logits = model(inputs.to(device))
    return logits, labels

# ---------------------------------------------------------------------------
# Scheduler factory
# ---------------------------------------------------------------------------

def _make_scheduler(optimizer, name: str, epochs: int, steps_per_epoch: int):
    name = name.lower()
    if name == "none":
        return None
    if name == "cosine":
        return CosineAnnealingLR(optimizer, T_max=epochs)
    if name == "step":
        return StepLR(optimizer, step_size=max(1, epochs // 3), gamma=0.5)
    if name == "warmup_cosine":
        warmup = LinearLR(optimizer, start_factor=0.1, end_factor=1.0,
                          total_iters=max(1, epochs // 10))
        cosine = CosLR(optimizer, T_max=epochs - max(1, epochs // 10))
        return SequentialLR(optimizer, schedulers=[warmup, cosine],
                            milestones=[max(1, epochs // 10)])
    return None

# ---------------------------------------------------------------------------
# PyTorch training loop
# ---------------------------------------------------------------------------

def train_pytorch(
    model: nn.Module,
    train_loader,
    val_loader,
    epochs: int,
    lr: float,
    optimizer_name: str  = "adam",
    patience: int        = 3,
    scheduler_name: str  = "cosine",
    use_amp: bool        = False,
    task: str            = "classification",
    class_weights: torch.Tensor | None = None,
    checkpoint_dir: str | None         = None,
    checkpoint_every: int              = 0,   # 0 = only save best
    device: str                        = DEVICE,
) -> Generator[dict, None, None]:
    """
    Unified training generator.

    Yields per-epoch progress dicts, then a final {"done": True, ...} dict.

    Parameters
    ----------
    task : "classification" | "multi-label" | "regression"
    class_weights : optional tensor of shape (num_classes,) for imbalanced data
    checkpoint_dir : if set, saves checkpoints here
    checkpoint_every : save a checkpoint every N epochs (0 = best-only)
    """
    model     = model.to(device)
    is_clf    = (task == "classification")
    is_multi  = (task == "multi-label")
    is_reg    = (task == "regression")

    # Loss function
    if is_clf:
        w = class_weights.to(device) if class_weights is not None else None
        criterion = nn.CrossEntropyLoss(weight=w)
    elif is_multi:
        criterion = nn.BCEWithLogitsLoss()
    else:  # regression
        criterion = nn.MSELoss()

    optimizer = _make_optimizer(model, optimizer_name, lr)
    scheduler = _make_scheduler(optimizer, scheduler_name, epochs, len(train_loader))

    # AMP on CUDA (with GradScaler) and MPS (autocast only, no scaler)
    _cuda_amp   = use_amp and device == "cuda" and torch.cuda.is_available()
    _mps_amp    = use_amp and device == "mps" and torch.backends.mps.is_available()
    amp_enabled = _cuda_amp or _mps_amp
    scaler      = torch.cuda.amp.GradScaler() if _cuda_amp else None

    best_val      = float("inf")
    patience_ctr  = 0
    best_state    = None
    best_epoch    = 0
    history       = []
    epoch_times   = []

    for epoch in range(1, epochs + 1):
        t0 = time.time()

        # ---- train ----
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            optimizer.zero_grad()
            if amp_enabled:
                with torch.autocast(device):
                    logits, labels = _forward(model, batch, device=device)
                    loss = _compute_loss(criterion, logits, labels, task)
                if scaler:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()
            else:
                logits, labels = _forward(model, batch, device=device)
                loss = _compute_loss(criterion, logits, labels, task)
                loss.backward()
                optimizer.step()
            train_loss += loss.item()
        train_loss /= max(len(train_loader), 1)

        # ---- validate ----
        model.eval()
        val_loss   = 0.0
        val_metric = 0.0
        n_correct = n_total = 0
        with torch.no_grad():
            for batch in val_loader:
                logits, labels = _forward(model, batch, device=device)
                lv = _compute_loss(criterion, logits, labels, task)
                val_loss += lv.item()
                if is_clf:
                    n_correct += (logits.argmax(1) == labels).sum().item()
                    n_total   += labels.size(0)
                elif is_multi:
                    preds = (torch.sigmoid(logits) > 0.5).float()
                    val_metric += (preds == labels.float()).float().mean().item()
                else:  # regression → MAE
                    val_metric += (logits.squeeze() - labels.float()).abs().mean().item()

        val_loss /= max(len(val_loader), 1)
        if is_clf:
            val_metric = (n_correct / max(n_total, 1)) * 100
        elif is_multi:
            val_metric = (val_metric / max(len(val_loader), 1)) * 100
        else:
            val_metric /= max(len(val_loader), 1)

        if scheduler:
            scheduler.step()

        elapsed = time.time() - t0
        epoch_times.append(elapsed)
        avg_epoch_time = sum(epoch_times) / len(epoch_times)
        eta = avg_epoch_time * (epochs - epoch)

        record = {
            "epoch":       epoch,
            "train_loss":  round(train_loss,  4),
            "val_loss":    round(val_loss,     4),
            "val_acc":     round(val_metric,   2),
            "eta_seconds": round(eta,          1),
            "lr":          round(optimizer.param_groups[0]["lr"], 6),
        }
        history.append(record)

        # Checkpointing
        if checkpoint_dir:
            if checkpoint_every > 0 and epoch % checkpoint_every == 0:
                _save_ckpt(model, optimizer, epoch, history, val_loss, checkpoint_dir,
                           tag=f"epoch{epoch:03d}")
            if val_loss < best_val:
                _save_ckpt(model, optimizer, epoch, history, val_loss, checkpoint_dir,
                           tag="best")

        # Early stopping
        if val_loss < best_val:
            best_val     = val_loss
            best_epoch   = epoch
            patience_ctr = 0
            best_state   = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_ctr += 1

        stopped = patience_ctr >= patience
        yield {**record, "epochs": epochs, "done": False, "stopped_early": stopped}
        if stopped:
            break

    if best_state:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})

    yield {"done": True, "model": model, "history": history,
           "best_epoch": best_epoch, "best_val": best_val}


# ---------------------------------------------------------------------------
# Loss dispatcher
# ---------------------------------------------------------------------------

def _compute_loss(criterion, logits, labels, task: str) -> torch.Tensor:
    if task == "classification":
        return criterion(logits, labels)
    if task == "multi-label":
        return criterion(logits, labels.float())
    # regression
    return criterion(logits.squeeze(), labels.float())


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def _save_ckpt(model, optimizer, epoch, history, val_loss, ckpt_dir, tag):
    import os, json
    os.makedirs(ckpt_dir, exist_ok=True)
    path = os.path.join(ckpt_dir, f"checkpoint_{tag}.pt")
    torch.save({
        "epoch":      epoch,
        "model":      model.state_dict(),
        "optimizer":  optimizer.state_dict(),
        "history":    history,
        "val_loss":   val_loss,
    }, path)
    # Write latest pointer
    with open(os.path.join(ckpt_dir, "latest.json"), "w") as f:
        json.dump({"path": path, "epoch": epoch, "val_loss": round(val_loss, 6)}, f)


# ---------------------------------------------------------------------------
# Class-weight helpers
# ---------------------------------------------------------------------------

def compute_class_weights(labels_list: list) -> torch.Tensor:
    """
    Compute inverse-frequency class weights for imbalanced datasets.

    Parameters
    ----------
    labels_list : flat list of integer class indices

    Returns
    -------
    torch.Tensor of shape (n_classes,) with float weights
    """
    import numpy as np
    labels = np.array(labels_list)
    classes, counts = np.unique(labels, return_counts=True)
    # inverse frequency, normalised so mean weight ≈ 1
    weights = 1.0 / counts.astype(float)
    weights = weights / weights.mean()
    weight_tensor = torch.ones(int(classes.max()) + 1)
    for cls, w in zip(classes, weights):
        weight_tensor[int(cls)] = float(w)
    return weight_tensor


# ---------------------------------------------------------------------------
# sklearn trainer
# ---------------------------------------------------------------------------

def train_sklearn(model, X_train, y_train, X_val=None, y_val=None) -> Generator[dict, None, None]:
    yield {"epoch": 1, "epochs": 1, "train_loss": 0.0, "val_loss": 0.0,
           "val_acc": 0.0, "done": False, "stopped_early": False,
           "status": "Fitting model…", "eta_seconds": 0}
    model.fit(X_train, y_train)
    val_acc = float(model.score(X_val, y_val)) * 100 if X_val is not None else 0.0
    yield {
        "done": True,
        "model": model,
        "history": [{"epoch": 1, "train_loss": 0.0, "val_loss": 0.0,
                     "val_acc": round(val_acc, 2), "eta_seconds": 0, "lr": 0}],
        "best_epoch": 1, "best_val": 0.0,
    }


# ---------------------------------------------------------------------------
# Optimizer factory
# ---------------------------------------------------------------------------

def _make_optimizer(model: nn.Module, name: str, lr: float) -> optim.Optimizer:
    params = [p for p in model.parameters() if p.requires_grad]
    name   = name.lower()
    if name == "adam":  return optim.Adam(params, lr=lr)
    if name == "adamw": return optim.AdamW(params, lr=lr)
    if name == "sgd":   return optim.SGD(params,  lr=lr, momentum=0.9)
    raise ValueError(f"Unknown optimizer: {name}")

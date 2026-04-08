"""
Model checkpoint utilities: save, load, list, and prune checkpoints.

Checkpoints are standard PyTorch ``.pt`` files containing model weights,
optimizer state, training epoch, loss history, and the best validation loss
observed so far.  A lightweight ``latest.json`` sidecar is kept up-to-date so
callers can quickly resume from the most recent checkpoint without scanning the
directory.
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.optim as optim

from config import DEVICE

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_LATEST_JSON = "latest.json"
_CHECKPOINT_RE = re.compile(r"_epoch(\d{3})\.pt$")

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------

def save_checkpoint(
    model: nn.Module,
    optimizer: optim.Optimizer,
    epoch: int,
    history: list[dict[str, Any]],
    val_loss: float,
    checkpoint_dir: str,
    model_name: str = "model",
) -> str:
    """Persist a training checkpoint to disk.

    The checkpoint file is named ``{model_name}_epoch{epoch:03d}.pt`` and
    lives inside *checkpoint_dir*, which is created if it does not exist.
    After saving the checkpoint, ``{checkpoint_dir}/latest.json`` is updated
    so that :func:`get_latest_checkpoint` can find it instantly.

    Parameters
    ----------
    model:
        The ``nn.Module`` whose weights should be saved.
    optimizer:
        The optimizer whose state should be saved (enables exact resume).
    epoch:
        Current (1-based) epoch number.
    history:
        List of per-epoch metric dicts accumulated so far.
    val_loss:
        Validation loss at this epoch (used for "best checkpoint" decisions).
    checkpoint_dir:
        Directory where the checkpoint file will be written.
    model_name:
        Stem used to construct the filename.

    Returns
    -------
    str
        Absolute path of the saved checkpoint file.
    """
    dir_path = Path(checkpoint_dir)
    dir_path.mkdir(parents=True, exist_ok=True)

    filename   = f"{model_name}_epoch{epoch:03d}.pt"
    checkpoint_path = dir_path / filename

    payload: dict[str, Any] = {
        "epoch":           epoch,
        "model_state":     {k: v.cpu() for k, v in model.state_dict().items()},
        "optimizer_state": optimizer.state_dict(),
        "history":         history,
        "val_loss":        val_loss,
    }
    torch.save(payload, checkpoint_path)

    # Update latest.json
    latest_info = {
        "path":     str(checkpoint_path.resolve()),
        "epoch":    epoch,
        "val_loss": val_loss,
    }
    latest_path = dir_path / _LATEST_JSON
    with open(latest_path, "w", encoding="utf-8") as fh:
        json.dump(latest_info, fh, indent=2)

    return str(checkpoint_path.resolve())


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------

def load_checkpoint(
    checkpoint_path: str,
    model: nn.Module,
    optimizer: Optional[optim.Optimizer] = None,
) -> tuple[nn.Module, Optional[optim.Optimizer], int, list[dict[str, Any]], float]:
    """Restore model (and optionally optimizer) state from a checkpoint file.

    The checkpoint is mapped to the correct device automatically so the same
    file works on CPU, CUDA, and MPS hosts.

    Parameters
    ----------
    checkpoint_path:
        Path to a ``.pt`` checkpoint produced by :func:`save_checkpoint`.
    model:
        An ``nn.Module`` with the same architecture as was checkpointed.
        Weights are loaded in-place *and* the model is moved to ``DEVICE``.
    optimizer:
        If provided, optimizer state is restored in-place.  Pass ``None`` for
        inference-only usage.

    Returns
    -------
    tuple
        ``(model, optimizer_or_None, epoch, history, val_loss)``

        * ``model``  — weight-restored model on ``DEVICE``
        * ``optimizer_or_None`` — state-restored optimizer, or ``None``
        * ``epoch``  — epoch at which the checkpoint was saved
        * ``history`` — list of per-epoch metric dicts
        * ``val_loss`` — validation loss recorded in the checkpoint
    """
    # map_location handles CPU/MPS/CUDA portably
    map_location = torch.device(DEVICE)
    payload: dict[str, Any] = torch.load(
        checkpoint_path,
        map_location=map_location,
        weights_only=False,
    )

    model.load_state_dict(
        {k: v.to(DEVICE) for k, v in payload["model_state"].items()}
    )
    model.to(DEVICE)

    if optimizer is not None:
        optimizer.load_state_dict(payload["optimizer_state"])

    epoch   = int(payload.get("epoch", 0))
    history = payload.get("history", [])
    val_loss = float(payload.get("val_loss", float("inf")))

    return model, optimizer, epoch, history, val_loss


# ---------------------------------------------------------------------------
# List / query
# ---------------------------------------------------------------------------

def list_checkpoints(checkpoint_dir: str) -> list[dict[str, Any]]:
    """Return metadata for every checkpoint in *checkpoint_dir*, sorted by epoch.

    Only files whose names match the pattern ``*_epoch###.pt`` are returned.

    Parameters
    ----------
    checkpoint_dir:
        Directory to scan.

    Returns
    -------
    list[dict]
        Each entry is ``{"path": str, "epoch": int, "val_loss": float}``,
        sorted ascending by epoch.  ``val_loss`` is loaded from the checkpoint
        file itself so it is always accurate.
    """
    dir_path = Path(checkpoint_dir)
    if not dir_path.is_dir():
        return []

    results: list[dict[str, Any]] = []
    for entry in sorted(dir_path.iterdir()):
        if not entry.is_file():
            continue
        match = _CHECKPOINT_RE.search(entry.name)
        if match is None:
            continue
        epoch = int(match.group(1))
        # Load only the scalar metadata, not the full tensors
        try:
            payload = torch.load(
                entry,
                map_location=torch.device("cpu"),
                weights_only=False,
            )
            val_loss = float(payload.get("val_loss", float("inf")))
        except Exception:
            val_loss = float("inf")

        results.append({
            "path":     str(entry.resolve()),
            "epoch":    epoch,
            "val_loss": val_loss,
        })

    results.sort(key=lambda r: r["epoch"])
    return results


def get_latest_checkpoint(checkpoint_dir: str) -> Optional[str]:
    """Return the path stored in ``latest.json``, or ``None`` if absent.

    This is an O(1) operation — it reads the sidecar JSON written by
    :func:`save_checkpoint` rather than scanning the directory.

    Parameters
    ----------
    checkpoint_dir:
        Directory to look in.

    Returns
    -------
    str or None
        Absolute path to the latest checkpoint, or ``None`` if no
        ``latest.json`` exists or the path it references is missing.
    """
    latest_path = Path(checkpoint_dir) / _LATEST_JSON
    if not latest_path.is_file():
        return None

    try:
        with open(latest_path, "r", encoding="utf-8") as fh:
            info = json.load(fh)
        candidate = info.get("path")
        if candidate and Path(candidate).is_file():
            return str(candidate)
    except (json.JSONDecodeError, OSError):
        pass

    return None


# ---------------------------------------------------------------------------
# Cleanup
# ---------------------------------------------------------------------------

def cleanup_old_checkpoints(
    checkpoint_dir: str,
    keep_last_n: int = 3,
) -> list[str]:
    """Delete old checkpoints, retaining the best and the last *keep_last_n*.

    "Best" is defined as the checkpoint with the lowest ``val_loss``.
    "Last N" are the *keep_last_n* checkpoints with the highest epoch numbers.
    Checkpoints that qualify under either criterion are kept; all others are
    removed.

    Parameters
    ----------
    checkpoint_dir:
        Directory containing ``.pt`` checkpoints.
    keep_last_n:
        How many of the most recent (by epoch) checkpoints to keep in addition
        to the best checkpoint.

    Returns
    -------
    list[str]
        Paths of the files that were deleted.
    """
    checkpoints = list_checkpoints(checkpoint_dir)
    if len(checkpoints) <= keep_last_n:
        return []

    # Identify checkpoints to keep
    keep_paths: set[str] = set()

    # Best by val_loss
    best = min(checkpoints, key=lambda c: c["val_loss"])
    keep_paths.add(best["path"])

    # Last N by epoch
    for ckpt in checkpoints[-keep_last_n:]:
        keep_paths.add(ckpt["path"])

    deleted: list[str] = []
    for ckpt in checkpoints:
        if ckpt["path"] not in keep_paths:
            try:
                os.remove(ckpt["path"])
                deleted.append(ckpt["path"])
            except OSError:
                pass  # already gone or permission error — skip silently

    # Refresh latest.json if it pointed to a deleted file
    latest_path = Path(checkpoint_dir) / _LATEST_JSON
    if latest_path.is_file():
        try:
            with open(latest_path, "r", encoding="utf-8") as fh:
                info = json.load(fh)
            if info.get("path") in deleted:
                # Re-point to the highest-epoch surviving checkpoint
                surviving = [c for c in checkpoints if c["path"] not in deleted]
                if surviving:
                    newest = max(surviving, key=lambda c: c["epoch"])
                    with open(latest_path, "w", encoding="utf-8") as fh:
                        json.dump(
                            {
                                "path":     newest["path"],
                                "epoch":    newest["epoch"],
                                "val_loss": newest["val_loss"],
                            },
                            fh,
                            indent=2,
                        )
                else:
                    latest_path.unlink(missing_ok=True)
        except (json.JSONDecodeError, OSError):
            pass

    return deleted

"""
Video modality pipeline.

Strategy: sample N evenly-spaced frames → (C, T, H, W) tensor.
This matches the input format expected by R3D-18 / TinyR3D.

Expected data format:
    data_path/
        class_a/
            video1.mp4
            ...
        class_b/
            ...
"""
import random
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from config import DEFAULTS

VIDEO_EXTS  = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
# Kinetics-400 normalisation statistics (used by R3D-18 pretrained weights)
KINETICS_MEAN = np.array([0.43216, 0.394666, 0.37645],  dtype=np.float32).reshape(3, 1, 1, 1)
KINETICS_STD  = np.array([0.22803,  0.22145,  0.216989], dtype=np.float32).reshape(3, 1, 1, 1)


class VideoDataset(Dataset):
    def __init__(
        self,
        samples: list[tuple[str, int]],
        n_frames: int = 8,
        frame_size: int = 112,
    ):
        self.samples    = samples
        self.n_frames   = n_frames
        self.frame_size = frame_size

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        frames = self._load_frames(path)
        return frames, label

    def _load_frames(self, path: str) -> torch.Tensor:
        cap   = cv2.VideoCapture(path)
        total = max(1, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))

        if total <= self.n_frames:
            frame_idxs = list(range(total)) + [total - 1] * (self.n_frames - total)
        else:
            frame_idxs = [int(i * total / self.n_frames) for i in range(self.n_frames)]

        frames = []
        for fi in frame_idxs:
            cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
            ret, frame = cap.read()
            if not ret:
                frame = np.zeros((self.frame_size, self.frame_size, 3), dtype=np.uint8)
            frame = cv2.resize(frame, (self.frame_size, self.frame_size))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        cap.release()

        arr = np.stack(frames).transpose(3, 0, 1, 2).astype(np.float32) / 255.0
        arr = (arr - KINETICS_MEAN) / KINETICS_STD          # (C, T, H, W)
        return torch.tensor(arr)


def _filter_valid_video_samples(samples: list[tuple[str, int]]) -> list[tuple[str, int]]:
    valid: list[tuple[str, int]] = []
    for path, label in samples:
        cap = cv2.VideoCapture(path)
        ok, _frame = cap.read()
        cap.release()
        if ok:
            valid.append((path, label))
    return valid


def load_video_data(
    data_path: str,
    n_frames: int = 8,
    frame_size: int = 112,
    batch_size: int = 4,
    val_split: float = 0.2,
    verify_files: bool = False,
) -> tuple[DataLoader, DataLoader, list[str], dict]:
    data_path = Path(data_path)
    classes   = sorted(d.name for d in data_path.iterdir() if d.is_dir())
    if not classes:
        raise ValueError(f"No class subdirectories found in {data_path}")
    class_to_idx = {c: i for i, c in enumerate(classes)}

    samples = [
        (str(f), class_to_idx[cls])
        for cls in classes
        for f in (data_path / cls).iterdir()
        if f.suffix.lower() in VIDEO_EXTS
    ]
    if verify_files:
        samples = _filter_valid_video_samples(samples)
    if not samples:
        raise ValueError(f"No video files found in {data_path}")

    random.seed(42)
    random.shuffle(samples)
    n_val         = max(1, int(len(samples) * val_split))
    train_samples = samples[n_val:]
    val_samples   = samples[:n_val]

    # Video is I/O heavy (multiple frames per sample) — use workers but cap at 2
    # to avoid excessive memory use from large frame buffers per worker.
    nw = min(DEFAULTS["num_workers"], 2)
    pm = DEFAULTS["pin_memory"]
    ds_kwargs = dict(n_frames=n_frames, frame_size=frame_size)
    train_loader = DataLoader(
        VideoDataset(train_samples, **ds_kwargs),
        batch_size=batch_size, shuffle=True,
        num_workers=nw, pin_memory=pm,
        persistent_workers=(nw > 0),
        prefetch_factor=2 if nw > 0 else None,
    )
    val_loader = DataLoader(
        VideoDataset(val_samples, **ds_kwargs),
        batch_size=batch_size, shuffle=False,
        num_workers=nw, pin_memory=pm,
        persistent_workers=(nw > 0),
        prefetch_factor=2 if nw > 0 else None,
    )

    preprocessing_config = {
        "modality": "video",
        "n_frames": n_frames,
        "frame_size": frame_size,
        "mean": KINETICS_MEAN.flatten().tolist(),
        "std": KINETICS_STD.flatten().tolist(),
        "verify_files": verify_files,
    }

    return train_loader, val_loader, classes, preprocessing_config

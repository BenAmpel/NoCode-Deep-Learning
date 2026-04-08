"""
Image modality pipeline.

Expected data format:
    data_path/
        class_a/
            img1.jpg
            img2.png
            ...
        class_b/
            ...
"""
import random
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from config import DEFAULTS

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]
IMAGE_EXTS    = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}


class ImageDataset(Dataset):
    def __init__(self, samples: list[tuple[str, int]], transform=None):
        self.samples   = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label


def _filter_valid_image_samples(samples: list[tuple[str, int]]) -> list[tuple[str, int]]:
    valid: list[tuple[str, int]] = []
    for path, label in samples:
        try:
            with Image.open(path) as img:
                img.verify()
            valid.append((path, label))
        except Exception:
            continue
    return valid


def load_image_data(
    data_path: str,
    mode: str,
    batch_size: int = 16,
    val_split: float = 0.2,
    augmentation: str = "light",
    verify_files: bool = False,
    image_size: int = 224,
    augmentation_options: dict | None = None,
    normalization_preset: str = "imagenet",
    force_grayscale: bool = False,
) -> tuple[DataLoader, DataLoader, list[str], dict, list]:
    data_path = Path(data_path)
    classes   = sorted(d.name for d in data_path.iterdir() if d.is_dir())
    if not classes:
        raise ValueError(f"No class subdirectories found in {data_path}")
    class_to_idx = {c: i for i, c in enumerate(classes)}

    samples = [
        (str(img_path), class_to_idx[cls])
        for cls in classes
        for img_path in (data_path / cls).iterdir()
        if img_path.suffix.lower() in IMAGE_EXTS
    ]
    if verify_files:
        samples = _filter_valid_image_samples(samples)
    if not samples:
        raise ValueError(f"No images found in {data_path}")

    random.seed(42)
    random.shuffle(samples)
    n_val          = max(1, int(len(samples) * val_split))
    train_samples  = samples[n_val:]
    val_samples    = samples[:n_val]

    normalization_presets = {
        "imagenet": (IMAGENET_MEAN, IMAGENET_STD),
        "simple_0_1": ([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        "none": ([0.0, 0.0, 0.0], [1.0, 1.0, 1.0]),
    }
    mean, std = normalization_presets.get(normalization_preset, (IMAGENET_MEAN, IMAGENET_STD))

    # Use augmentation pipeline from data_pipeline/augmentation.py
    try:
        from data_pipeline.augmentation import get_image_transforms
        train_tf, val_tf = get_image_transforms(
            augmentation,
            image_size=image_size,
            options=augmentation_options,
            normalization={"mean": mean, "std": std},
            force_grayscale=force_grayscale,
        )
    except Exception:
        # Fallback: simple transforms if augmentation module unavailable
        train_tf = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.Grayscale(num_output_channels=3) if force_grayscale else transforms.Lambda(lambda x: x),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        val_tf = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.Grayscale(num_output_channels=3) if force_grayscale else transforms.Lambda(lambda x: x),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])

    nw = DEFAULTS["num_workers"]
    pm = DEFAULTS["pin_memory"]
    train_loader = DataLoader(
        ImageDataset(train_samples, train_tf),
        batch_size=batch_size, shuffle=True,
        num_workers=nw, pin_memory=pm,
        persistent_workers=(nw > 0),
        prefetch_factor=2 if nw > 0 else None,
    )
    val_loader = DataLoader(
        ImageDataset(val_samples, val_tf),
        batch_size=batch_size, shuffle=False,
        num_workers=nw, pin_memory=pm,
        persistent_workers=(nw > 0),
        prefetch_factor=2 if nw > 0 else None,
    )

    preprocessing_config = {
        "modality": "image",
        "resize": [image_size, image_size],
        "mean": mean,
        "std": std,
        "verify_files": verify_files,
        "augmentation_level": augmentation,
        "augmentation_options": augmentation_options or {},
        "normalization_preset": normalization_preset,
        "force_grayscale": force_grayscale,
    }

    return train_loader, val_loader, classes, preprocessing_config, val_samples

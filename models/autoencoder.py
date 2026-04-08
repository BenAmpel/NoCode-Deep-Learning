"""
Autoencoder models for unsupervised learning, clustering, and anomaly detection.
Supports image (2D conv) and tabular (MLP) inputs.
"""
from __future__ import annotations
import torch
import torch.nn as nn


class ImageAutoencoder(nn.Module):
    def __init__(self, latent_dim: int = 128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 4, stride=2, padding=1), nn.ReLU(),   # 112
            nn.Conv2d(32, 64, 4, stride=2, padding=1), nn.ReLU(),  # 56
            nn.Conv2d(64, 128, 4, stride=2, padding=1), nn.ReLU(), # 28
            nn.Conv2d(128, 256, 4, stride=2, padding=1), nn.ReLU(),# 14
            nn.AdaptiveAvgPool2d(4),                                # 4x4
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256 * 4 * 4),
            nn.Unflatten(1, (256, 4, 4)),
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1), nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),  nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),   nn.ReLU(),
            nn.ConvTranspose2d(32, 3, 4, stride=2, padding=1),
            nn.Upsample(size=(224, 224), mode="bilinear", align_corners=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))

    def get_features(self, x):
        return self.encoder(x)


class TabularAutoencoder(nn.Module):
    def __init__(self, input_size: int, latent_dim: int = 32):
        super().__init__()
        h = max(64, input_size)
        self.encoder = nn.Sequential(
            nn.Linear(input_size, h), nn.ReLU(),
            nn.Linear(h, h // 2),    nn.ReLU(),
            nn.Linear(h // 2, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, h // 2), nn.ReLU(),
            nn.Linear(h // 2, h),          nn.ReLU(),
            nn.Linear(h, input_size),
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))

    def get_features(self, x):
        return self.encoder(x)


def get_autoencoder(modality: str, input_size: int = None, latent_dim: int = 128):
    if modality in ("image", "audio"):
        return ImageAutoencoder(latent_dim=latent_dim)
    if modality == "tabular":
        if input_size is None:
            raise ValueError("input_size required for tabular autoencoder")
        return TabularAutoencoder(input_size=input_size, latent_dim=latent_dim)
    raise ValueError(f"Autoencoder not supported for modality: {modality}")

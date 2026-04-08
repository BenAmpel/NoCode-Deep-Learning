import torch.nn as nn


def get_video_model(model_name: str, num_classes: int, mode: str) -> nn.Module:
    if model_name == "R3D-18":
        from torchvision.models.video import r3d_18, R3D_18_Weights
        weights = R3D_18_Weights.DEFAULT if mode == "fine-tune" else None
        backbone = r3d_18(weights=weights)
        if mode == "fine-tune":
            for param in backbone.parameters():
                param.requires_grad = False
        return R3DWrapper(backbone, num_classes)

    if model_name == "MC3-18":
        from torchvision.models.video import mc3_18, MC3_18_Weights
        weights = MC3_18_Weights.DEFAULT if mode == "fine-tune" else None
        backbone = mc3_18(weights=weights)
        if mode == "fine-tune":
            for param in backbone.parameters():
                param.requires_grad = False
        return R3DWrapper(backbone, num_classes)

    if model_name == "R(2+1)D-18":
        from torchvision.models.video import r2plus1d_18, R2Plus1D_18_Weights
        weights = R2Plus1D_18_Weights.DEFAULT if mode == "fine-tune" else None
        backbone = r2plus1d_18(weights=weights)
        if mode == "fine-tune":
            for param in backbone.parameters():
                param.requires_grad = False
        return R3DWrapper(backbone, num_classes)

    if model_name == "TinyR3D":
        return TinyR3D(num_classes=num_classes)

    raise ValueError(f"Unknown video model: {model_name}")


class R3DWrapper(nn.Module):
    """
    Thin wrapper around torchvision R3D-18 that exposes get_features()
    for clustering support.
    """
    def __init__(self, backbone: nn.Module, num_classes: int):
        super().__init__()
        self.stem    = backbone.stem
        self.layer1  = backbone.layer1
        self.layer2  = backbone.layer2
        self.layer3  = backbone.layer3
        self.layer4  = backbone.layer4
        self.avgpool = backbone.avgpool
        self.head    = nn.Linear(backbone.fc.in_features, num_classes)

    def _embed(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        return x.flatten(1)

    def forward(self, x):
        return self.head(self._embed(x))

    def get_features(self, x):
        return self._embed(x)


class TinyR3D(nn.Module):
    """Lightweight 3D CNN — designed for CPU with small batches."""
    def __init__(self, num_classes: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv3d(3, 16, kernel_size=3, padding=1),
            nn.BatchNorm3d(16),
            nn.ReLU(),
            nn.MaxPool3d((1, 2, 2)),
            nn.Conv3d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.AdaptiveAvgPool3d((1, 4, 4)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 4 * 4, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))

    def get_features(self, x):
        return self.features(x).flatten(1)

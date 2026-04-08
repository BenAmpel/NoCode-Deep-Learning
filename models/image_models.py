import torch.nn as nn
from torchvision import models


def get_image_model(model_name: str, num_classes: int, mode: str, dropout: float = 0.3) -> nn.Module:
    if model_name == "MobileNetV3-Small":
        weights = models.MobileNet_V3_Small_Weights.DEFAULT if mode == "fine-tune" else None
        backbone = models.mobilenet_v3_small(weights=weights)
        if mode == "fine-tune":
            for param in backbone.features.parameters():
                param.requires_grad = False
        return MobileNetV3Wrapper(backbone, num_classes)

    if model_name == "MobileNetV3-Large":
        weights = models.MobileNet_V3_Large_Weights.DEFAULT if mode == "fine-tune" else None
        backbone = models.mobilenet_v3_large(weights=weights)
        if mode == "fine-tune":
            for param in backbone.features.parameters():
                param.requires_grad = False
        return MobileNetV3Wrapper(backbone, num_classes)

    if model_name == "EfficientNet-B0":
        weights = models.EfficientNet_B0_Weights.DEFAULT if mode == "fine-tune" else None
        backbone = models.efficientnet_b0(weights=weights)
        if mode == "fine-tune":
            for param in backbone.features.parameters():
                param.requires_grad = False
        return EfficientNetWrapper(backbone, num_classes)

    if model_name == "EfficientNet-B2":
        weights = models.EfficientNet_B2_Weights.DEFAULT if mode == "fine-tune" else None
        backbone = models.efficientnet_b2(weights=weights)
        if mode == "fine-tune":
            for param in backbone.features.parameters():
                param.requires_grad = False
        return EfficientNetWrapper(backbone, num_classes)

    if model_name == "ResNet18":
        weights = models.ResNet18_Weights.DEFAULT if mode == "fine-tune" else None
        backbone = models.resnet18(weights=weights)
        if mode == "fine-tune":
            for param in backbone.parameters():
                param.requires_grad = False
        return ResNetWrapper(backbone, num_classes)

    if model_name == "ResNet50":
        weights = models.ResNet50_Weights.DEFAULT if mode == "fine-tune" else None
        backbone = models.resnet50(weights=weights)
        if mode == "fine-tune":
            for param in backbone.parameters():
                param.requires_grad = False
        return ResNetWrapper(backbone, num_classes)

    if model_name == "DenseNet121":
        weights = models.DenseNet121_Weights.DEFAULT if mode == "fine-tune" else None
        backbone = models.densenet121(weights=weights)
        if mode == "fine-tune":
            for param in backbone.features.parameters():
                param.requires_grad = False
        return DenseNetWrapper(backbone, num_classes)

    if model_name == "ConvNeXt-Tiny":
        weights = models.ConvNeXt_Tiny_Weights.DEFAULT if mode == "fine-tune" else None
        backbone = models.convnext_tiny(weights=weights)
        if mode == "fine-tune":
            for param in backbone.features.parameters():
                param.requires_grad = False
        return ConvNeXtWrapper(backbone, num_classes)

    if model_name == "TinyCNN":
        return TinyCNN(num_classes=num_classes, dropout=dropout)

    raise ValueError(f"Unknown image model: {model_name}")


class MobileNetV3Wrapper(nn.Module):
    """
    Thin wrapper around torchvision MobileNetV3 that exposes get_features()
    for clustering support, using the penultimate representation.
    """
    def __init__(self, backbone: nn.Module, num_classes: int):
        super().__init__()
        self.features  = backbone.features
        self.avgpool   = backbone.avgpool
        # Keep everything up to (but not including) the final classification Linear
        self.pre_head  = backbone.classifier[:-1]
        self.head      = nn.Linear(backbone.classifier[-1].in_features, num_classes)

    def _embed(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.flatten(1)
        return self.pre_head(x)

    def forward(self, x):
        return self.head(self._embed(x))

    def get_features(self, x):
        return self._embed(x)


class EfficientNetWrapper(nn.Module):
    """
    Thin wrapper around torchvision EfficientNet-B0 that exposes get_features()
    for clustering support, using the penultimate representation.
    """
    def __init__(self, backbone: nn.Module, num_classes: int):
        super().__init__()
        self.features = backbone.features
        self.avgpool  = backbone.avgpool
        # classifier is [Dropout, Linear] — keep Dropout, replace Linear
        self.pre_head = backbone.classifier[:-1]
        self.head     = nn.Linear(backbone.classifier[-1].in_features, num_classes)

    def _embed(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.flatten(1)
        return self.pre_head(x)

    def forward(self, x):
        return self.head(self._embed(x))

    def get_features(self, x):
        return self._embed(x)


class ResNetWrapper(nn.Module):
    """Thin wrapper around torchvision ResNet backbones."""
    def __init__(self, backbone: nn.Module, num_classes: int):
        super().__init__()
        self.stem = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
        )
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4
        self.avgpool = backbone.avgpool
        self.head = nn.Linear(backbone.fc.in_features, num_classes)

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


class DenseNetWrapper(nn.Module):
    """Thin wrapper around torchvision DenseNet backbones."""
    def __init__(self, backbone: nn.Module, num_classes: int):
        super().__init__()
        self.features = backbone.features
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.head = nn.Linear(backbone.classifier.in_features, num_classes)

    def _embed(self, x):
        x = self.features(x)
        x = nn.functional.relu(x, inplace=False)
        x = self.pool(x)
        return x.flatten(1)

    def forward(self, x):
        return self.head(self._embed(x))

    def get_features(self, x):
        return self._embed(x)


class ConvNeXtWrapper(nn.Module):
    """Thin wrapper around torchvision ConvNeXt backbones."""
    def __init__(self, backbone: nn.Module, num_classes: int):
        super().__init__()
        self.features = backbone.features
        self.avgpool = backbone.avgpool
        self.pre_head = backbone.classifier[:-1]
        self.head = nn.Linear(backbone.classifier[-1].in_features, num_classes)

    def _embed(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.flatten(1)
        return self.pre_head(x)

    def forward(self, x):
        return self.head(self._embed(x))

    def get_features(self, x):
        return self._embed(x)


class TinyCNN(nn.Module):
    def __init__(self, num_classes: int, dropout: float = 0.3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.AdaptiveAvgPool2d(4),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))

    def get_features(self, x):
        return self.features(x).flatten(1)

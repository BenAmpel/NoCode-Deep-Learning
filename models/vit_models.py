"""
Vision Transformer (ViT) image classification models backed by the `timm` library.

Supported model names (user-facing → timm identifier):
    "ViT-Tiny"  → vit_tiny_patch16_224
    "ViT-Small" → vit_small_patch16_224

All ViT variants require 224×224 RGB input images (VIT_INPUT_SIZE = 224).

Fine-tune mode: only the classification head (model.head) is trainable;
                the transformer backbone is frozen.
From-scratch  : all parameters are trainable.
"""
from __future__ import annotations

import torch
import torch.nn as nn

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

VIT_INPUT_SIZE: int = 224  # ViT patch embeddings are tied to 224×224 images

# Maps human-readable names (shown in the UI) to timm model identifiers
_VIT_NAME_MAP: dict[str, str] = {
    "ViT-Tiny":  "vit_tiny_patch16_224",
    "ViT-Small": "vit_small_patch16_224",
}


# ─────────────────────────────────────────────────────────────────────────────
# ViTWrapper
# ─────────────────────────────────────────────────────────────────────────────

class ViTWrapper(nn.Module):
    """
    Thin wrapper around a timm ViT backbone that adds a `get_features()` method
    for clustering and embedding workflows.

    Parameters
    ----------
    model_name : str
        A timm model identifier, e.g. ``"vit_tiny_patch16_224"``.
    num_classes : int
        Number of output classes for the classification head.
    pretrained : bool
        Whether to initialise with ImageNet-pretrained weights.

    Attributes
    ----------
    feature_dim : int
        Dimensionality of the penultimate (CLS-token / pooled) representation,
        taken directly from the timm model's ``num_features`` attribute.
    """

    def __init__(
        self,
        model_name: str,
        num_classes: int,
        pretrained: bool = True,
    ) -> None:
        super().__init__()
        try:
            import timm  # noqa: PLC0415
        except ImportError as exc:
            raise ImportError(
                "The `timm` library is required for ViT models.\n"
                "Install it with:  pip install timm"
            ) from exc

        self._model = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=num_classes,
        )
        # Cache the feature dimensionality so callers never need to inspect internals
        self.feature_dim: int = self._model.num_features

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Standard forward pass returning class logits of shape (B, num_classes)."""
        return self._model(x)

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Return penultimate-layer features of shape (B, feature_dim).

        Implementation: temporarily replaces the model head with an identity
        function, runs the forward pass, then restores the original head.
        This is safe for inference; do NOT call during a training step with
        gradients flowing through the head.
        """
        original_head = self._model.head
        self._model.head = nn.Identity()
        try:
            features = self._model(x)
        finally:
            self._model.head = original_head
        return features


# ─────────────────────────────────────────────────────────────────────────────
# Factory
# ─────────────────────────────────────────────────────────────────────────────

def get_vit_model(
    model_name: str,
    num_classes: int,
    pretrained: bool = True,
) -> ViTWrapper:
    """
    Build and return a :class:`ViTWrapper` for the requested model.

    Parameters
    ----------
    model_name : str
        Human-readable model name.  Supported values:

        * ``"ViT-Tiny"``  — vit_tiny_patch16_224 (~5.7 M params)
        * ``"ViT-Small"`` — vit_small_patch16_224 (~22 M params)

    num_classes : int
        Number of output classes.
    pretrained : bool
        * ``True``  (fine-tune mode): loads ImageNet weights and freezes
          all layers **except** the classification head (``model.head``).
        * ``False`` (from-scratch mode): random initialisation, all layers
          are trainable.

    Returns
    -------
    ViTWrapper

    Raises
    ------
    ImportError
        If the ``timm`` package is not installed.
    ValueError
        If ``model_name`` is not one of the supported identifiers.
    """
    try:
        import timm  # noqa: F401, PLC0415
    except ImportError as exc:
        raise ImportError(
            "The `timm` library is required for ViT models.\n"
            "Install it with:  pip install timm"
        ) from exc

    if model_name not in _VIT_NAME_MAP:
        supported = ", ".join(f'"{k}"' for k in _VIT_NAME_MAP)
        raise ValueError(
            f"Unknown ViT model '{model_name}'. "
            f"Supported names: {supported}"
        )

    timm_name = _VIT_NAME_MAP[model_name]
    wrapper = ViTWrapper(
        model_name=timm_name,
        num_classes=num_classes,
        pretrained=pretrained,
    )

    if pretrained:
        # Fine-tune mode: freeze every parameter except the head
        for param in wrapper._model.parameters():
            param.requires_grad = False
        # Unfreeze only the classification head
        for param in wrapper._model.head.parameters():
            param.requires_grad = True
    # else: from-scratch — all parameters are already trainable by default

    return wrapper

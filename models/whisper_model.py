"""
Whisper-based audio classification models backed by HuggingFace `transformers`.

Architecture overview
---------------------
The Whisper encoder (a convolutional stem + transformer) processes an 80-band
mel spectrogram and produces a sequence of hidden-state vectors.  We apply
global average pooling over the time axis to obtain a fixed-size embedding,
then project to ``num_classes`` via a linear head.

Constants
---------
WHISPER_SAMPLE_RATE : int = 16000   — required audio sample rate (Hz)
WHISPER_N_MELS      : int = 80      — number of mel filterbank channels

Supported model sizes : "tiny", "base"
    Model ID template  : ``openai/whisper-{model_size}``
"""
from __future__ import annotations

import torch
import torch.nn as nn

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

WHISPER_SAMPLE_RATE: int = 16_000  # Whisper is trained on 16 kHz audio
WHISPER_N_MELS: int = 80           # Standard Whisper mel filterbank dimension

# Supported size tags → HuggingFace repo IDs
_WHISPER_SIZES: dict[str, str] = {
    "tiny": "openai/whisper-tiny",
    "base": "openai/whisper-base",
}


# ─────────────────────────────────────────────────────────────────────────────
# WhisperClassifier
# ─────────────────────────────────────────────────────────────────────────────

class WhisperClassifier(nn.Module):
    """
    Audio classifier built on the Whisper encoder.

    Only the encoder half of the Whisper seq2seq model is used; the decoder
    is discarded.  A global-average-pooling + linear head produces class logits.

    Parameters
    ----------
    num_classes : int
        Number of output classes.
    model_size : str
        Whisper variant — ``"tiny"`` or ``"base"``.
    freeze_encoder : bool
        If ``True`` (default), all encoder parameters are frozen so that only
        the classification head is updated during training.  Set to ``False``
        for full fine-tuning.

    Attributes
    ----------
    hidden_size : int
        Dimensionality of the encoder's hidden states (used as the input width
        of the classification head).
    """

    def __init__(
        self,
        num_classes: int,
        model_size: str = "tiny",
        freeze_encoder: bool = True,
    ) -> None:
        super().__init__()

        try:
            from transformers import WhisperModel  # noqa: PLC0415
        except ImportError as exc:
            raise ImportError(
                "The `transformers` library is required for Whisper models.\n"
                "Install it with:  pip install transformers"
            ) from exc

        if model_size not in _WHISPER_SIZES:
            supported = ", ".join(f'"{s}"' for s in _WHISPER_SIZES)
            raise ValueError(
                f"Unsupported Whisper model size '{model_size}'. "
                f"Supported: {supported}"
            )

        repo_id = _WHISPER_SIZES[model_size]

        # Load the full Whisper model and extract its encoder only
        full_model = WhisperModel.from_pretrained(repo_id)
        self.encoder = full_model.encoder

        # Resolve encoder hidden size from the model config
        self.hidden_size: int = self.encoder.config.d_model

        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

        # Classification head: global-avg-pool (applied in forward) + linear
        self.classifier = nn.Linear(self.hidden_size, num_classes)

    # ------------------------------------------------------------------
    # Internal helper
    # ------------------------------------------------------------------

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Run the Whisper encoder on a mel spectrogram batch and return the
        globally-pooled hidden states.

        Parameters
        ----------
        x : torch.Tensor
            Shape ``(B, 80, T)`` — batch of mel spectrograms.

        Returns
        -------
        torch.Tensor
            Shape ``(B, hidden_size)`` — pooled encoder output.
        """
        # WhisperEncoder expects (B, n_mels, T) — the same shape we receive
        encoder_outputs = self.encoder(input_features=x)
        # last_hidden_state: (B, T', hidden_size)
        hidden = encoder_outputs.last_hidden_state
        # Global average pooling over the time dimension
        pooled = hidden.mean(dim=1)  # (B, hidden_size)
        return pooled

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Classify a batch of mel spectrograms.

        Parameters
        ----------
        x : torch.Tensor
            Shape ``(B, 80, T)``.

        Returns
        -------
        torch.Tensor
            Class logits of shape ``(B, num_classes)``.
        """
        pooled = self._encode(x)
        return self.classifier(pooled)

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Return pooled encoder representations before the classification head.

        Parameters
        ----------
        x : torch.Tensor
            Shape ``(B, 80, T)``.

        Returns
        -------
        torch.Tensor
            Shape ``(B, hidden_size)`` — suitable for clustering or embedding
            visualisation workflows.
        """
        return self._encode(x)


# ─────────────────────────────────────────────────────────────────────────────
# Factory
# ─────────────────────────────────────────────────────────────────────────────

def get_whisper_model(
    num_classes: int,
    model_size: str = "tiny",
    freeze_encoder: bool = True,
) -> WhisperClassifier:
    """
    Build and return a :class:`WhisperClassifier`.

    Parameters
    ----------
    num_classes : int
        Number of output classes.
    model_size : str
        Whisper encoder size.  Supported values:

        * ``"tiny"`` — ~37 M params (fastest, smallest download)
        * ``"base"`` — ~74 M params (better accuracy)

    freeze_encoder : bool
        * ``True`` (default): encoder weights are frozen; only the linear head
          is updated.  Recommended when labelled data is limited.
        * ``False``: full encoder + head fine-tuning.  Requires more data and
          a lower learning rate.

    Returns
    -------
    WhisperClassifier

    Raises
    ------
    ImportError
        If the ``transformers`` package is not installed.
    ValueError
        If ``model_size`` is not one of the supported values.
    """
    try:
        import transformers  # noqa: F401, PLC0415
    except ImportError as exc:
        raise ImportError(
            "The `transformers` library is required for Whisper models.\n"
            "Install it with:  pip install transformers"
        ) from exc

    return WhisperClassifier(
        num_classes=num_classes,
        model_size=model_size,
        freeze_encoder=freeze_encoder,
    )

"""
Text classification model architectures.

From-scratch models: RNN, LSTM, BiLSTM, CNN-LSTM, Transformer
Fine-tune models:    DistilBERT, BERT, RoBERTa (pretrained HuggingFace)
"""
import math

import torch
import torch.nn as nn

# ── Pretrained transformer name mapping ──────────────────────────────────────

_TRANSFORMER_NAME_MAP = {
    "DistilBERT": "distilbert-base-uncased",
    "BERT": "bert-base-uncased",
    "RoBERTa": "roberta-base",
}

# ── From-scratch model name mapping ──────────────────────────────────────────

_SCRATCH_MODELS = {"RNN", "LSTM", "BiLSTM", "CNN-LSTM", "Transformer"}


# ── Factory ──────────────────────────────────────────────────────────────────

def get_text_model(
    model_name: str,
    num_classes: int,
    mode: str,
    vocab_size: int = 30522,
    hidden_size: int = 128,
    num_layers: int = 1,
    dropout: float = 0.3,
) -> nn.Module:
    if model_name in _TRANSFORMER_NAME_MAP and mode == "fine-tune":
        return PretrainedTransformerClassifier(
            num_classes=num_classes,
            repo_id=_TRANSFORMER_NAME_MAP[model_name],
        )

    kwargs = dict(
        vocab_size=vocab_size,
        num_classes=num_classes,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
    )

    if model_name == "RNN":
        return SimpleRNNClassifier(**kwargs)
    if model_name == "LSTM":
        return LSTMClassifier(**kwargs)
    if model_name == "BiLSTM":
        return BiLSTMClassifier(**kwargs)
    if model_name == "CNN-LSTM":
        return CNNLSTMClassifier(**kwargs)
    if model_name == "Transformer":
        return ScratchTransformerClassifier(**kwargs)

    # Legacy fallback
    if model_name == "GRU":
        return LSTMClassifier(**kwargs)

    raise ValueError(f"Unknown text model: {model_name} (mode={mode})")


# ═══════════════════════════════════════════════════════════════════════════════
# Pretrained transformer (fine-tune only)
# ═══════════════════════════════════════════════════════════════════════════════

class PretrainedTransformerClassifier(nn.Module):
    """Fine-tune a pretrained HuggingFace transformer."""

    def __init__(self, num_classes: int, repo_id: str):
        super().__init__()
        from transformers import AutoModel
        self.encoder = AutoModel.from_pretrained(repo_id)
        self.hidden_size = self.encoder.config.hidden_size

        # Freeze embeddings
        if hasattr(self.encoder, "embeddings"):
            for param in self.encoder.embeddings.parameters():
                param.requires_grad = False

        # Freeze all but last 2 transformer layers
        layers = None
        if hasattr(self.encoder, "transformer") and hasattr(self.encoder.transformer, "layer"):
            layers = self.encoder.transformer.layer
        elif hasattr(self.encoder, "encoder") and hasattr(self.encoder.encoder, "layer"):
            layers = self.encoder.encoder.layer
        if layers is not None and len(layers) > 2:
            for layer in layers[:-2]:
                for param in layer.parameters():
                    param.requires_grad = False

        self.classifier = nn.Sequential(
            nn.Linear(self.hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, input_ids, attention_mask):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls = out.last_hidden_state[:, 0, :]
        return self.classifier(cls)

    def get_features(self, input_ids, attention_mask):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        return out.last_hidden_state[:, 0, :]


# ═══════════════════════════════════════════════════════════════════════════════
# Shared base for from-scratch models
# ═══════════════════════════════════════════════════════════════════════════════

_EMBED_DIM = 128


class _TextClassifierBase(nn.Module):
    """Shared embedding layer, dropout, and classifier head.

    Subclasses must set ``self.encoder_dim`` and implement ``_encode()``.
    """

    def __init__(self, vocab_size: int, num_classes: int, dropout: float = 0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, _EMBED_DIM, padding_idx=0)
        self.drop = nn.Dropout(dropout)
        # Subclass __init__ must call self._build_head(num_classes) after
        # setting self.encoder_dim.

    def _build_head(self, num_classes: int):
        self.classifier = nn.Linear(self.encoder_dim, num_classes)

    def _encode(self, x: torch.Tensor, attention_mask: torch.Tensor | None) -> torch.Tensor:
        raise NotImplementedError

    def forward(self, input_ids, attention_mask=None):
        x = self.embedding(input_ids)
        features = self._encode(x, attention_mask)
        return self.classifier(self.drop(features))

    def get_features(self, input_ids, attention_mask=None):
        x = self.embedding(input_ids)
        return self._encode(x, attention_mask)


# ── Pooling helpers ──────────────────────────────────────────────────────────

def _last_token_pool(rnn_out: torch.Tensor, attention_mask: torch.Tensor | None) -> torch.Tensor:
    """Get hidden state at the last non-padding position (unidirectional)."""
    if attention_mask is None:
        return rnn_out[:, -1, :]
    lengths = attention_mask.sum(dim=1).clamp(min=1).long()
    idx = (lengths - 1).unsqueeze(1).unsqueeze(2).expand(-1, 1, rnn_out.size(2))
    return rnn_out.gather(1, idx).squeeze(1)


def _bidir_pool(rnn_out: torch.Tensor, attention_mask: torch.Tensor | None) -> torch.Tensor:
    """Pool bidirectional RNN: forward last-real-token + backward first-token."""
    hidden_size = rnn_out.size(2) // 2
    fwd = rnn_out[:, :, :hidden_size]
    bwd = rnn_out[:, :, hidden_size:]

    if attention_mask is None:
        return torch.cat([fwd[:, -1, :], bwd[:, 0, :]], dim=1)

    lengths = attention_mask.sum(dim=1).clamp(min=1).long()
    idx = (lengths - 1).unsqueeze(1).unsqueeze(2).expand(-1, 1, hidden_size)
    fwd_pooled = fwd.gather(1, idx).squeeze(1)
    bwd_pooled = bwd[:, 0, :]
    return torch.cat([fwd_pooled, bwd_pooled], dim=1)


def _masked_mean_pool(hidden: torch.Tensor, attention_mask: torch.Tensor | None) -> torch.Tensor:
    """Mean-pool over non-padding positions."""
    if attention_mask is None:
        return hidden.mean(dim=1)
    mask = attention_mask.unsqueeze(2).float()  # (B, T, 1)
    return (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)


# ═══════════════════════════════════════════════════════════════════════════════
# 1. Simple RNN
# ═══════════════════════════════════════════════════════════════════════════════

class SimpleRNNClassifier(_TextClassifierBase):
    """Vanilla RNN (Elman network) — fast baseline, limited context."""

    def __init__(self, vocab_size, num_classes, hidden_size=128,
                 num_layers=1, dropout=0.3):
        super().__init__(vocab_size, num_classes, dropout)
        self.rnn = nn.RNN(
            input_size=_EMBED_DIM,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.encoder_dim = hidden_size
        self._build_head(num_classes)

    def _encode(self, x, attention_mask):
        out, _ = self.rnn(x)
        return _last_token_pool(out, attention_mask)


# ═══════════════════════════════════════════════════════════════════════════════
# 2. LSTM (unidirectional)
# ═══════════════════════════════════════════════════════════════════════════════

class LSTMClassifier(_TextClassifierBase):
    """Unidirectional LSTM — better gradient flow than RNN."""

    def __init__(self, vocab_size, num_classes, hidden_size=128,
                 num_layers=1, dropout=0.3):
        super().__init__(vocab_size, num_classes, dropout)
        self.rnn = nn.LSTM(
            input_size=_EMBED_DIM,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.encoder_dim = hidden_size
        self._build_head(num_classes)

    def _encode(self, x, attention_mask):
        out, _ = self.rnn(x)
        return _last_token_pool(out, attention_mask)


# ═══════════════════════════════════════════════════════════════════════════════
# 3. BiLSTM (bidirectional)
# ═══════════════════════════════════════════════════════════════════════════════

class BiLSTMClassifier(_TextClassifierBase):
    """Bidirectional LSTM — sees context from both directions."""

    def __init__(self, vocab_size, num_classes, hidden_size=128,
                 num_layers=1, dropout=0.3):
        super().__init__(vocab_size, num_classes, dropout)
        self.rnn = nn.LSTM(
            input_size=_EMBED_DIM,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=True,
        )
        self.encoder_dim = hidden_size * 2
        self._build_head(num_classes)

    def _encode(self, x, attention_mask):
        out, _ = self.rnn(x)
        return _bidir_pool(out, attention_mask)


# ═══════════════════════════════════════════════════════════════════════════════
# 4. CNN-LSTM
# ═══════════════════════════════════════════════════════════════════════════════

class CNNLSTMClassifier(_TextClassifierBase):
    """CNN for local n-gram features → LSTM for sequential context."""

    def __init__(self, vocab_size, num_classes, hidden_size=128,
                 num_layers=1, dropout=0.3):
        super().__init__(vocab_size, num_classes, dropout)
        # Multi-kernel CNN: filter sizes 3, 4, 5
        self.convs = nn.ModuleList([
            nn.Conv1d(_EMBED_DIM, 64, kernel_size=k, padding=k // 2)
            for k in (3, 4, 5)
        ])
        self.conv_drop = nn.Dropout(dropout)
        # LSTM operates on concatenated CNN channels
        self.rnn = nn.LSTM(
            input_size=64 * 3,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.encoder_dim = hidden_size
        self._build_head(num_classes)

    def _encode(self, x, attention_mask):
        # x: (B, T, embed_dim) → Conv1d wants (B, C, T)
        c = x.transpose(1, 2)
        conv_outs = [torch.relu(conv(c)).transpose(1, 2) for conv in self.convs]
        # Truncate to shortest length (different kernels may produce different lengths)
        min_len = min(o.size(1) for o in conv_outs)
        conv_cat = torch.cat([o[:, :min_len, :] for o in conv_outs], dim=2)  # (B, T, 192)
        conv_cat = self.conv_drop(conv_cat)
        # Trim attention_mask to match conv output length
        if attention_mask is not None and attention_mask.size(1) > min_len:
            attention_mask = attention_mask[:, :min_len]
        out, _ = self.rnn(conv_cat)
        return _last_token_pool(out, attention_mask)


# ═══════════════════════════════════════════════════════════════════════════════
# 5. Scratch Transformer
# ═══════════════════════════════════════════════════════════════════════════════

class _PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding."""

    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.drop = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(x + self.pe[:, :x.size(1)])


class ScratchTransformerClassifier(_TextClassifierBase):
    """Small transformer encoder trained from scratch — no pretrained weights."""

    def __init__(self, vocab_size, num_classes, hidden_size=128,
                 num_layers=2, dropout=0.3):
        super().__init__(vocab_size, num_classes, dropout)
        nhead = max(1, hidden_size // 32)  # 4 heads for 128-dim
        # Project embedding dim to hidden_size if they differ
        self.proj = nn.Linear(_EMBED_DIM, hidden_size) if _EMBED_DIM != hidden_size else nn.Identity()
        self.pos_enc = _PositionalEncoding(hidden_size, dropout=dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=nhead,
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.encoder_dim = hidden_size
        self._build_head(num_classes)

    def _encode(self, x, attention_mask):
        x = self.proj(x)
        x = self.pos_enc(x)
        # TransformerEncoder expects src_key_padding_mask: True = ignore
        pad_mask = (~attention_mask.bool()) if attention_mask is not None else None
        out = self.transformer(x, src_key_padding_mask=pad_mask)
        return _masked_mean_pool(out, attention_mask)

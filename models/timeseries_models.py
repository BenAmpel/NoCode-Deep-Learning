import math

import torch
import torch.nn as nn


def get_timeseries_model(
    model_name: str,
    num_classes: int,
    input_size: int,
    hidden_size: int = 64,
    num_layers: int = 1,
    dropout: float = 0.3,
) -> nn.Module:
    if model_name in ("LSTM", "GRU"):
        return TimeSeriesRNN(
            model_type=model_name,
            input_size=input_size,
            num_classes=num_classes,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
        )
    if model_name == "MLP-Window":
        return TimeSeriesWindowMLP(
            input_size=input_size,
            num_classes=num_classes,
            hidden_size=hidden_size,
            dropout=dropout,
        )
    if model_name == "CNN1D":
        return TimeSeriesCNN(
            input_size=input_size,
            num_classes=num_classes,
            hidden_size=hidden_size,
            dropout=dropout,
        )
    if model_name == "TCN":
        return TimeSeriesTCN(
            input_size=input_size,
            num_classes=num_classes,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
        )
    if model_name == "Transformer-Tiny":
        return TimeSeriesTransformer(
            input_size=input_size,
            num_classes=num_classes,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
        )
    if model_name == "Transformer-Small":
        return TimeSeriesTransformer(
            input_size=input_size,
            num_classes=num_classes,
            hidden_size=max(hidden_size, 128),
            num_layers=max(num_layers, 3),
            dropout=dropout,
        )
    raise ValueError(f"Unknown time-series model: {model_name}")


class TimeSeriesRNN(nn.Module):
    def __init__(
        self,
        model_type: str,
        input_size: int,
        num_classes: int,
        hidden_size: int = 64,
        num_layers: int = 1,
        dropout: float = 0.3,
    ):
        super().__init__()
        rnn_cls = nn.LSTM if model_type == "LSTM" else nn.GRU
        self.rnn = rnn_cls(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out, _ = self.rnn(x)
        return self.classifier(out[:, -1, :])

    def get_features(self, x):
        out, _ = self.rnn(x)
        return out[:, -1, :]


class TimeSeriesWindowMLP(nn.Module):
    def __init__(
        self,
        input_size: int,
        num_classes: int,
        hidden_size: int = 64,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.proj = nn.LazyLinear(hidden_size)
        self.norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_classes)

    def get_features(self, x):
        x = x.flatten(start_dim=1)
        x = torch.relu(self.proj(x))
        x = self.norm(x)
        x = self.dropout(x)
        return x

    def forward(self, x):
        return self.classifier(self.get_features(x))


class TimeSeriesCNN(nn.Module):
    def __init__(
        self,
        input_size: int,
        num_classes: int,
        hidden_size: int = 64,
        dropout: float = 0.3,
    ):
        super().__init__()
        mid = max(32, hidden_size // 2)
        self.encoder = nn.Sequential(
            nn.Conv1d(input_size, mid, kernel_size=5, padding=2),
            nn.BatchNorm1d(mid),
            nn.ReLU(),
            nn.Conv1d(mid, hidden_size, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_classes)

    def get_features(self, x):
        x = x.transpose(1, 2)
        feats = self.encoder(x).squeeze(-1)
        return self.dropout(feats)

    def forward(self, x):
        return self.classifier(self.get_features(x))


class _TemporalBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dilation: int, dropout: float):
        super().__init__()
        padding = dilation
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=padding, dilation=dilation)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=padding, dilation=dilation)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.residual = nn.Conv1d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        out = self.conv1(x)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.conv2(out)
        out = self.relu(out)
        out = self.dropout(out)
        return out + self.residual(x)


class TimeSeriesTCN(nn.Module):
    def __init__(
        self,
        input_size: int,
        num_classes: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()
        layers = []
        in_ch = input_size
        for layer_idx in range(max(1, num_layers)):
            dilation = 2 ** layer_idx
            layers.append(_TemporalBlock(in_ch, hidden_size, dilation=dilation, dropout=dropout))
            in_ch = hidden_size
        self.network = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_classes)

    def get_features(self, x):
        x = x.transpose(1, 2)
        feats = self.network(x)
        feats = self.pool(feats).squeeze(-1)
        return self.dropout(feats)

    def forward(self, x):
        return self.classifier(self.get_features(x))


class _PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 2048):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x):
        return x + self.pe[:, : x.size(1)]


class TimeSeriesTransformer(nn.Module):
    def __init__(
        self,
        input_size: int,
        num_classes: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()
        nhead = 4 if hidden_size >= 64 else 2
        while hidden_size % nhead != 0 and nhead > 1:
            nhead -= 1
        self.input_proj = nn.Linear(input_size, hidden_size)
        self.positional = _PositionalEncoding(hidden_size)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=max(1, nhead),
            dim_feedforward=hidden_size * 2,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=max(1, num_layers))
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_classes)

    def get_features(self, x):
        x = self.input_proj(x)
        x = self.positional(x)
        feats = self.encoder(x).mean(dim=1)
        return self.dropout(feats)

    def forward(self, x):
        return self.classifier(self.get_features(x))

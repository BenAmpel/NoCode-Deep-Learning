from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn


def get_graph_model(
    model_name: str,
    num_classes: int,
    input_size: int,
    hidden_size: int = 64,
    num_layers: int = 2,
    dropout: float = 0.3,
) -> nn.Module:
    if model_name == "GCN":
        return GCN(input_size, hidden_size, num_classes, num_layers=max(2, num_layers), dropout=dropout)
    if model_name == "GraphSAGE":
        return GraphSAGE(input_size, hidden_size, num_classes, num_layers=max(2, num_layers), dropout=dropout)
    if model_name == "GraphTransformer":
        return GraphTransformer(input_size, hidden_size=max(hidden_size, 64), num_classes=num_classes, num_layers=max(2, num_layers), dropout=dropout)
    raise ValueError(f"Unknown graph model: {model_name}")


class GCNLayer(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        return self.linear(adj @ x)


class GCN(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_classes: int, num_layers: int = 2, dropout: float = 0.3):
        super().__init__()
        layers = []
        in_dim = input_size
        for _ in range(max(1, num_layers - 1)):
            layers.append(GCNLayer(in_dim, hidden_size))
            in_dim = hidden_size
        self.layers = nn.ModuleList(layers)
        self.dropout = nn.Dropout(dropout)
        self.classifier = GCNLayer(in_dim, num_classes)

    def get_embeddings(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = torch.relu(layer(x, adj))
            x = self.dropout(x)
        return x

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        emb = self.get_embeddings(x, adj)
        return self.classifier(emb, adj)


class SageLayer(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.linear = nn.Linear(in_features * 2, out_features)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        neigh = adj @ x
        return self.linear(torch.cat([x, neigh], dim=-1))


class GraphSAGE(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_classes: int, num_layers: int = 2, dropout: float = 0.3):
        super().__init__()
        layers = []
        in_dim = input_size
        for _ in range(max(1, num_layers - 1)):
            layers.append(SageLayer(in_dim, hidden_size))
            in_dim = hidden_size
        self.layers = nn.ModuleList(layers)
        self.dropout = nn.Dropout(dropout)
        self.classifier = SageLayer(in_dim, num_classes)

    def get_embeddings(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = torch.relu(layer(x, adj))
            x = self.dropout(x)
        return x

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        emb = self.get_embeddings(x, adj)
        return self.classifier(emb, adj)


class PositionalProjection(nn.Module):
    def __init__(self, d_model: int, max_nodes: int = 8192):
        super().__init__()
        pe = torch.zeros(max_nodes, d_model)
        position = torch.arange(0, max_nodes, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[: x.size(0)]


class GraphTransformer(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_classes: int, num_layers: int = 2, dropout: float = 0.3):
        super().__init__()
        heads = 4 if hidden_size >= 64 else 2
        while hidden_size % heads != 0 and heads > 1:
            heads -= 1
        self.input_proj = nn.Linear(input_size, hidden_size)
        self.pos = PositionalProjection(hidden_size)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=max(1, heads),
            dim_feedforward=hidden_size * 2,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=max(1, num_layers))
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_classes)

    def get_embeddings(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x)
        x = self.pos(x)
        # Inject adjacency-smoothed signal before global attention.
        x = x + 0.5 * (adj @ x)
        encoded = self.encoder(x.unsqueeze(0)).squeeze(0)
        return self.dropout(encoded)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        emb = self.get_embeddings(x, adj)
        return self.classifier(emb)


def node2vec_embeddings(
    edge_pairs: list[tuple[int, int]],
    num_nodes: int,
    embedding_dim: int = 64,
    walk_length: int = 16,
    num_walks: int = 8,
    window_size: int = 4,
    seed: int = 42,
) -> np.ndarray:
    rng = np.random.default_rng(int(seed))
    neighbors: list[list[int]] = [[] for _ in range(num_nodes)]
    for src, tgt in edge_pairs:
        neighbors[src].append(tgt)

    counts = np.zeros((num_nodes, num_nodes), dtype=np.float32)
    for start in range(num_nodes):
        if not neighbors[start]:
            continue
        for _ in range(max(1, int(num_walks))):
            walk = [start]
            current = start
            for _ in range(max(1, int(walk_length)) - 1):
                next_hops = neighbors[current]
                if not next_hops:
                    break
                current = int(rng.choice(next_hops))
                walk.append(current)
            for idx, src in enumerate(walk):
                left = max(0, idx - int(window_size))
                right = min(len(walk), idx + int(window_size) + 1)
                for jdx in range(left, right):
                    if jdx == idx:
                        continue
                    counts[src, walk[jdx]] += 1.0

    counts = counts + np.eye(num_nodes, dtype=np.float32)
    row_sums = counts.sum(axis=1, keepdims=True)
    probs = counts / np.maximum(row_sums, 1e-8)
    ppmi = np.maximum(np.log(np.maximum(probs * num_nodes, 1e-8)), 0.0)

    from sklearn.decomposition import TruncatedSVD

    n_components = int(max(2, min(embedding_dim, num_nodes - 1)))
    if n_components >= num_nodes:
        n_components = max(1, num_nodes - 1)
    if n_components < 1:
        return np.zeros((num_nodes, 1), dtype=np.float32)

    svd = TruncatedSVD(n_components=n_components, random_state=int(seed))
    emb = svd.fit_transform(ppmi)
    if emb.shape[1] < embedding_dim:
        pad = np.zeros((num_nodes, embedding_dim - emb.shape[1]), dtype=np.float32)
        emb = np.concatenate([emb.astype(np.float32), pad], axis=1)
    return emb[:, :embedding_dim].astype(np.float32)

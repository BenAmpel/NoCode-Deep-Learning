"""
Graph modality pipeline.

Expected folder format:
    graph_dataset/
        nodes.csv   # must include node_id, label (for classification), and optional feature columns
        edges.csv   # must include source,target

This first pass focuses on node-level graph classification with full-graph
training and validation splits over labelled nodes.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from data_pipeline.io_utils import read_structured_file


def _graph_files(data_path: str) -> tuple[Path, Path]:
    root = Path(data_path)
    if not root.is_dir():
        raise ValueError("Graph datasets must be folders containing nodes.csv and edges.csv.")
    nodes_path = root / "nodes.csv"
    edges_path = root / "edges.csv"
    if not nodes_path.is_file() or not edges_path.is_file():
        raise ValueError("Expected graph folder to contain nodes.csv and edges.csv.")
    return nodes_path, edges_path


def _normalise_adj(adj: np.ndarray) -> np.ndarray:
    degree = adj.sum(axis=1)
    degree = np.where(degree == 0, 1.0, degree)
    inv_sqrt = np.diag(1.0 / np.sqrt(degree))
    return (inv_sqrt @ adj @ inv_sqrt).astype(np.float32)


def load_graph_data(
    data_path: str,
    label_col: str = "label",
    feature_cols: list[str] | None = None,
    val_split: float = 0.2,
    subset_percent: float = 100.0,
    subset_seed: int = 42,
) -> tuple[list[str], dict, dict]:
    nodes_path, edges_path = _graph_files(data_path)
    nodes_df = read_structured_file(nodes_path)
    edges_df = read_structured_file(edges_path)

    if "node_id" not in nodes_df.columns:
        raise ValueError("nodes.csv must contain a 'node_id' column.")
    if not {"source", "target"}.issubset(edges_df.columns):
        raise ValueError("edges.csv must contain 'source' and 'target' columns.")

    node_ids = nodes_df["node_id"].astype(str).tolist()
    node_to_idx = {node_id: idx for idx, node_id in enumerate(node_ids)}
    n_nodes = len(node_ids)
    if n_nodes < 3:
        raise ValueError("Graph dataset needs at least 3 nodes.")

    edge_pairs: list[tuple[int, int]] = []
    for _, row in edges_df.iterrows():
        source = str(row["source"])
        target = str(row["target"])
        if source not in node_to_idx or target not in node_to_idx:
            continue
        src_idx = node_to_idx[source]
        tgt_idx = node_to_idx[target]
        edge_pairs.append((src_idx, tgt_idx))
        edge_pairs.append((tgt_idx, src_idx))

    adj = np.zeros((n_nodes, n_nodes), dtype=np.float32)
    for src_idx, tgt_idx in edge_pairs:
        adj[src_idx, tgt_idx] = 1.0
    np.fill_diagonal(adj, 1.0)
    adj_norm = _normalise_adj(adj)

    usable_feature_cols = [col for col in (feature_cols or []) if col in nodes_df.columns and col not in {label_col, "node_id"}]
    if not usable_feature_cols:
        usable_feature_cols = [
            col for col in nodes_df.columns
            if col not in {label_col, "node_id"} and pd.api.types.is_numeric_dtype(nodes_df[col])
        ]

    if usable_feature_cols:
        features = nodes_df[usable_feature_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
    else:
        degree = adj.sum(axis=1, keepdims=True)
        self_loops = np.diag(adj).reshape(-1, 1)
        features = np.concatenate([degree, self_loops], axis=1).astype(np.float32)
        usable_feature_cols = ["degree", "self_loop"]

    feat_mean = features.mean(axis=0, keepdims=True)
    feat_std = features.std(axis=0, keepdims=True) + 1e-8
    features = ((features - feat_mean) / feat_std).astype(np.float32)

    if label_col not in nodes_df.columns:
        raise ValueError(f"nodes.csv must contain the selected label column '{label_col}' for graph classification.")
    labelled_df = nodes_df[nodes_df[label_col].notna()].copy()
    if labelled_df.empty:
        raise ValueError(f"No labelled nodes remain after dropping missing values from '{label_col}'.")

    labelled_idx = np.array([node_to_idx[str(node_id)] for node_id in labelled_df["node_id"].astype(str)], dtype=np.int64)
    label_values = labelled_df[label_col].astype(str).tolist()
    classes = sorted(set(label_values))
    if len(classes) < 2:
        raise ValueError("Graph classification needs at least 2 classes across labelled nodes.")
    class_to_idx = {label: idx for idx, label in enumerate(classes)}

    y_full = np.full(n_nodes, -1, dtype=np.int64)
    for idx, label in zip(labelled_idx, label_values):
        y_full[idx] = class_to_idx[str(label)]

    rng = np.random.default_rng(int(subset_seed))
    working_idx = labelled_idx.copy()
    if float(subset_percent) < 100.0:
        sample_count = max(2, int(round(len(working_idx) * (float(subset_percent) / 100.0))))
        sample_count = min(sample_count, len(working_idx))
        working_idx = np.sort(rng.choice(working_idx, size=sample_count, replace=False))

    labels_for_split = np.array([y_full[idx] for idx in working_idx], dtype=np.int64)
    per_class_counts = pd.Series(labels_for_split).value_counts()
    can_stratify = len(per_class_counts) > 1 and int(per_class_counts.min()) >= 2

    if len(working_idx) < 4:
        raise ValueError("Graph dataset needs at least 4 labelled nodes for a train/validation split.")

    if can_stratify:
        from sklearn.model_selection import train_test_split
        train_idx, val_idx = train_test_split(
            working_idx,
            test_size=float(val_split),
            random_state=int(subset_seed),
            stratify=labels_for_split,
        )
    else:
        shuffled = working_idx.copy()
        rng.shuffle(shuffled)
        n_val = max(1, int(round(len(shuffled) * float(val_split))))
        n_val = min(n_val, len(shuffled) - 1)
        val_idx = np.sort(shuffled[:n_val])
        train_idx = np.sort(shuffled[n_val:])

    prep = {
        "modality": "graph",
        "label_col": label_col,
        "feature_columns": usable_feature_cols,
        "selected_feature_columns": usable_feature_cols,
        "feature_order": usable_feature_cols,
        "input_size": int(features.shape[1]),
        "num_nodes": int(n_nodes),
        "num_edges": int(len(edge_pairs) // 2),
        "node_ids": node_ids,
        "graph_format": "folder:nodes.csv+edges.csv",
        "subset_percent": float(subset_percent),
    }
    graph_data = {
        "x": torch.tensor(features, dtype=torch.float32),
        "adj": torch.tensor(adj_norm, dtype=torch.float32),
        "y": torch.tensor(y_full, dtype=torch.long),
        "train_idx": torch.tensor(train_idx, dtype=torch.long),
        "val_idx": torch.tensor(val_idx, dtype=torch.long),
        "node_ids": node_ids,
        "feature_names": usable_feature_cols,
        "edge_pairs": edge_pairs,
        "input_size": int(features.shape[1]),
        "num_nodes": int(n_nodes),
    }
    return classes, prep, graph_data

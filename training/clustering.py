"""
Clustering via embedding extraction → KMeans.

Works with any PyTorch model that implements get_features().
Optionally produces a 2-D t-SNE visualisation.
"""
from __future__ import annotations

import numpy as np
import torch

from config import DEVICE


def extract_features(model: torch.nn.Module, data_loader) -> np.ndarray:
    """Run a forward pass over all data and collect penultimate embeddings."""
    model = model.to(DEVICE)
    model.eval()
    all_features: list[np.ndarray] = []

    with torch.no_grad():
        for batch in data_loader:
            inputs, _ = batch
            if isinstance(inputs, dict):
                inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
                if "attention_mask" in inputs:
                    feats = model.get_features(inputs["input_ids"], inputs["attention_mask"])
                else:
                    feats = model.get_features(inputs["input_ids"])
            else:
                feats = model.get_features(inputs.to(DEVICE))
            all_features.append(feats.cpu().numpy())

    return np.concatenate(all_features, axis=0)


def run_clustering(
    model: torch.nn.Module,
    data_loader,
    n_clusters: int = 8,
) -> dict:
    """
    Returns:
        cluster_labels   : list[int]
        silhouette_score : float  (-1 to 1, higher = better)
        n_clusters       : int
        features         : np.ndarray  (for downstream TSNE/UMAP)
        kmeans           : fitted KMeans object
    """
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    features = extract_features(model, data_loader)

    kmeans         = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(features)

    sil = (
        float(silhouette_score(features, cluster_labels))
        if len(set(cluster_labels.tolist())) > 1
        else 0.0
    )

    return {
        "cluster_labels":   cluster_labels.tolist(),
        "silhouette_score": round(sil, 4),
        "n_clusters":       n_clusters,
        "features":         features,
        "kmeans":           kmeans,
    }


def tsne_plot(features: np.ndarray, cluster_labels: list[int]) -> "matplotlib.figure.Figure":
    """Return a matplotlib figure of the 2-D t-SNE projection."""
    import matplotlib.pyplot as plt
    from sklearn.manifold import TSNE

    n   = len(features)
    ppl = min(30, max(5, n // 10))

    coords = TSNE(n_components=2, perplexity=ppl, random_state=42).fit_transform(features)

    fig, ax = plt.subplots(figsize=(7, 5))
    scatter = ax.scatter(coords[:, 0], coords[:, 1], c=cluster_labels, cmap="tab10", s=20, alpha=0.7)
    fig.colorbar(scatter, ax=ax, label="Cluster")
    ax.set_title("t-SNE of learned embeddings")
    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")
    fig.tight_layout()
    return fig

from __future__ import annotations

import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression

from config import DEVICE
from models.graph_models import node2vec_embeddings


def train_graph_pytorch(
    model: nn.Module,
    graph_data: dict,
    epochs: int,
    lr: float,
    optimizer_name: str = "adam",
    patience: int = 5,
    class_weights: torch.Tensor | None = None,
):
    x = graph_data["x"].to(DEVICE)
    adj = graph_data["adj"].to(DEVICE)
    y = graph_data["y"].to(DEVICE)
    train_idx = graph_data["train_idx"].to(DEVICE)
    val_idx = graph_data["val_idx"].to(DEVICE)

    model = model.to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(DEVICE) if class_weights is not None else None)
    if optimizer_name.lower() == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    elif optimizer_name.lower() == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_state = None
    best_val_loss = float("inf")
    best_epoch = 0
    patience_ctr = 0
    history = []
    epoch_times = []

    for epoch in range(1, int(epochs) + 1):
        t0 = time.time()
        model.train()
        optimizer.zero_grad()
        logits = model(x, adj)
        train_loss = criterion(logits[train_idx], y[train_idx])
        train_loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            logits = model(x, adj)
            val_loss = criterion(logits[val_idx], y[val_idx])
            preds = logits[val_idx].argmax(dim=1)
            val_acc = (preds == y[val_idx]).float().mean().item() * 100.0

        elapsed = time.time() - t0
        epoch_times.append(elapsed)
        avg_epoch = sum(epoch_times) / len(epoch_times)
        eta_seconds = avg_epoch * (int(epochs) - epoch)

        record = {
            "epoch": epoch,
            "train_loss": round(float(train_loss.item()), 4),
            "val_loss": round(float(val_loss.item()), 4),
            "val_acc": round(float(val_acc), 2),
            "eta_seconds": round(float(eta_seconds), 1),
            "lr": round(float(optimizer.param_groups[0]["lr"]), 6),
        }
        history.append(record)

        if float(val_loss.item()) < best_val_loss:
            best_val_loss = float(val_loss.item())
            best_epoch = epoch
            patience_ctr = 0
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        else:
            patience_ctr += 1

        stopped = patience_ctr >= int(patience)
        yield {**record, "epochs": int(epochs), "done": False, "stopped_early": stopped}
        if stopped:
            break

    if best_state is not None:
        model.load_state_dict({key: value.to(DEVICE) for key, value in best_state.items()})

    yield {"done": True, "model": model, "history": history, "best_epoch": best_epoch, "best_val": best_val_loss}


def train_node2vec_classifier(
    graph_data: dict,
    embedding_dim: int = 64,
    max_iter: int = 1000,
):
    embeddings = node2vec_embeddings(
        graph_data["edge_pairs"],
        num_nodes=int(graph_data["num_nodes"]),
        embedding_dim=int(embedding_dim),
    )
    train_idx = graph_data["train_idx"].cpu().numpy()
    val_idx = graph_data["val_idx"].cpu().numpy()
    y = graph_data["y"].cpu().numpy()

    model = LogisticRegression(max_iter=int(max_iter), random_state=42)
    model.fit(embeddings[train_idx], y[train_idx])
    val_probs = model.predict_proba(embeddings[val_idx]) if hasattr(model, "predict_proba") else None
    val_pred = model.predict(embeddings[val_idx])
    val_acc = float((val_pred == y[val_idx]).mean() * 100.0)
    history = [{"epoch": 1, "train_loss": 0.0, "val_loss": 0.0, "val_acc": round(val_acc, 2), "eta_seconds": 0.0, "lr": 0.0}]
    return {
        "model": model,
        "history": history,
        "embeddings": embeddings,
        "y_true": y[val_idx],
        "y_pred": val_pred,
        "y_prob": val_probs,
        "train_embeddings": embeddings[train_idx],
        "val_embeddings": embeddings[val_idx],
    }


def predict_graph_classification(model, graph_data: dict, *, sklearn_model: bool = False, embeddings: np.ndarray | None = None):
    y = graph_data["y"].cpu().numpy()
    val_idx = graph_data["val_idx"].cpu().numpy()
    if sklearn_model:
        if embeddings is None:
            raise ValueError("Embeddings are required for graph sklearn evaluation.")
        y_true = y[val_idx]
        y_pred = model.predict(embeddings[val_idx])
        y_prob = model.predict_proba(embeddings[val_idx]) if hasattr(model, "predict_proba") else None
        return y_true, y_pred, y_prob

    x = graph_data["x"].to(DEVICE)
    adj = graph_data["adj"].to(DEVICE)
    model = model.to(DEVICE).eval()
    with torch.no_grad():
        logits = model(x, adj)[graph_data["val_idx"].to(DEVICE)]
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        preds = logits.argmax(dim=1).cpu().numpy()
    return y[val_idx], preds, probs

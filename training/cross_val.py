"""
K-fold cross-validation for PyTorch models.

Works with any ``torch.utils.data.Dataset`` (tabular, image, text, etc.) by
using ``SubsetRandomSampler`` to create non-overlapping fold splits without
copying data.
"""
from __future__ import annotations

from typing import Callable, Generator

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, SubsetRandomSampler

from config import DEVICE
from training.trainer import _forward, _make_optimizer, _make_scheduler
from eval.metrics import classification_metric_summary, regression_metric_summary

# ---------------------------------------------------------------------------
# Internal fold-training helper
# ---------------------------------------------------------------------------

def _train_one_fold(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int,
    lr: float,
    optimizer_name: str,
    task: str,
) -> Generator[dict, None, None]:
    """Train *model* on a single fold, yielding per-epoch metrics.

    Yields dicts with keys: ``epoch``, ``train_loss``, ``val_loss``,
    ``val_acc``, and after the final epoch a ``history`` list.
    This is a leaner version of ``train_pytorch`` (no early-stopping, no AMP,
    no scheduler step-count) suitable for cross-validation inner loops.
    """
    model = model.to(DEVICE)
    is_clf    = task == "classification"
    criterion = nn.CrossEntropyLoss() if is_clf else nn.MSELoss()
    optimizer = _make_optimizer(model, optimizer_name, lr)
    scheduler = _make_scheduler(optimizer, "cosine", epochs, len(train_loader))

    history: list[dict] = []

    for epoch in range(1, epochs + 1):
        # ---- train ----
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            optimizer.zero_grad()
            logits, labels = _forward(model, batch)
            loss = criterion(logits, labels.float() if not is_clf else labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= max(len(train_loader), 1)

        # ---- validate ----
        model.eval()
        val_loss   = 0.0
        n_correct  = 0
        n_total    = 0
        val_metric = 0.0
        with torch.no_grad():
            for batch in val_loader:
                logits, labels = _forward(model, batch)
                lbl = labels.float() if not is_clf else labels
                val_loss += criterion(logits, lbl).item()
                if is_clf:
                    n_correct += (logits.argmax(1) == labels).sum().item()
                    n_total   += labels.size(0)
                else:
                    val_metric += (logits.squeeze() - labels.float()).abs().mean().item()
        val_loss /= max(len(val_loader), 1)
        if is_clf:
            val_metric = (n_correct / max(n_total, 1)) * 100
        else:
            val_metric /= max(len(val_loader), 1)

        if scheduler:
            scheduler.step()

        record = {
            "epoch":      epoch,
            "train_loss": round(train_loss, 4),
            "val_loss":   round(val_loss, 4),
            "val_acc":    round(val_metric, 2),
            "lr":         round(optimizer.param_groups[0]["lr"], 6),
        }
        history.append(record)
        yield record

    yield {"history": history}


def _extract_labels(dataset: Dataset) -> np.ndarray | None:
    if hasattr(dataset, "labels"):
        labels = getattr(dataset, "labels")
        if hasattr(labels, "cpu"):
            return labels.cpu().numpy()
        return np.asarray(labels)
    if hasattr(dataset, "tensors") and len(getattr(dataset, "tensors")) >= 2:
        labels = dataset.tensors[1]
        if hasattr(labels, "cpu"):
            return labels.cpu().numpy()
        return np.asarray(labels)
    if hasattr(dataset, "samples"):
        return np.asarray([sample[1] for sample in dataset.samples])
    return None


def _collect_fold_predictions(model: nn.Module, val_loader: DataLoader, task: str) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    import torch.nn.functional as F

    model = model.to(DEVICE)
    model.eval()
    all_true, all_pred, all_prob = [], [], []
    with torch.no_grad():
        for batch in val_loader:
            logits, labels = _forward(model, batch)
            if task == "classification":
                probs = F.softmax(logits, dim=1).cpu().numpy()
                preds = logits.argmax(1).cpu().numpy()
                all_prob.extend(probs)
                all_pred.extend(preds)
                all_true.extend(labels.cpu().numpy())
            else:
                preds = logits.squeeze().cpu().numpy()
                all_pred.extend(np.atleast_1d(preds))
                all_true.extend(np.atleast_1d(labels.float().cpu().numpy()))
    y_true = np.asarray(all_true)
    y_pred = np.asarray(all_pred)
    y_prob = np.asarray(all_prob) if all_prob else None
    return y_true, y_pred, y_prob


def _aggregate_metric_dicts(metric_dicts: list[dict[str, float | None]]) -> dict[str, dict[str, float | None]]:
    if not metric_dicts:
        return {}
    keys = metric_dicts[0].keys()
    aggregate: dict[str, dict[str, float | None]] = {}
    for key in keys:
        vals = [m[key] for m in metric_dicts if m.get(key) is not None]
        if not vals:
            aggregate[key] = {"mean": None, "std": None}
            continue
        arr = np.asarray(vals, dtype=float)
        aggregate[key] = {"mean": float(arr.mean()), "std": float(arr.std())}
    return aggregate


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def cross_validate(
    dataset: Dataset,
    model_factory: Callable[[], nn.Module],
    k: int = 5,
    epochs: int = 10,
    lr: float = 1e-3,
    optimizer_name: str = "adam",
    task: str = "classification",
    batch_size: int = 32,
    splits: list[tuple[np.ndarray, np.ndarray]] | None = None,
    split_strategy_name: str | None = None,
) -> Generator[dict, None, None]:
    """Run stratification-free k-fold cross-validation on *dataset*.

    Each fold uses a fresh model produced by *model_factory* so weights are
    never shared across folds.

    Parameters
    ----------
    dataset:
        A ``torch.utils.data.Dataset`` whose ``__len__`` is implemented.
    model_factory:
        Zero-argument callable returning a new, untrained ``nn.Module``.
    k:
        Number of folds.  Must satisfy ``2 <= k <= len(dataset)``.
    epochs:
        Training epochs per fold.
    lr:
        Learning rate for the optimizer.
    optimizer_name:
        One of ``"adam"``, ``"adamw"``, ``"sgd"``.
    task:
        ``"classification"`` or ``"regression"``.
    batch_size:
        Batch size used for both train and validation ``DataLoader`` objects.

    Yields
    ------
    dict
        Per-epoch progress within each fold::

            {"fold": int, "total_folds": int, "epoch": int,
             "val_acc": float, "done": False}

        After all folds are complete::

            {"done": True, "fold_accs": list[float], "mean_acc": float,
             "std_acc": float, "fold_histories": list[list[dict]]}
    """
    n = len(dataset)  # type: ignore[arg-type]
    if k < 2:
        raise ValueError(f"k must be >= 2, got {k}")
    if k > n:
        raise ValueError(f"k ({k}) cannot exceed dataset size ({n})")

    if splits is not None:
        k = len(splits)

    fold_accs: list[float]         = []
    fold_histories: list[list[dict]] = []
    fold_metrics: list[dict[str, float | None]] = []

    labels = _extract_labels(dataset)
    split_warning = None
    split_strategy = split_strategy_name or "kfold"
    if splits is None:
        if task == "classification" and labels is not None:
            from sklearn.model_selection import StratifiedKFold
            unique, counts = np.unique(labels, return_counts=True)
            min_class_size = int(counts.min()) if len(counts) else 0
            if min_class_size >= k:
                splitter = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
                splits = list(splitter.split(np.zeros(n), labels))
                split_strategy = "stratified_kfold"
            else:
                from sklearn.model_selection import KFold

                splitter = KFold(n_splits=k, shuffle=True, random_state=42)
                splits = list(splitter.split(np.zeros(n)))
                split_strategy = "kfold"
                split_warning = (
                    f"Using plain K-Fold instead of StratifiedKFold because the smallest class "
                    f"has only {min_class_size} samples, which is fewer than k={k}."
                )
        else:
            from sklearn.model_selection import KFold

            splitter = KFold(n_splits=k, shuffle=True, random_state=42)
            splits = list(splitter.split(np.zeros(n)))

    for fold, (train_idx, val_idx) in enumerate(splits):
        train_idx = train_idx.tolist()
        val_idx = val_idx.tolist()

        train_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=SubsetRandomSampler(train_idx),
        )
        val_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=SubsetRandomSampler(val_idx),
        )

        model = model_factory()
        fold_history: list[dict] = []

        for update in _train_one_fold(
            model, train_loader, val_loader,
            epochs, lr, optimizer_name, task,
        ):
            if "history" in update:
                # Terminal sentinel from _train_one_fold
                fold_history = update["history"]
            else:
                yield {
                    "fold":        fold + 1,
                    "total_folds": k,
                    "epoch":       update["epoch"],
                    "val_acc":     update["val_acc"],
                    "done":        False,
                }

        best_acc = max((r["val_acc"] for r in fold_history), default=0.0)
        fold_accs.append(best_acc)
        fold_histories.append(fold_history)
        y_true, y_pred, y_prob = _collect_fold_predictions(model, val_loader, task)
        if task == "classification":
            metric_summary = classification_metric_summary(y_true, y_pred, y_prob, n_classes=len(np.unique(y_true)))
            fold_accs[-1] = round(float(metric_summary["accuracy"]) * 100, 2)
        else:
            metric_summary = regression_metric_summary(y_true, y_pred)
            fold_accs[-1] = round(float(metric_summary["mae"]), 4)
        fold_metrics.append(metric_summary)

    mean_acc = sum(fold_accs) / len(fold_accs) if fold_accs else 0.0
    variance = (
        sum((a - mean_acc) ** 2 for a in fold_accs) / len(fold_accs)
        if fold_accs else 0.0
    )
    std_acc = variance ** 0.5

    yield {
        "done":           True,
        "fold_accs":      fold_accs,
        "mean_acc":       round(mean_acc, 4),
        "std_acc":        round(std_acc, 4),
        "fold_histories": fold_histories,
        "fold_metrics":   fold_metrics,
        "aggregate_metrics": _aggregate_metric_dicts(fold_metrics),
        "split_strategy": split_strategy,
        "warning": split_warning,
    }


def format_cv_results(
    fold_accs: list[float],
    mean_acc: float,
    std_acc: float,
    fold_metrics: list[dict[str, float | None]] | None = None,
    aggregate_metrics: dict[str, dict[str, float | None]] | None = None,
    task: str = "classification",
) -> str:
    """Render cross-validation results as a Markdown string.

    Parameters
    ----------
    fold_accs:
        Per-fold best validation accuracies (percentages).
    mean_acc:
        Mean accuracy across folds.
    std_acc:
        Standard deviation of fold accuracies.

    Returns
    -------
    str
        Formatted Markdown, e.g.::

            ### Cross-Validation Results (5-fold)
            | Fold | Val Accuracy |
            |------|-------------|
            | 1    | 87.30%      |
            | 2    | 84.10%      |
            ...
            **Mean: 85.40% ± 2.10%**
    """
    k = len(fold_accs)
    if task == "classification":
        header = "| Fold | Accuracy | Precision | Recall | F1 | AUC | MCC |"
        divider = "|------|----------|-----------|--------|----|-----|-----|"
        rows = []
        for i, metrics in enumerate(fold_metrics or []):
            auc_val = metrics.get("auc_ovr")
            rows.append(
                f"| {i + 1} | {float(metrics['accuracy']) * 100:.2f}% | "
                f"{float(metrics['precision_macro']) * 100:.2f}% | "
                f"{float(metrics['recall_macro']) * 100:.2f}% | "
                f"{float(metrics['f1_macro']) * 100:.2f}% | "
                f"{(f'{float(auc_val):.3f}' if auc_val is not None else 'n/a')} | "
                f"{float(metrics['mcc']):.3f} |"
            )
        summary_lines = []
        for label, key, scale_pct in [
            ("Accuracy", "accuracy", True),
            ("Precision (macro)", "precision_macro", True),
            ("Recall (macro)", "recall_macro", True),
            ("F1 (macro)", "f1_macro", True),
            ("Balanced accuracy", "balanced_accuracy", True),
            ("AUC (OvR)", "auc_ovr", False),
            ("MCC", "mcc", False),
        ]:
            aggregate = (aggregate_metrics or {}).get(key, {})
            if aggregate.get("mean") is None:
                summary_lines.append(f"- **{label}**: n/a")
                continue
            mean_val = float(aggregate["mean"])
            std_val = float(aggregate["std"])
            if scale_pct:
                summary_lines.append(f"- **{label}**: {mean_val * 100:.2f}% ± {std_val * 100:.2f}%")
            else:
                summary_lines.append(f"- **{label}**: {mean_val:.3f} ± {std_val:.3f}")
    else:
        header = "| Fold | MAE | RMSE | R² |"
        divider = "|------|-----|------|----|"
        rows = [
            f"| {i + 1} | {float(metrics['mae']):.4f} | {float(metrics['rmse']):.4f} | {float(metrics['r2']):.4f} |"
            for i, metrics in enumerate(fold_metrics or [])
        ]
        summary_lines = []
        for label, key in [("MAE", "mae"), ("RMSE", "rmse"), ("R²", "r2")]:
            aggregate = (aggregate_metrics or {}).get(key, {})
            if aggregate.get("mean") is None:
                summary_lines.append(f"- **{label}**: n/a")
                continue
            summary_lines.append(f"- **{label}**: {float(aggregate['mean']):.4f} ± {float(aggregate['std']):.4f}")

    lines = [
        f"### Cross-Validation Results ({k}-fold)",
        "",
        header,
        divider,
        *rows,
        "",
        "### Aggregate metrics",
        *summary_lines,
        "",
        f"**Primary fold summary: {mean_acc:.2f}{'%' if task == 'classification' else ''} ± {std_acc:.2f}{'%' if task == 'classification' else ''}**",
    ]
    return "\n".join(lines)

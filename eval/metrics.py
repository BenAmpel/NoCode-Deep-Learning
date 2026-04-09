"""
Confusion matrix, classification report, ROC/AUC curves.
Works for both PyTorch models and sklearn models.
"""
from __future__ import annotations
import warnings
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix, classification_report,
    roc_curve, auc, accuracy_score, precision_score, recall_score, f1_score,
    balanced_accuracy_score, matthews_corrcoef, roc_auc_score,
    mean_absolute_error, mean_squared_error, r2_score,
)
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.preprocessing import label_binarize


def compute_confusion_matrix_from_arrays(y_true, y_pred, classes) -> tuple[plt.Figure | None, str]:
    """Build a confusion matrix figure and report text from arrays."""
    if y_true is None or y_pred is None or len(y_true) == 0:
        return None, "Evaluation data is unavailable."
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    acc = (y_true == y_pred).mean() * 100
    label_space = list(range(len(classes)))
    report = classification_report(
        y_true,
        y_pred,
        labels=label_space,
        target_names=classes,
        zero_division=0,
    )
    summary = f"Validation Accuracy: {acc:.2f}%\n\n{report}"

    cm = confusion_matrix(y_true, y_pred, labels=label_space)
    n = len(classes)
    fig, ax = plt.subplots(figsize=(max(6, n), max(5, n - 1)))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    fig.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(n), yticks=np.arange(n),
        xticklabels=classes, yticklabels=classes,
        ylabel="True label", xlabel="Predicted label",
        title="Confusion Matrix",
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    thresh = cm.max() / 2.0 if cm.size else 0.0
    for i in range(n):
        for j in range(n):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return fig, summary


def compute_roc_curves_from_arrays(y_true, y_prob, classes) -> plt.Figure | None:
    if y_prob is None or len(classes) < 2 or len(y_true) == 0:
        return None

    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    n = len(classes)
    y_bin = label_binarize(y_true, classes=list(range(n)))
    if n == 2:
        y_bin = np.hstack([1 - y_bin, y_bin])

    fig, ax = plt.subplots(figsize=(7, 5))
    colors = plt.cm.tab10(np.linspace(0, 1, n))
    for i, (cls, color) in enumerate(zip(classes, colors)):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_prob[:, i])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=color, lw=2, label=f"{cls} (AUC={roc_auc:.2f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set(xlabel="False Positive Rate", ylabel="True Positive Rate", title="ROC Curves (One-vs-Rest)")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    return fig

def _get_preds_and_probs(model, val_loader, classes, is_sklearn, X_val, y_val, device):
    """Returns (y_true np.ndarray, y_pred np.ndarray, y_prob np.ndarray shape [N, C])"""
    import torch, torch.nn.functional as F
    if is_sklearn:
        y_pred = model.predict(X_val)
        y_prob = model.predict_proba(X_val) if hasattr(model, "predict_proba") else None
        return y_val, y_pred, y_prob
    model = model.to(device)
    model.eval()
    all_pred, all_true, all_prob = [], [], []
    with torch.no_grad():
        for batch in val_loader:
            inputs, labels = batch
            if isinstance(inputs, dict):
                inputs = {k: v.to(device) for k, v in inputs.items()}
                logits = model(inputs["input_ids"], inputs.get("attention_mask")) if "attention_mask" in inputs else model(inputs["input_ids"])
            else:
                logits = model(inputs.to(device))
            probs = F.softmax(logits, dim=1).cpu().numpy()
            preds = logits.argmax(1).cpu().numpy()
            all_pred.extend(preds)
            all_true.extend(labels.numpy())
            all_prob.extend(probs)
    return np.array(all_true), np.array(all_pred), np.array(all_prob)


def classification_metric_summary(y_true, y_pred, y_prob, n_classes: int) -> dict[str, float | None]:
    """Compute a richer classification metric suite from prediction arrays."""
    label_space = list(range(n_classes))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        warnings.simplefilter("ignore", category=UndefinedMetricWarning)
        metrics = {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "precision_macro": float(precision_score(y_true, y_pred, labels=label_space, average="macro", zero_division=0)),
            "recall_macro": float(recall_score(y_true, y_pred, labels=label_space, average="macro", zero_division=0)),
            "f1_macro": float(f1_score(y_true, y_pred, labels=label_space, average="macro", zero_division=0)),
            "precision_weighted": float(precision_score(y_true, y_pred, labels=label_space, average="weighted", zero_division=0)),
            "recall_weighted": float(recall_score(y_true, y_pred, labels=label_space, average="weighted", zero_division=0)),
            "f1_weighted": float(f1_score(y_true, y_pred, labels=label_space, average="weighted", zero_division=0)),
            "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
            "mcc": float(matthews_corrcoef(y_true, y_pred)),
            "auc_ovr": None,
        }
    if y_prob is not None:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=UserWarning)
                warnings.simplefilter("ignore", category=UndefinedMetricWarning)
                if n_classes == 2:
                    score_source = y_prob[:, 1] if y_prob.ndim == 2 and y_prob.shape[1] > 1 else y_prob
                    metrics["auc_ovr"] = float(roc_auc_score(y_true, score_source))
                else:
                    metrics["auc_ovr"] = float(roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro"))
        except Exception:
            metrics["auc_ovr"] = None
    return metrics


def regression_metric_summary(y_true, y_pred) -> dict[str, float]:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": rmse,
        "r2": float(r2_score(y_true, y_pred)),
    }

def compute_confusion_matrix(model, val_loader, classes, modality, task, is_sklearn=False, X_val=None, y_val=None) -> tuple[plt.Figure | None, str]:
    """Returns (fig, summary_text)"""
    if task == "clustering":
        return None, "Confusion matrix not available for clustering task."
    from config import DEVICE
    try:
        y_true, y_pred, _ = _get_preds_and_probs(model, val_loader, classes, is_sklearn, X_val, y_val, DEVICE)
    except Exception as e:
        return None, f"Evaluation error: {e}"

    acc = (y_true == y_pred).mean() * 100
    label_space = list(range(len(classes)))
    report = classification_report(
        y_true,
        y_pred,
        labels=label_space,
        target_names=classes,
        zero_division=0,
    )
    summary = f"Validation Accuracy: {acc:.2f}%\n\n{report}"

    cm = confusion_matrix(y_true, y_pred, labels=label_space)
    n = len(classes)
    fig, ax = plt.subplots(figsize=(max(6, n), max(5, n - 1)))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    fig.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(n), yticks=np.arange(n),
        xticklabels=classes, yticklabels=classes,
        ylabel="True label", xlabel="Predicted label",
        title="Confusion Matrix",
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    thresh = cm.max() / 2.0
    for i in range(n):
        for j in range(n):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return fig, summary

def compute_roc_curves(model, val_loader, classes, modality, task, is_sklearn=False, X_val=None, y_val=None) -> plt.Figure | None:
    if task == "clustering":
        return None
    from config import DEVICE
    try:
        y_true, y_pred, y_prob = _get_preds_and_probs(model, val_loader, classes, is_sklearn, X_val, y_val, DEVICE)
    except Exception as e:
        return None
    if y_prob is None or len(classes) < 2:
        return None

    n = len(classes)
    y_bin = label_binarize(y_true, classes=list(range(n)))
    if n == 2:
        y_bin = np.hstack([1 - y_bin, y_bin])

    fig, ax = plt.subplots(figsize=(7, 5))
    colors = plt.cm.tab10(np.linspace(0, 1, n))
    for i, (cls, color) in enumerate(zip(classes, colors)):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_prob[:, i])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=color, lw=2, label=f"{cls} (AUC={roc_auc:.2f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set(xlabel="False Positive Rate", ylabel="True Positive Rate", title="ROC Curves (One-vs-Rest)")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    return fig

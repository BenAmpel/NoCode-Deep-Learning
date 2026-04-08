"""
Generates plain-English summaries and diagnostics for completed training runs.

Usage
-----
    from ui.training_summary import summarise_run, diagnose_loss_curve, format_hyperparams

    summary_md  = summarise_run(history, task="classification",
                                model_name="EfficientNet-B0", modality="image",
                                stopped_early=False)
    diagnostics = diagnose_loss_curve(history)
    table_md    = format_hyperparams({"lr": 1e-3, "batch_size": 16, "epochs": 10})
"""
from __future__ import annotations

from typing import Any


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

def diagnose_loss_curve(history: list[dict]) -> dict:
    """
    Analyse a training history and return a structured diagnostic report.

    Parameters
    ----------
    history:
        List of per-epoch dicts with keys:
        ``epoch``, ``train_loss``, ``val_loss``, ``val_acc``, ``lr``.

    Returns
    -------
    dict
        Keys:

        - ``converged``  (bool) — whether val_loss decreased overall.
        - ``overfit_score`` (float 0–1) — normalised train/val loss gap at the
          last epoch; 0 = no overfit, 1 = severe overfit.
        - ``best_epoch`` (int) — epoch index (0-based) with the lowest val_loss.
        - ``trend`` (str) — ``"improving"`` / ``"plateau"`` / ``"diverging"``
          based on the slope of the last 3 val_loss values.
    """
    if not history:
        return {
            "converged": False,
            "overfit_score": 0.0,
            "best_epoch": 0,
            "trend": "plateau",
        }

    val_losses  = [ep["val_loss"]   for ep in history]
    train_losses = [ep["train_loss"] for ep in history]

    # Best epoch
    best_epoch = int(min(range(len(val_losses)), key=lambda i: val_losses[i]))

    # Convergence: final val_loss < initial val_loss
    converged = val_losses[-1] < val_losses[0] if len(val_losses) > 1 else False

    # Overfit score — gap between train and val at the final epoch, normalised
    final_train = train_losses[-1]
    final_val   = val_losses[-1]
    gap = final_val - final_train          # positive → val worse than train
    normaliser  = max(abs(final_val), 1e-9)
    overfit_score = float(max(0.0, min(1.0, gap / normaliser)))

    # Trend: slope over last 3 epochs
    window = val_losses[-3:] if len(val_losses) >= 3 else val_losses
    if len(window) >= 2:
        slope = (window[-1] - window[0]) / max(len(window) - 1, 1)
        if slope < -0.005:
            trend = "improving"
        elif slope > 0.005:
            trend = "diverging"
        else:
            trend = "plateau"
    else:
        trend = "plateau"

    return {
        "converged":     converged,
        "overfit_score": overfit_score,
        "best_epoch":    best_epoch,
        "trend":         trend,
    }


def summarise_run(
    history: list[dict],
    task: str,
    model_name: str,
    modality: str,
    stopped_early: bool,
) -> str:
    """
    Generate a 3–5 sentence plain-English markdown summary of a training run.

    Parameters
    ----------
    history:
        List of per-epoch dicts (see :func:`diagnose_loss_curve` for keys).
    task:
        ``"classification"``, ``"regression"``, or ``"clustering"``.
    model_name:
        Architecture name, e.g. ``"EfficientNet-B0"``.
    modality:
        Data modality, e.g. ``"image"``, ``"text"``, ``"tabular"``.
    stopped_early:
        Whether early stopping fired before the configured epoch limit.

    Returns
    -------
    str
        A markdown string with emoji indicators (✅ ⚠️ ❌) for quick scanning.
    """
    if not history:
        return "❌ **No training history available.** The run may have failed before the first epoch."

    diag = diagnose_loss_curve(history)
    final_epoch_data = history[-1]
    n_epochs = len(history)

    final_val_loss = final_epoch_data.get("val_loss", float("nan"))
    final_train_loss = final_epoch_data.get("train_loss", float("nan"))
    final_val_acc  = final_epoch_data.get("val_acc")   # may be None for regression

    sentences: list[str] = []

    # ------------------------------------------------------------------
    # 1. Convergence sentence
    # ------------------------------------------------------------------
    if diag["trend"] == "improving":
        sentences.append(
            f"✅ **Convergence looks good** — the validation loss was still decreasing "
            f"at epoch {n_epochs}, suggesting the model has room to improve with more epochs."
        )
    elif diag["trend"] == "plateau":
        sentences.append(
            f"⚠️ **Training has plateaued** — the validation loss barely changed over "
            f"the last few epochs (final val loss: `{final_val_loss:.4f}`)."
        )
    else:  # diverging
        sentences.append(
            f"❌ **Validation loss is rising** (final: `{final_val_loss:.4f}`) — "
            f"the model may be overfitting or the learning rate may be too high."
        )

    # ------------------------------------------------------------------
    # 2. Performance sentence (classification only)
    # ------------------------------------------------------------------
    if task == "classification" and final_val_acc is not None:
        pct = final_val_acc * 100 if final_val_acc <= 1.0 else final_val_acc
        quality = _contextualise_accuracy(pct, modality, history)
        sentences.append(quality)
    elif task == "regression":
        sentences.append(
            f"📉 Final validation loss: `{final_val_loss:.4f}`. "
            "For regression tasks, compare this against a naïve baseline (e.g. predicting the mean) "
            "to gauge real-world usefulness."
        )

    # ------------------------------------------------------------------
    # 3. Overfitting sentence
    # ------------------------------------------------------------------
    os_ = diag["overfit_score"]
    if os_ > 0.25:
        sentences.append(
            f"⚠️ **Possible overfitting detected** — the gap between training loss "
            f"(`{final_train_loss:.4f}`) and validation loss (`{final_val_loss:.4f}`) "
            f"is significant (overfit score: `{os_:.2f}/1.0`). "
            "Consider increasing dropout, adding augmentation, or gathering more data."
        )
    elif os_ > 0.05:
        sentences.append(
            f"⚠️ **Mild train/val gap** (overfit score `{os_:.2f}`) — keep an eye on this "
            "if you train for more epochs."
        )
    else:
        sentences.append(
            f"✅ **Train and validation losses are close** (overfit score `{os_:.2f}`) — "
            "no obvious overfitting."
        )

    # ------------------------------------------------------------------
    # 4. Early stopping sentence
    # ------------------------------------------------------------------
    if stopped_early:
        best = diag["best_epoch"] + 1  # 1-based for display
        sentences.append(
            f"🛑 **Early stopping triggered** — training halted after epoch {n_epochs} "
            f"(best checkpoint at epoch {best})."
        )

    # ------------------------------------------------------------------
    # 5. Recommendation
    # ------------------------------------------------------------------
    sentences.append(_recommend(diag, task, modality, n_epochs, stopped_early, os_))

    return "\n\n".join(sentences)


# ---------------------------------------------------------------------------
# format_hyperparams
# ---------------------------------------------------------------------------

def format_hyperparams(params: dict[str, Any]) -> str:
    """
    Render a hyperparameter dictionary as a clean, two-column markdown table.

    Parameters
    ----------
    params:
        Flat dict of hyperparameter names to values.

    Returns
    -------
    str
        A markdown table string.

    Example
    -------
    ::

        | Hyperparameter | Value |
        |----------------|-------|
        | learning_rate  | 0.001 |
        | batch_size     | 16    |
    """
    if not params:
        return "_No hyperparameters recorded._"

    rows = ["| Hyperparameter | Value |", "|:---------------|------:|"]
    for key, value in params.items():
        # Pretty-print floats in scientific notation if very small
        if isinstance(value, float) and (abs(value) < 0.001 or abs(value) >= 10_000):
            display_value = f"`{value:.2e}`"
        elif isinstance(value, float):
            display_value = f"`{value:.4f}`"
        elif isinstance(value, bool):
            display_value = "✅ Yes" if value else "❌ No"
        elif value is None:
            display_value = "_none_"
        else:
            display_value = f"`{value}`"

        label = key.replace("_", " ").title()
        rows.append(f"| {label} | {display_value} |")

    return "\n".join(rows)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _contextualise_accuracy(pct: float, modality: str, history: list[dict]) -> str:
    """Return a performance sentence with context-appropriate benchmarks."""
    # Infer number of classes from accuracy magnitude is unreliable;
    # instead use broad thresholds tuned per modality.
    _THRESHOLDS: dict[str, tuple[float, float]] = {
        # (strong, weak)
        "image":     (85.0, 60.0),
        "text":      (88.0, 65.0),
        "audio":     (80.0, 55.0),
        "video":     (75.0, 50.0),
        "tabular":   (85.0, 65.0),
        "timeseries":(80.0, 60.0),
    }
    strong_thresh, weak_thresh = _THRESHOLDS.get(modality, (80.0, 60.0))

    if pct >= strong_thresh:
        verdict = f"✅ **{pct:.1f}% validation accuracy** is strong for {modality} classification."
    elif pct >= weak_thresh:
        verdict = (
            f"⚠️ **{pct:.1f}% validation accuracy** is moderate for {modality} classification — "
            "there's room for improvement."
        )
    else:
        # Check if near random for 2-class
        n_epochs = len(history)
        near_random_hint = ""
        if pct <= 55.0:
            near_random_hint = (
                " This is near random-chance for a binary problem — "
                "double-check your labels and data pipeline."
            )
        verdict = (
            f"❌ **{pct:.1f}% validation accuracy** is low for {modality} classification.{near_random_hint}"
        )

    return verdict


def _recommend(
    diag: dict,
    task: str,
    modality: str,
    n_epochs: int,
    stopped_early: bool,
    overfit_score: float,
) -> str:
    """Return a single concrete next-step recommendation."""
    trend = diag["trend"]

    if overfit_score > 0.25:
        return (
            "💡 **Next step:** Try increasing **dropout**, adding **data augmentation**, "
            "or collecting more training samples to close the train/val gap."
        )
    if trend == "improving" and not stopped_early:
        return (
            f"💡 **Next step:** The model is still learning — try increasing **epochs** "
            f"(currently {n_epochs}) to see if performance continues to improve."
        )
    if trend == "diverging":
        return (
            "💡 **Next step:** Reduce the **learning rate** (try 10× smaller) or enable a "
            "**cosine/warmup scheduler** to stabilise training."
        )
    if task == "classification" and modality in ("image", "audio", "video"):
        return (
            "💡 **Next step:** Results look reasonable — try **exporting** the model and "
            "testing it on a few real samples to confirm it generalises."
        )
    if stopped_early and n_epochs <= 3:
        return (
            "💡 **Next step:** Your dataset may be too small. Consider **data augmentation** "
            "or using a smaller pretrained model to avoid overfitting on limited data."
        )
    return (
        "💡 **Next step:** The model looks well-trained. Try **exporting** it and "
        "running the inference panel to test on unseen samples."
    )

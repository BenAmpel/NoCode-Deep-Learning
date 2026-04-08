"""SHAP feature importance for tabular models and token importance for text."""
from __future__ import annotations
import contextlib
import html as html_mod
import io
import logging
import numpy as np
import matplotlib.pyplot as plt

logger = logging.getLogger("nocode_dl")

def compute_shap(model, extra_data: dict, modality: str, prep: dict) -> plt.Figure | None:
    if modality != "tabular":
        return None
    X_val = extra_data.get("X_val")
    if X_val is None or len(X_val) == 0:
        return None

    try:
        import shap
    except ImportError:
        return _fallback_importance(model, extra_data, prep)

    # Reconstruct feature names
    num_cols  = prep.get("numeric_columns", [])
    ohe_cats  = prep.get("ohe_categories",  {})
    ohe_names = [f"{col}={val}" for col, cats in ohe_cats.items() for val in cats]
    feature_names = list(num_cols) + ohe_names

    model_name = type(model).__name__
    n_bg = min(50, len(X_val))
    X_bg = X_val[:n_bg]

    try:
        if model_name == "RandomForestClassifier":
            explainer  = shap.TreeExplainer(model)
            shap_vals  = explainer.shap_values(X_val[:100])
            if isinstance(shap_vals, list):
                shap_vals = np.abs(np.stack(shap_vals)).mean(0)
        elif model_name == "LogisticRegression":
            explainer = shap.LinearExplainer(model, X_bg)
            sv = explainer.shap_values(X_val[:100])
            shap_vals = np.abs(sv if sv.ndim == 2 else np.stack(sv).mean(0))
        else:  # MLP — use KernelExplainer with kmeans background
            bg = shap.kmeans(X_bg, min(10, n_bg))
            def predict_fn(x):
                import torch
                with torch.no_grad():
                    return torch.softmax(model(torch.tensor(x, dtype=torch.float32)), dim=1).numpy()
            with contextlib.redirect_stdout(io.StringIO()):
                explainer = shap.KernelExplainer(predict_fn, bg)
                sv = explainer.shap_values(X_val[:30], nsamples=50)
            shap_vals = np.abs(np.stack(sv)).mean(0)

        mean_abs = np.abs(shap_vals).mean(0)
        top_k = min(20, len(mean_abs))
        idx   = np.argsort(mean_abs)[-top_k:][::-1]
        names = [feature_names[i] if i < len(feature_names) else f"feat_{i}" for i in idx]
        vals  = mean_abs[idx]

        fig, ax = plt.subplots(figsize=(7, max(4, top_k * 0.35)))
        bars = ax.barh(range(top_k), vals[::-1], color="steelblue")
        ax.set_yticks(range(top_k))
        ax.set_yticklabels(names[::-1], fontsize=8)
        ax.set_xlabel("Mean |SHAP value|")
        ax.set_title(f"SHAP Feature Importance — {model_name}")
        fig.tight_layout()
        return fig
    except Exception as e:
        return _fallback_importance(model, extra_data, prep)

def _fallback_importance(model, extra_data, prep):
    """For RandomForest, fall back to built-in feature_importances_."""
    if not hasattr(model, "feature_importances_"):
        return None
    fi = model.feature_importances_
    num_cols  = prep.get("numeric_columns", [])
    ohe_cats  = prep.get("ohe_categories", {})
    ohe_names = [f"{col}={val}" for col, cats in ohe_cats.items() for val in cats]
    names = list(num_cols) + ohe_names
    top_k = min(20, len(fi))
    idx   = np.argsort(fi)[-top_k:][::-1]
    fig, ax = plt.subplots(figsize=(7, max(4, top_k * 0.35)))
    ax.barh(range(top_k), fi[idx][::-1], color="steelblue")
    ax.set_yticks(range(top_k))
    ax.set_yticklabels([names[i] if i < len(names) else f"f{i}" for i in idx][::-1], fontsize=8)
    ax.set_xlabel("Feature Importance")
    ax.set_title("Feature Importance (built-in)")
    fig.tight_layout()
    return fig


# ── Text token importance via occlusion-based SHAP ───────────────────────────

def compute_text_shap(
    model,
    val_loader,
    classes: list[str],
    prep: dict,
    n_samples: int = 3,
) -> str | None:
    """Compute per-token importance for text models and return HTML.

    Uses an occlusion (leave-one-out) approach: for each token, replace it with
    the padding index and measure the drop in predicted-class probability.
    This is fast, deterministic, and doesn't require the SHAP library.

    Returns an HTML string with tokens highlighted red (opacity ∝ importance),
    or ``None`` if the modality isn't text.
    """
    import torch

    device = next(model.parameters()).device
    model.eval()

    # Build reverse vocabulary for decoding token IDs → words
    vocab_map = prep.get("vocab")
    tokenizer_name = prep.get("tokenizer")
    if vocab_map:
        idx_to_word = {int(v): k for k, v in vocab_map.items()}
    elif tokenizer_name:
        from transformers import AutoTokenizer
        _tok = AutoTokenizer.from_pretrained(tokenizer_name)
        idx_to_word = None  # will use tokenizer.decode
    else:
        return None

    # Grab a batch
    batch = next(iter(val_loader))
    inputs, labels = batch
    input_ids = inputs["input_ids"][:n_samples].to(device)
    attention_mask = inputs.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask[:n_samples].to(device)
    labels = labels[:n_samples].to(device)

    html_parts = [
        '<div style="font-family: inherit; line-height: 2.2;">',
        '<h4 style="margin: 0 0 12px; color: var(--color-text-primary, #1f3b35);">'
        'Token importance (SHAP occlusion)</h4>',
    ]

    with torch.no_grad():
        # Baseline: full-sequence probabilities
        base_logits = model(input_ids, attention_mask)
        base_probs = torch.softmax(base_logits, dim=1)

        for i in range(input_ids.size(0)):
            ids_i = input_ids[i]               # (seq_len,)
            mask_i = attention_mask[i] if attention_mask is not None else torch.ones_like(ids_i)
            real_len = int(mask_i.sum().item())
            pred_cls = int(base_probs[i].argmax().item())
            pred_conf = float(base_probs[i, pred_cls].item())
            true_cls = int(labels[i].item())

            # Per-token occlusion: replace each token with pad (0) and re-run
            ids_batch = ids_i.unsqueeze(0).expand(real_len, -1).clone()  # (real_len, seq_len)
            mask_batch = mask_i.unsqueeze(0).expand(real_len, -1).clone()
            for t in range(real_len):
                ids_batch[t, t] = 0          # mask this token
                mask_batch[t, t] = 0

            occ_logits = model(ids_batch, mask_batch)
            occ_probs = torch.softmax(occ_logits, dim=1)

            # Importance = drop in predicted-class probability when token is removed
            importance = (pred_conf - occ_probs[:, pred_cls]).cpu().numpy()  # (real_len,)

            # Decode tokens
            tokens = []
            for t in range(real_len):
                tid = int(ids_i[t].item())
                if vocab_map:
                    tokens.append(idx_to_word.get(tid, f"[{tid}]"))
                else:
                    tokens.append(_tok.decode([tid], skip_special_tokens=False).strip())

            # Normalize importance to [0, 1] for opacity
            abs_imp = np.abs(importance)
            max_imp = abs_imp.max() if abs_imp.max() > 0 else 1.0

            # Build HTML spans
            pred_label = classes[pred_cls] if pred_cls < len(classes) else str(pred_cls)
            true_label = classes[true_cls] if true_cls < len(classes) else str(true_cls)
            status = "✓" if pred_cls == true_cls else "✗"

            html_parts.append(
                f'<div style="margin-bottom: 16px; padding: 12px; border: 1px solid var(--color-border, #d7e5db); '
                f'border-radius: 12px; background: var(--color-surface, #fff);">'
                f'<div style="font-size: 0.8rem; margin-bottom: 8px; color: var(--color-text-muted, #6a857c);">'
                f'{status} True: <b>{html_mod.escape(true_label)}</b> · '
                f'Predicted: <b>{html_mod.escape(pred_label)}</b> '
                f'({pred_conf:.0%})</div>'
                f'<div style="line-height: 2;">'
            )
            for t, token in enumerate(tokens):
                opacity = float(abs_imp[t] / max_imp) * 0.7  # cap at 0.7 opacity
                # Positive importance (token helps prediction) = red
                # Negative importance (token hurts prediction) = blue
                color = "220,38,38" if importance[t] >= 0 else "37,99,235"
                safe_token = html_mod.escape(token)
                html_parts.append(
                    f'<span style="background: rgba({color},{opacity:.2f}); '
                    f'padding: 2px 4px; border-radius: 4px; margin: 1px;" '
                    f'title="importance: {importance[t]:.4f}">{safe_token}</span> '
                )
            html_parts.append('</div></div>')

    html_parts.append(
        '<p style="font-size: 0.75rem; color: var(--color-text-muted, #6a857c); margin-top: 8px;">'
        'Red = token supports the prediction · Blue = token opposes it · '
        'Opacity = magnitude · Hover for exact values</p>'
    )
    html_parts.append('</div>')

    return "".join(html_parts)

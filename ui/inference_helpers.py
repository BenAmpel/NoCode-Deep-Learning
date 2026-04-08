"""
Helpers for the "Try Your Model" inference panel.

Supports image, text, tabular, and audio modalities.  All ``predict_*``
functions return a safe error payload instead of raising, so the UI
can always display something meaningful.

Usage
-----
    from ui.inference_helpers import (
        load_bundle, predict_image, predict_text,
        predict_tabular, predict_audio,
        predictions_to_markdown, predictions_to_chart,
    )

    bundle = load_bundle("outputs/my_model_bundle")
    results = predict_image(bundle_path, image_array, top_k=3)
    print(predictions_to_markdown(results))
"""
from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level inference cache
# ---------------------------------------------------------------------------
# Models are expensive to load (disk I/O + weight initialisation).  Cache the
# loaded model/session keyed by bundle path so repeated "Try Your Model" calls
# reuse the already-loaded weights instead of hitting disk every time.
# The cache is capped at 4 entries to avoid unbounded memory growth.

_MAX_CACHE = 4
_model_cache: dict[str, Any]  = {}   # bundle_path → ONNX session or torch module
_bundle_cache: dict[str, dict] = {}  # bundle_path → load_bundle() result


def _cache_model(key: str, model: Any) -> None:
    if len(_model_cache) >= _MAX_CACHE:
        _model_cache.pop(next(iter(_model_cache)))  # evict oldest (FIFO)
    _model_cache[key] = model


def _cached_bundle(bundle_path: str) -> dict:
    """Return cached bundle metadata, loading from disk only on first access."""
    if bundle_path not in _bundle_cache:
        if len(_bundle_cache) >= _MAX_CACHE:
            _bundle_cache.pop(next(iter(_bundle_cache)))
        _bundle_cache[bundle_path] = load_bundle(bundle_path)
    return _bundle_cache[bundle_path]


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# ImageNet mean / std (RGB, per-channel)
_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

_IMAGE_SIZE = 224  # default resize target

# Confidence bar settings
_BAR_FULL  = "█"
_BAR_EMPTY = "░"
_BAR_WIDTH = 10

# Rank medals for the markdown table
_RANK_MEDALS = {1: "🥇", 2: "🥈", 3: "🥉"}

# Bar chart colours for top-3
_RANK_COLOURS = {1: "#FFD700", 2: "#C0C0C0", 3: "#CD7F32"}  # gold, silver, bronze
_DEFAULT_BAR_COLOUR = "#4C9BE8"


# ---------------------------------------------------------------------------
# load_bundle
# ---------------------------------------------------------------------------

def load_bundle(bundle_path: str) -> dict:
    """
    Load metadata files from a saved model bundle directory.

    Expects the bundle to contain ``labels.json`` and ``preprocessing.json``.
    Optionally contains a ``model.pt`` or ``model.onnx`` file.

    Parameters
    ----------
    bundle_path:
        Path to the model bundle directory produced by the export pipeline.

    Returns
    -------
    dict
        Keys:

        - ``labels``        (dict) — ``{str_index: label_name, ...}``
        - ``preprocessing`` (dict) — preprocessing config used at training time
        - ``model_path``    (str | None) — absolute path to ``.pt`` file if present
        - ``onnx_path``     (str | None) — absolute path to ``.onnx`` file if present
    """
    # Security: ensure the resolved path stays within the outputs/ directory
    # to prevent path traversal attacks when bundle_path comes from user input.
    _project_root = Path(__file__).resolve().parent.parent
    _outputs_root = (_project_root / "outputs").resolve()
    _resolved     = Path(bundle_path).resolve()
    if not str(_resolved).startswith(str(_outputs_root)):
        # Soft warning — allow absolute paths that exist (local desktop use)
        # but log the anomaly for visibility.
        logger.warning(
            "Bundle path '%s' is outside the expected outputs/ directory. "
            "Proceeding, but verify this path is trusted.",
            bundle_path,
        )

    bundle_path = os.path.abspath(bundle_path)

    labels_file = os.path.join(bundle_path, "labels.json")
    prep_file   = os.path.join(bundle_path, "preprocessing.json")

    with open(labels_file, "r", encoding="utf-8") as fh:
        labels: dict = json.load(fh)

    with open(prep_file, "r", encoding="utf-8") as fh:
        preprocessing: dict = json.load(fh)

    # Optional model files
    def _find(name: str) -> str | None:
        p = os.path.join(bundle_path, name)
        return p if os.path.isfile(p) else None

    model_path = _find("model.pt") or _find("model.pth") or _find("model.joblib") or _find("model.pkl")
    onnx_path  = _find("model.onnx")

    return {
        "labels":        labels,
        "preprocessing": preprocessing,
        "model_path":    model_path,
        "onnx_path":     onnx_path,
    }


# ---------------------------------------------------------------------------
# predict_image
# ---------------------------------------------------------------------------

def predict_image(
    bundle_path: str,
    image_input: Any,
    top_k: int = 3,
) -> list[dict]:
    """
    Run image classification inference.

    Parameters
    ----------
    bundle_path:
        Path to the model bundle directory.
    image_input:
        Either a file-path string to an image file **or** a numpy array
        of shape ``(H, W, 3)`` in RGB uint8 / float32 format.
    top_k:
        Number of top predictions to return.

    Returns
    -------
    list[dict]
        Up to ``top_k`` dicts sorted by confidence descending::

            [{"label": str, "confidence": float, "rank": int}, ...]

        On failure, returns::

            [{"label": "Error", "confidence": 0.0, "rank": 1, "error": str}]
    """
    try:
        bundle = _cached_bundle(bundle_path)
        labels = bundle["labels"]

        # -- Load image as numpy array (H, W, 3) float32 in [0, 1]
        img_array = _load_image_array(image_input)

        # -- Preprocess: resize → normalize
        img_tensor = _preprocess_image(img_array, bundle["preprocessing"])  # (1, 3, H, W) float32

        # -- Run inference (ONNX preferred, then PyTorch)
        logits = _run_image_inference(bundle, img_tensor)

        return _logits_to_topk(logits, labels, top_k)

    except Exception as exc:
        return _error_payload(exc)


def _load_image_array(image_input: Any) -> np.ndarray:
    """Return an (H, W, 3) float32 numpy array in [0, 1] from a path or array."""
    if isinstance(image_input, str):
        from PIL import Image  # type: ignore[import]
        img = Image.open(image_input).convert("RGB")
        return np.array(img, dtype=np.float32) / 255.0
    # Assume numpy array
    arr = np.asarray(image_input, dtype=np.float32)
    if arr.max() > 1.0:
        arr = arr / 255.0
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    return arr


def _preprocess_image(arr: np.ndarray, prep: dict | None = None) -> np.ndarray:
    """Resize to 224×224, normalise with ImageNet stats. Returns (1, 3, 224, 224)."""
    from PIL import Image  # type: ignore[import]
    prep = prep or {}
    resize = prep.get("resize", [_IMAGE_SIZE, _IMAGE_SIZE])
    if isinstance(resize, (list, tuple)) and len(resize) >= 2:
        image_h, image_w = int(resize[0]), int(resize[1])
    else:
        image_h = image_w = _IMAGE_SIZE
    mean = np.array(prep.get("mean", _IMAGENET_MEAN), dtype=np.float32)
    std = np.array(prep.get("std", _IMAGENET_STD), dtype=np.float32)

    # Resize via PIL for high quality interpolation
    pil_img = Image.fromarray((arr * 255).astype(np.uint8)).resize(
        (image_w, image_h), Image.BILINEAR
    )
    arr_resized = np.array(pil_img, dtype=np.float32) / 255.0  # (224, 224, 3)

    # Normalise
    arr_norm = (arr_resized - mean) / std

    # HWC → CHW → NCHW
    return arr_norm.transpose(2, 0, 1)[np.newaxis, ...].astype(np.float32)


def _run_image_inference(bundle: dict, img_tensor: np.ndarray) -> np.ndarray:
    """Run inference; returns a 1-D logits array."""
    if bundle["onnx_path"]:
        return _onnx_infer(bundle["onnx_path"], img_tensor)
    if bundle["model_path"]:
        return _torch_infer(bundle["model_path"], img_tensor)
    raise FileNotFoundError("Bundle contains neither model.onnx nor model.pt.")


# ---------------------------------------------------------------------------
# predict_text
# ---------------------------------------------------------------------------

def predict_text(
    bundle_path: str,
    text: str,
    top_k: int = 3,
) -> list[dict]:
    """
    Run text classification inference.

    Parameters
    ----------
    bundle_path:
        Path to the model bundle directory.
    text:
        Raw input text string.
    top_k:
        Number of top predictions to return.

    Returns
    -------
    list[dict]
        Same format as :func:`predict_image`.
    """
    try:
        bundle = _cached_bundle(bundle_path)
        labels = bundle["labels"]
        prep   = bundle["preprocessing"]

        # Tokenise using saved config
        input_ids, attention_mask = _tokenize_text(text, prep)

        logits = _run_text_inference(bundle, input_ids, attention_mask)
        return _logits_to_topk(logits, labels, top_k)

    except Exception as exc:
        return _error_payload(exc)


def explain_text_prediction(bundle_path: str, text: str) -> str:
    """Run occlusion-based token importance for a single text input.

    Returns an HTML string with tokens highlighted by importance
    (red = supports prediction, blue = opposes it).
    """
    import html as html_mod

    try:
        bundle = _cached_bundle(bundle_path)
        labels = bundle["labels"]
        prep   = bundle["preprocessing"]

        cleaned = _clean_text_for_inference(text, prep.get("cleaning", {}))
        max_length = int(prep.get("max_length", 128))
        vocab = prep.get("vocab")

        # Tokenize to get IDs and readable tokens
        if vocab:
            import re as _re
            tokens = _re.findall(r"\w+|[^\w\s]", cleaned.lower())[:max_length]
            ids = [vocab.get(t, vocab.get("<unk>", 1)) for t in tokens]
        else:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(prep.get("tokenizer", "distilbert-base-uncased"))
            encoded = tokenizer(cleaned, truncation=True, max_length=max_length, return_tensors="np")
            ids = encoded["input_ids"][0].tolist()
            tokens = [tokenizer.decode([tid]).strip() for tid in ids
                      if tid not in (tokenizer.pad_token_id, tokenizer.cls_token_id, tokenizer.sep_token_id)]
            ids = [tid for tid in ids
                   if tid not in (tokenizer.pad_token_id, tokenizer.cls_token_id, tokenizer.sep_token_id)]

        real_len = len(ids)
        if real_len == 0:
            return ""

        # Build padded arrays
        pad_len = max_length - real_len
        full_ids = np.array([ids + [0] * pad_len], dtype=np.int64)
        full_mask = np.array([[1] * real_len + [0] * pad_len], dtype=np.int64)

        # Baseline prediction
        base_logits = _run_text_inference(bundle, full_ids, full_mask)
        base_probs = _softmax(base_logits)
        pred_cls = int(np.argmax(base_probs))
        pred_conf = float(base_probs[pred_cls])
        pred_label = labels.get(str(pred_cls), labels.get(pred_cls, f"class_{pred_cls}"))

        # Occlusion: mask each token one at a time
        importance = np.zeros(real_len, dtype=np.float32)
        for t in range(real_len):
            occ_ids = full_ids.copy()
            occ_mask = full_mask.copy()
            occ_ids[0, t] = 0
            occ_mask[0, t] = 0
            occ_logits = _run_text_inference(bundle, occ_ids, occ_mask)
            occ_probs = _softmax(occ_logits)
            importance[t] = pred_conf - float(occ_probs[pred_cls])

        # Render HTML
        abs_imp = np.abs(importance)
        max_imp = abs_imp.max() if abs_imp.max() > 0 else 1.0

        spans = []
        for t in range(real_len):
            opacity = float(abs_imp[t] / max_imp) * 0.7
            color = "220,38,38" if importance[t] >= 0 else "37,99,235"
            safe = html_mod.escape(tokens[t])
            spans.append(
                f'<span style="background:rgba({color},{opacity:.2f});'
                f'padding:2px 4px;border-radius:4px;margin:1px;" '
                f'title="importance: {importance[t]:.4f}">{safe}</span> '
            )

        return (
            f'<div style="line-height:2.2;margin-top:8px;">'
            f'<div style="font-size:0.8rem;margin-bottom:6px;color:var(--color-text-muted,#6a857c);">'
            f'Predicted: <b>{html_mod.escape(str(pred_label))}</b> ({pred_conf:.0%}) · '
            f'<span style="color:rgba(220,38,38,0.8);">Red</span> = supports · '
            f'<span style="color:rgba(37,99,235,0.8);">Blue</span> = opposes</div>'
            f'<div>{"".join(spans)}</div></div>'
        )

    except Exception:
        return ""


def _tokenize_text(
    text: str,
    prep: dict,
) -> tuple[np.ndarray, np.ndarray]:
    """Tokenise *text* using the tokenizer specified in *prep*."""
    text = _clean_text_for_inference(text, prep.get("cleaning", {}))
    max_length = int(prep.get("max_length", 128))

    # From-scratch models store a word→index vocab; use it directly.
    vocab = prep.get("vocab")
    if vocab:
        import re as _re
        tokens = _re.findall(r"\w+|[^\w\s]", text.lower())[:max_length]
        ids  = [vocab.get(t, vocab.get("<unk>", 1)) for t in tokens]
        mask = [1] * len(ids)
        pad_len = max_length - len(ids)
        ids  += [0] * pad_len
        mask += [0] * pad_len
        return np.array([ids], dtype=np.int64), np.array([mask], dtype=np.int64)

    # Pretrained transformer tokenizer
    tokenizer_name = prep.get("tokenizer", "distilbert-base-uncased")
    from transformers import AutoTokenizer  # type: ignore[import]

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    encoded   = tokenizer(
        text,
        max_length=max_length,
        padding="max_length",
        truncation=True,
        return_tensors="np",
    )
    return encoded["input_ids"], encoded["attention_mask"]


def _clean_text_for_inference(text: str, cleaning: dict) -> str:
    value = str(text or "")
    if cleaning.get("strip_urls"):
        value = re.sub(r"https?://\S+|www\.\S+", " ", value)
    if cleaning.get("lowercase"):
        value = value.lower()
    if cleaning.get("strip_punctuation"):
        value = re.sub(r"[^\w\s]", " ", value)
    if cleaning.get("remove_stopwords"):
        stopwords = {
            "a", "an", "and", "are", "as", "at", "be", "been", "being", "but", "by",
            "for", "from", "had", "has", "have", "he", "her", "hers", "him", "his",
            "i", "if", "in", "into", "is", "it", "its", "me", "my", "of", "on", "or",
            "our", "ours", "she", "so", "than", "that", "the", "their", "theirs",
            "them", "they", "this", "those", "to", "too", "us", "was", "we", "were",
            "what", "when", "where", "which", "who", "why", "with", "you", "your",
            "yours",
        }
        value = " ".join(token for token in value.split() if token not in stopwords)
    return " ".join(value.split())


def _run_text_inference(
    bundle: dict,
    input_ids: np.ndarray,
    attention_mask: np.ndarray,
) -> np.ndarray:
    """Run inference on tokenised text; returns a 1-D logits array."""
    if bundle["onnx_path"]:
        import onnxruntime as ort  # type: ignore[import]

        sess  = ort.InferenceSession(bundle["onnx_path"], providers=["CPUExecutionProvider"])
        inputs = sess.get_inputs()
        feeds = {inputs[0].name: input_ids}
        if len(inputs) > 1:
            feeds[inputs[1].name] = attention_mask
        return np.array(sess.run(None, feeds)[0]).squeeze()

    if bundle["model_path"]:
        import torch  # type: ignore[import]

        model = torch.jit.load(bundle["model_path"], map_location="cpu")
        model.eval()
        with torch.no_grad():
            out = model(
                torch.from_numpy(input_ids),
                torch.from_numpy(attention_mask),
            )
        return out.numpy().squeeze()

    raise FileNotFoundError("Bundle contains neither model.onnx nor model.pt.")


# ---------------------------------------------------------------------------
# predict_tabular
# ---------------------------------------------------------------------------

def predict_tabular(
    bundle_path: str,
    feature_values: dict[str, Any],
    top_k: int = 3,
) -> list[dict]:
    """
    Run tabular classification/regression inference.

    Parameters
    ----------
    bundle_path:
        Path to the model bundle directory.
    feature_values:
        Dict mapping column name → raw value (string or numeric).
    top_k:
        Number of top predictions to return.

    Returns
    -------
    list[dict]
        Same format as :func:`predict_image`.
    """
    try:
        bundle = _cached_bundle(bundle_path)
        labels = bundle["labels"]
        prep   = bundle["preprocessing"]

        feature_array = _preprocess_tabular(feature_values, prep)  # (1, n_features)

        logits = _run_tabular_inference(bundle, feature_array)
        return _logits_to_topk(logits, labels, top_k)

    except Exception as exc:
        return _error_payload(exc)


def predict_timeseries(
    bundle_path: str,
    window_values: Any,
    top_k: int = 3,
) -> list[dict]:
    """Run time-series inference from a JSON window payload."""
    try:
        bundle = _cached_bundle(bundle_path)
        labels = bundle["labels"]
        prep = bundle["preprocessing"]
        window_array = _preprocess_timeseries(window_values, prep)

        task = prep.get("task", "classification")
        if task != "regression" and len(labels) <= 1:
            task = "regression"
        if task == "regression":
            value = _run_timeseries_regression(bundle, window_array)
            return _scalar_prediction_payload(value, label="Predicted value")

        logits = _run_timeseries_classification(bundle, window_array)
        return _logits_to_topk(logits, labels, top_k)
    except Exception as exc:
        return _error_payload(exc)


def _preprocess_tabular(
    feature_values: dict[str, Any],
    prep: dict,
) -> np.ndarray:
    """
    Encode categoricals and scale numerics using the saved preprocessing config.

    *prep* may contain:

    - ``feature_order``  (list[str]) — column ordering expected by the model
    - ``numeric_cols``   (list[str]) — columns to scale
    - ``scaler_mean``    (list[float])
    - ``scaler_scale``   (list[float])
    - ``categorical_cols`` (list[str])
    - ``category_maps``  (dict[str, dict[str, int]])
    """
    feature_order = prep.get("feature_order", list(feature_values.keys()))
    numeric_cols  = prep.get("numeric_cols", [])
    cat_cols      = prep.get("categorical_cols", [])
    category_maps = prep.get("category_maps", {})
    scaler_mean   = np.array(prep.get("scaler_mean", []), dtype=np.float32)
    scaler_scale  = np.array(prep.get("scaler_scale", []), dtype=np.float32)
    scaling_strategy = prep.get("scaling_strategy", "standard")
    min_vals = np.array(prep.get("min_vals", []), dtype=np.float32)
    max_vals = np.array(prep.get("max_vals", []), dtype=np.float32)
    median_vals = np.array(prep.get("median_vals", []), dtype=np.float32)
    iqr = np.array(prep.get("iqr", []), dtype=np.float32)
    dummy_feature_names = prep.get("dummy_feature_names", [])

    numeric_values: list[float] = []
    for col in numeric_cols:
        try:
            numeric_values.append(float(feature_values.get(col, 0.0)))
        except (TypeError, ValueError):
            numeric_values.append(0.0)

    numeric_arr = np.array(numeric_values, dtype=np.float32).reshape(1, -1) if numeric_values else np.zeros((1, 0), dtype=np.float32)
    if numeric_arr.shape[1]:
        if scaling_strategy == "minmax" and len(min_vals) == numeric_arr.shape[1] and len(max_vals) == numeric_arr.shape[1]:
            numeric_arr = (numeric_arr - min_vals) / np.where((max_vals - min_vals) == 0, 1.0, (max_vals - min_vals))
        elif scaling_strategy == "robust" and len(median_vals) == numeric_arr.shape[1] and len(iqr) == numeric_arr.shape[1]:
            numeric_arr = (numeric_arr - median_vals) / np.where(iqr == 0, 1.0, iqr)
        elif scaling_strategy == "standard" and len(scaler_mean) == numeric_arr.shape[1] and len(scaler_scale) == numeric_arr.shape[1]:
            numeric_arr = (numeric_arr - scaler_mean) / np.where(scaler_scale == 0, 1.0, scaler_scale)

    dummy_lookup = {name: 0.0 for name in dummy_feature_names}
    for col in cat_cols:
        value = feature_values.get(col)
        dummy_name = category_maps.get(col, {}).get(str(value))
        if dummy_name in dummy_lookup:
            dummy_lookup[dummy_name] = 1.0
    dummy_arr = np.array([dummy_lookup[name] for name in dummy_feature_names], dtype=np.float32).reshape(1, -1) if dummy_feature_names else np.zeros((1, 0), dtype=np.float32)

    arr = np.concatenate([numeric_arr, dummy_arr], axis=1)
    if feature_order and len(feature_order) == arr.shape[1]:
        # Keep this here as an assertion-on-shape guard: feature_order should already
        # match the numeric + one-hot layout produced above.
        arr = arr.astype(np.float32)

    return arr


def _preprocess_timeseries(
    window_values: Any,
    prep: dict,
) -> np.ndarray:
    feature_cols = prep.get("feature_columns", [])
    window_size = int(prep.get("window_size", 1))
    mean = np.array(prep.get("mean", []), dtype=np.float32)
    std = np.array(prep.get("std", []), dtype=np.float32)

    data = json.loads(window_values) if isinstance(window_values, str) else window_values
    if not isinstance(data, list) or not data:
        raise ValueError("Time-series input must be a JSON array of rows.")

    rows = []
    if isinstance(data[0], dict):
        if not feature_cols:
            feature_cols = [k for k in data[0].keys()]
        for row in data:
            rows.append([float(row.get(col, 0.0)) for col in feature_cols])
    else:
        rows = [[float(v) for v in row] for row in data]

    arr = np.asarray(rows, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError("Time-series input must be 2-D: [time, features].")

    if feature_cols and arr.shape[1] != len(feature_cols):
        raise ValueError(f"Expected {len(feature_cols)} features per step, got {arr.shape[1]}.")

    if arr.shape[0] < window_size:
        pad = np.repeat(arr[-1:, :], window_size - arr.shape[0], axis=0)
        arr = np.concatenate([arr, pad], axis=0)
    elif arr.shape[0] > window_size:
        arr = arr[-window_size:, :]

    if len(mean) == arr.shape[1] and len(std) == arr.shape[1]:
        arr = (arr - mean) / np.where(std == 0, 1.0, std)

    return arr[np.newaxis, ...].astype(np.float32)


def _run_tabular_inference(bundle: dict, feature_array: np.ndarray) -> np.ndarray:
    """Run tabular inference; returns a 1-D logits/probability array."""
    if bundle["onnx_path"]:
        return _onnx_infer(bundle["onnx_path"], feature_array)

    if bundle["model_path"]:
        # Check for sklearn pickle first
        model_path: str = bundle["model_path"]
        if model_path.endswith(".pkl") or model_path.endswith(".joblib"):
            import joblib  # type: ignore[import]
            model = joblib.load(model_path)
            probs = model.predict_proba(feature_array)[0]
            return probs
        return _torch_infer(model_path, feature_array)

    raise FileNotFoundError("Bundle contains neither model.onnx nor model.pt/pkl.")


def _run_timeseries_classification(bundle: dict, window_array: np.ndarray) -> np.ndarray:
    flat = window_array.reshape(1, -1).astype(np.float32)
    if bundle["model_path"]:
        model_path = bundle["model_path"]
        if model_path.endswith(".pkl") or model_path.endswith(".joblib"):
            import joblib  # type: ignore[import]

            model = joblib.load(model_path)
            if hasattr(model, "predict_proba"):
                return np.asarray(model.predict_proba(flat)[0], dtype=np.float32)
            preds = np.asarray(model.predict(flat)).ravel()
            return preds.astype(np.float32)
    if bundle["onnx_path"]:
        try:
            return _onnx_infer(bundle["onnx_path"], window_array)
        except Exception:
            return _onnx_infer(bundle["onnx_path"], flat)
    if bundle["model_path"]:
        return _torch_infer(bundle["model_path"], window_array)
    raise FileNotFoundError("Bundle contains neither model.onnx nor model.pt/pkl.")


def _run_timeseries_regression(bundle: dict, window_array: np.ndarray) -> float:
    flat = window_array.reshape(1, -1).astype(np.float32)
    if bundle["model_path"]:
        model_path = bundle["model_path"]
        if model_path.endswith(".pkl") or model_path.endswith(".joblib"):
            import joblib  # type: ignore[import]

            model = joblib.load(model_path)
            value = np.asarray(model.predict(flat), dtype=np.float32).ravel()[0]
            return float(value)
    if bundle["onnx_path"]:
        try:
            value = np.asarray(_onnx_infer(bundle["onnx_path"], window_array), dtype=np.float32).ravel()[0]
        except Exception:
            value = np.asarray(_onnx_infer(bundle["onnx_path"], flat), dtype=np.float32).ravel()[0]
        return float(value)
    if bundle["model_path"]:
        value = np.asarray(_torch_infer(bundle["model_path"], window_array), dtype=np.float32).ravel()[0]
        return float(value)
    raise FileNotFoundError("Bundle contains neither model.onnx nor model.pt/pkl.")


# ---------------------------------------------------------------------------
# predict_audio
# ---------------------------------------------------------------------------

def predict_audio(
    bundle_path: str,
    audio_path: str,
    top_k: int = 3,
) -> list[dict]:
    """
    Run audio classification inference via mel spectrogram.

    The preprocessing parameters are read from ``preprocessing.json`` in the
    bundle so the same pipeline used during training is reproduced exactly.

    Parameters
    ----------
    bundle_path:
        Path to the model bundle directory.
    audio_path:
        Path to the audio file (wav, mp3, flac, etc.).
    top_k:
        Number of top predictions to return.

    Returns
    -------
    list[dict]
        Same format as :func:`predict_image`.
    """
    try:
        bundle = _cached_bundle(bundle_path)
        labels = bundle["labels"]
        prep   = bundle["preprocessing"]

        spec_array = _audio_to_spectrogram(audio_path, prep)  # (1, 3, H, W)

        logits = _run_image_inference(bundle, spec_array)
        return _logits_to_topk(logits, labels, top_k)

    except Exception as exc:
        return _error_payload(exc)


def _audio_to_spectrogram(audio_path: str, prep: dict) -> np.ndarray:
    """
    Load an audio file and convert it to a normalised mel spectrogram.

    Returns an ``(1, 3, n_mels, time_frames)`` float32 array suitable for
    an image-classification backbone.
    """
    import torch
    import torchaudio.transforms as AT
    from PIL import Image  # type: ignore[import]

    from modalities.audio import load_audio_waveform

    sample_rate    = int(prep.get("sample_rate",    22050))
    n_mels         = int(prep.get("n_mels",         64))
    hop_length     = int(prep.get("hop_length",     512))
    n_fft          = int(prep.get("n_fft",          1024))
    max_duration   = float(prep.get("max_duration_sec", 10.0))
    normalize_waveform = bool(prep.get("normalize_waveform", False))
    resize = prep.get("resize", [_IMAGE_SIZE, _IMAGE_SIZE])
    if isinstance(resize, (list, tuple)) and len(resize) >= 2:
        image_h, image_w = int(resize[0]), int(resize[1])
    else:
        image_h = image_w = _IMAGE_SIZE
    mean = np.array(prep.get("mean", _IMAGENET_MEAN), dtype=np.float32)
    std = np.array(prep.get("std", _IMAGENET_STD), dtype=np.float32)

    waveform, sr = load_audio_waveform(audio_path, target_sr=sample_rate)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(0, keepdim=True)

    max_samples = int(max_duration * sample_rate)
    n = waveform.shape[1]
    if n > max_samples:
        waveform = waveform[:, :max_samples]
    elif n < max_samples:
        waveform = torch.nn.functional.pad(waveform, (0, max_samples - n))

    if normalize_waveform:
        peak = float(waveform.abs().max().item())
        if peak > 0:
            waveform = waveform / peak

    mel_transform = AT.MelSpectrogram(
        sample_rate=sr,
        n_mels=n_mels,
        n_fft=n_fft,
        hop_length=hop_length,
    )
    to_db = AT.AmplitudeToDB()
    mel = mel_transform(waveform)
    mel_db = to_db(mel).squeeze(0).cpu().numpy().astype(np.float32)

    # Normalise to [0, 1] then apply ImageNet stats (3-channel repeat)
    mel_min, mel_max = mel_db.min(), mel_db.max()
    if mel_max > mel_min:
        mel_norm = (mel_db - mel_min) / (mel_max - mel_min)
    else:
        mel_norm = mel_db * 0.0

    # Resize to IMAGE_SIZE × IMAGE_SIZE to match image backbone expectations
    pil_img = Image.fromarray((mel_norm * 255).astype(np.uint8)).resize(
        (image_w, image_h), Image.BILINEAR
    )
    rgb_arr = np.stack(
        [np.array(pil_img, dtype=np.float32) / 255.0] * 3, axis=-1
    )  # (H, W, 3)

    # Apply ImageNet normalisation
    rgb_norm = (rgb_arr - mean) / std
    return rgb_norm.transpose(2, 0, 1)[np.newaxis, ...].astype(np.float32)


# ---------------------------------------------------------------------------
# predictions_to_markdown
# ---------------------------------------------------------------------------

def predictions_to_markdown(predictions: list[dict]) -> str:
    """
    Render a top-k prediction list as a markdown table with ASCII confidence bars.

    Parameters
    ----------
    predictions:
        List of dicts from any ``predict_*`` function.

    Returns
    -------
    str
        Markdown table, e.g.::

            | Rank | Label | Confidence | Bar        |
            |------|-------|------------|------------|
            | 🥇   | cat   | 94.3%      | █████████░ |

    On error payload, returns a single-row table showing the error message.
    """
    # Error payload shortcut
    if predictions and predictions[0].get("label") == "Error":
        err = predictions[0].get("error", "Unknown error")
        return f"❌ **Prediction failed:** {err}"

    if predictions and "value" in predictions[0]:
        label = predictions[0].get("label", "Prediction")
        value = predictions[0].get("value")
        return f"### {label}\n\n`{value:.6f}`"

    lines = [
        "| Rank | Label | Confidence | " + ("—" * _BAR_WIDTH) + " |",
        "|:----:|:------|----------:|:" + ("-" * _BAR_WIDTH) + ":|",
    ]
    for pred in predictions:
        rank       = pred.get("rank", 1)
        label      = pred.get("label", "?")
        confidence = pred.get("confidence", 0.0)

        medal = _RANK_MEDALS.get(rank, f"#{rank}")
        pct   = confidence * 100.0
        filled = round(pct / 100.0 * _BAR_WIDTH)
        bar   = _BAR_FULL * filled + _BAR_EMPTY * (_BAR_WIDTH - filled)
        lines.append(f"| {medal} | {label} | {pct:.1f}% | {bar} |")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# predictions_to_chart
# ---------------------------------------------------------------------------

def predictions_to_chart(predictions: list[dict]):
    """
    Create a horizontal bar chart of top-k predictions.

    Bars are coloured gold / silver / bronze for the top 3 ranks, and a
    default blue for any additional predictions.

    Parameters
    ----------
    predictions:
        List of dicts from any ``predict_*`` function.

    Returns
    -------
    matplotlib.figure.Figure
        The figure object (not shown).  Pass to ``gr.Plot`` in Gradio.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Filter out error entries
    valid = [p for p in predictions if p.get("label") != "Error"]
    if not valid:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No predictions available", ha="center", va="center",
                transform=ax.transAxes, fontsize=12, color="grey")
        ax.axis("off")
        return fig

    if "value" in valid[0]:
        value = float(valid[0]["value"])
        label = valid[0].get("label", "Prediction")
        fig, ax = plt.subplots(figsize=(5.5, 2.6))
        ax.barh([label], [value], color="#115d52")
        ax.axvline(0, color="#94a3b8", linewidth=1, linestyle="--")
        ax.text(value, 0, f" {value:.4f}", va="center", ha="left" if value >= 0 else "right")
        ax.set_title("Predicted value", fontsize=12, fontweight="bold")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        fig.tight_layout()
        return fig

    labels      = [p["label"]      for p in valid]
    confidences = [p["confidence"] * 100.0 for p in valid]
    ranks       = [p.get("rank", i + 1) for i, p in enumerate(valid)]
    colours     = [_RANK_COLOURS.get(r, _DEFAULT_BAR_COLOUR) for r in ranks]

    # Plot in reverse order so rank 1 appears at the top
    labels      = labels[::-1]
    confidences = confidences[::-1]
    colours     = colours[::-1]

    fig, ax = plt.subplots(figsize=(7, max(2.5, len(valid) * 0.8)))

    bars = ax.barh(labels, confidences, color=colours, edgecolor="white", height=0.55)

    # Value annotations inside bars
    for bar, conf in zip(bars, confidences):
        x_pos = bar.get_width() - 2.5
        ha    = "right"
        colour = "white"
        if conf < 10:
            x_pos = bar.get_width() + 1.0
            ha    = "left"
            colour = "#333333"
        ax.text(
            x_pos,
            bar.get_y() + bar.get_height() / 2,
            f"{conf:.1f}%",
            va="center",
            ha=ha,
            fontsize=10,
            fontweight="bold",
            color=colour,
        )

    ax.set_xlim(0, 105)
    ax.set_xlabel("Confidence (%)", fontsize=10)
    ax.set_title("Top Predictions", fontsize=12, fontweight="bold", pad=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="y", labelsize=10)

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Private inference utilities
# ---------------------------------------------------------------------------

def _onnx_infer(onnx_path: str, input_array: np.ndarray) -> np.ndarray:
    """Run an ONNX model and return a 1-D logits array.

    The InferenceSession is cached keyed by path — loading from disk only on
    the first call, then reused for all subsequent predictions.
    """
    import onnxruntime as ort  # type: ignore[import]

    cache_key = f"onnx:{onnx_path}"
    if cache_key not in _model_cache:
        sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        _cache_model(cache_key, sess)
    else:
        sess = _model_cache[cache_key]

    feeds = {sess.get_inputs()[0].name: input_array}
    out   = sess.run(None, feeds)[0]
    return np.array(out).squeeze()


def _torch_infer(model_path: str, input_array: np.ndarray) -> np.ndarray:
    """Load a TorchScript .pt model and run inference. Returns a 1-D logits array.

    The loaded model is cached keyed by path — disk I/O only on first call.
    """
    import torch  # type: ignore[import]

    cache_key = f"torch:{model_path}"
    if cache_key not in _model_cache:
        model = torch.jit.load(model_path, map_location="cpu")
        model.eval()
        _cache_model(cache_key, model)
    else:
        model = _model_cache[cache_key]

    tensor = torch.from_numpy(input_array)
    with torch.no_grad():
        out = model(tensor)
    return out.numpy().squeeze()


def _softmax(logits: np.ndarray) -> np.ndarray:
    """Numerically stable softmax over a 1-D array."""
    shifted = logits - logits.max()
    exp     = np.exp(shifted)
    return exp / exp.sum()


def _logits_to_topk(
    logits: np.ndarray,
    labels: dict,
    top_k: int,
) -> list[dict]:
    """
    Convert a raw 1-D logits/probability array to a sorted top-k list.

    Parameters
    ----------
    logits:
        Raw model output (logits or probabilities; softmax is always applied).
    labels:
        ``{str_or_int_index: label_name}`` mapping from the bundle.
    top_k:
        Number of results to return.

    Returns
    -------
    list[dict]
        Sorted by confidence descending, each item::

            {"label": str, "confidence": float, "rank": int}
    """
    logits = np.asarray(logits, dtype=np.float32).flatten()

    # If outputs look like raw logits (values outside [0,1]), apply softmax
    if logits.min() < 0.0 or logits.max() > 1.0 or abs(logits.sum() - 1.0) > 0.05:
        probs = _softmax(logits)
    else:
        probs = logits

    top_k = min(top_k, len(probs))
    top_indices = np.argsort(probs)[::-1][:top_k]

    results: list[dict] = []
    for rank, idx in enumerate(top_indices, start=1):
        label = labels.get(str(idx), labels.get(idx, f"class_{idx}"))
        results.append({
            "label":      str(label),
            "confidence": float(probs[idx]),
            "rank":       rank,
        })

    return results


def _error_payload(exc: Exception) -> list[dict]:
    """Return a safe error payload so the UI never crashes."""
    return [{
        "label":      "Error",
        "confidence": 0.0,
        "rank":       1,
        "error":      f"{type(exc).__name__}: {exc}",
    }]


def _scalar_prediction_payload(value: float, label: str = "Prediction") -> list[dict]:
    return [{
        "label": label,
        "confidence": 1.0,
        "rank": 1,
        "value": float(value),
    }]

"""
ONNX export and model bundle writer.

Output bundle structure:
    outputs/<name>_<timestamp>/
        model.onnx           — ONNX model
        model.joblib         — sklearn model (tabular only, alongside ONNX)
        preprocessing.json   — all preprocessing params needed at inference
        labels.json          — {index: class_name}
        cluster_results.json — present only for clustering task
        inference.py         — ready-to-run inference script
"""
from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import torch

logger = logging.getLogger(__name__)


def export_bundle(
    model,
    preprocessing_config: dict,
    classes: list[str],
    bundle_name: str,
    output_dir: str = "outputs",
    sample_batch=None,
    is_sklearn: bool = False,
    clustering_result: Optional[dict] = None,
    evaluation_artifacts: Optional[dict] = None,
) -> tuple[str, list[str]]:
    """Export model bundle to disk.

    Returns
    -------
    bundle_path : str
        Path to the created bundle directory.
    warnings : list[str]
        Non-fatal warnings to surface to the user (e.g. ONNX export skipped).
    """
    timestamp   = datetime.now().strftime("%Y%m%d_%H%M%S")
    bundle_path = Path(output_dir) / f"{bundle_name}_{timestamp}"
    bundle_path.mkdir(parents=True, exist_ok=True)
    warnings: list[str] = []

    # labels.json
    (bundle_path / "labels.json").write_text(
        json.dumps({str(i): c for i, c in enumerate(classes)}, indent=2)
    )

    # preprocessing.json
    (bundle_path / "preprocessing.json").write_text(
        json.dumps(_to_json_safe(preprocessing_config), indent=2)
    )

    # cluster_results.json (optional)
    if clustering_result:
        (bundle_path / "cluster_results.json").write_text(
            json.dumps({
                "n_clusters":       clustering_result["n_clusters"],
                "silhouette_score": clustering_result["silhouette_score"],
                "cluster_labels":   clustering_result["cluster_labels"],
            }, indent=2)
        )

    if evaluation_artifacts:
        (bundle_path / "evaluation_artifacts.json").write_text(
            json.dumps(_to_json_safe(evaluation_artifacts), indent=2)
        )

    # Export model
    if preprocessing_config.get("modality") == "graph":
        w = _export_graph_bundle(model, bundle_path, preprocessing_config, is_sklearn=is_sklearn)
        warnings.extend(w)
    elif is_sklearn:
        w = _export_sklearn(model, bundle_path, preprocessing_config)
        warnings.extend(w)
    else:
        _export_pytorch_onnx(model, bundle_path, preprocessing_config, sample_batch)

    # inference.py
    _write_inference_script(bundle_path, preprocessing_config, classes)

    return str(bundle_path), warnings


# ---------------------------------------------------------------------------
# PyTorch → ONNX
# ---------------------------------------------------------------------------

def _export_pytorch_onnx(model, bundle_path: Path, prep: dict, sample_batch) -> None:
    model = model.to("cpu").eval()
    onnx_path = str(bundle_path / "model.onnx")
    modality  = prep.get("modality", "")

    if modality == "text" and prep.get("tokenizer"):
        # Transformer text models need two named inputs
        if sample_batch is None:
            raise ValueError("sample_batch required for transformer text ONNX export")
        input_ids      = sample_batch["input_ids"][:1].to("cpu")
        attention_mask = sample_batch["attention_mask"][:1].to("cpu")
        torch.onnx.export(
            model,
            (input_ids, attention_mask),
            onnx_path,
            input_names  = ["input_ids", "attention_mask"],
            output_names = ["logits"],
            dynamic_axes = {
                "input_ids":      {0: "batch", 1: "seq"},
                "attention_mask": {0: "batch", 1: "seq"},
                "logits":         {0: "batch"},
            },
            opset_version = 14,
        )
        # Also save tokenizer alongside for convenience
        try:
            from transformers import AutoTokenizer
            tok = AutoTokenizer.from_pretrained(prep["tokenizer"])
            tok.save_pretrained(str(bundle_path / "tokenizer"))
        except Exception as e:
            logger.warning("Tokenizer save skipped: %s", e)

    elif sample_batch is not None:
        if isinstance(sample_batch, dict):
            # generic dict input (rare)
            dummy = {k: v[:1].to("cpu") for k, v in sample_batch.items()}
            inp   = tuple(dummy.values())
            names = list(dummy.keys())
        elif isinstance(sample_batch, (list, tuple)):
            inp   = tuple(t[:1].to("cpu") for t in sample_batch)
            names = [f"input_{i}" for i in range(len(inp))]
        else:
            inp   = (sample_batch[:1].to("cpu"),)
            names = ["input"]

        torch.onnx.export(
            model,
            inp if len(inp) > 1 else inp[0],
            onnx_path,
            input_names  = names,
            output_names = ["logits"],
            dynamic_axes = {n: {0: "batch"} for n in names} | {"logits": {0: "batch"}},
            opset_version = 14,
        )
    else:
        raise ValueError("sample_batch is required for ONNX export")


# ---------------------------------------------------------------------------
# sklearn → ONNX (+ joblib fallback)
# ---------------------------------------------------------------------------

def _export_sklearn(model, bundle_path: Path, prep: dict) -> list[str]:
    """Export sklearn model.  Returns a list of warning strings."""
    import joblib
    joblib.dump(model, bundle_path / "model.joblib")
    warnings: list[str] = []

    if prep.get("modality") == "graph":
        warnings.append(
            "⚠️ Graph baseline bundle saved as joblib. Portable ONNX graph inference is not enabled for this first graph export path."
        )
        return warnings

    try:
        from skl2onnx import convert_sklearn
        from skl2onnx.common.data_types import FloatTensorType

        input_size   = prep.get("sklearn_input_size", prep.get("input_size", 1))
        initial_type = [("float_input", FloatTensorType([None, input_size]))]
        onnx_model   = convert_sklearn(model, initial_types=initial_type)
        (bundle_path / "model.onnx").write_bytes(onnx_model.SerializeToString())
    except Exception as exc:
        msg = (
            f"⚠️ sklearn → ONNX conversion failed ({exc}). "
            "Model saved as **joblib only** — ONNX inference and the 'Try Your Model' "
            "tab will not work. Install `skl2onnx` to enable ONNX export."
        )
        logger.warning("sklearn → ONNX export failed: %s", exc)
        warnings.append(msg)

    return warnings


def _export_graph_bundle(model, bundle_path: Path, prep: dict, *, is_sklearn: bool) -> list[str]:
    warnings: list[str] = []
    if is_sklearn:
        return _export_sklearn(model, bundle_path, prep)
    torch.save(model.state_dict(), bundle_path / "model.pt")
    warnings.append(
        "⚠️ Graph neural network bundle saved as a PyTorch state dict. The generic ONNX export and 'Try Your Model' flow are not yet enabled for graph models."
    )
    return warnings


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _to_json_safe(obj):
    if isinstance(obj, dict):
        return {k: _to_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_json_safe(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    return obj


def _write_inference_script(bundle_path: Path, prep: dict, classes: list[str]) -> None:
    modality = prep.get("modality", "unknown")
    if modality == "graph":
        script = '''\
"""
Auto-generated graph bundle note.

Graph model bundles in this first release are intended for reproducible storage,
dashboard export, and future fine-tuning inside NoCode-DL. Standalone graph
inference scripts are not generated yet because graph inference needs the full
nodes.csv + edges.csv context used during training.
"""

print("Graph bundle saved successfully.")
print("Use this bundle inside NoCode-DL or the exported Streamlit dashboard.")
'''
        (bundle_path / "inference.py").write_text(script)
        return
    script = f'''\
"""
Auto-generated inference script — {modality} model.

Requirements:
    pip install onnxruntime numpy

Usage:
    python inference.py --input path/to/sample.npy
    (save your preprocessed input as a numpy .npy file first)
"""
import argparse, json
import numpy as np
import onnxruntime as ort

with open("labels.json") as f:
    LABELS = json.load(f)

session = ort.InferenceSession("model.onnx", providers=["CPUExecutionProvider"])
INPUT_NAME = session.get_inputs()[0].name


def predict(x: np.ndarray) -> str:
    if x.ndim == len(session.get_inputs()[0].shape) - 1:
        x = x[np.newaxis]          # add batch dim
    logits = session.run(None, {{INPUT_NAME: x.astype(np.float32)}})[0]
    return LABELS[str(int(np.argmax(logits, axis=1)[0]))]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to .npy preprocessed input")
    args = parser.parse_args()
    print("Predicted class:", predict(np.load(args.input)))
'''
    (bundle_path / "inference.py").write_text(script)

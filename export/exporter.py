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
import shutil
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import torch

logger = logging.getLogger(__name__)

RUNTIME_SOURCE = Path(__file__).with_name("bundle_runtime.py")


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

    # bundle_runtime.py + inference.py: the same preprocessing code the app uses,
    # so the bundle works standalone and matches "Try Your Model".
    if preprocessing_config.get("modality") != "graph":
        shutil.copy2(RUNTIME_SOURCE, bundle_path / "bundle_runtime.py")
    _write_inference_script(bundle_path, preprocessing_config, classes)

    return str(bundle_path), warnings


# ---------------------------------------------------------------------------
# PyTorch → ONNX
# ---------------------------------------------------------------------------

# torch >= 2.9 defaults to the torch.export-based ("dynamo") ONNX exporter,
# which needs the onnxscript package, opset >= 18 and dynamic_shapes instead of
# dynamic_axes, and has trouble with RNN models. Keep the TorchScript exporter
# (dynamo=False) so bundles stay opset 14 with the same input/output names.
_LEGACY_ONNX_EXPORTER = False


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
            dynamo        = _LEGACY_ONNX_EXPORTER,
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
            dynamo        = _LEGACY_ONNX_EXPORTER,
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

        from sklearn.base import is_classifier

        _patch_skl2onnx_hist_gradient_boosting()
        input_size   = prep.get("sklearn_input_size", prep.get("input_size", 1))
        initial_type = [("float_input", FloatTensorType([None, input_size]))]
        # zipmap=False: classifiers output a plain (n, classes) "probabilities"
        # array instead of a list of {class: prob} dicts that numpy can't use.
        options      = {id(model): {"zipmap": False}} if is_classifier(model) else None
        onnx_model   = convert_sklearn(model, initial_types=initial_type, options=options)
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


def _patch_skl2onnx_hist_gradient_boosting() -> None:
    """Make skl2onnx able to convert HistGradientBoosting models.

    skl2onnx 1.20 passes a Python ``False`` for leaf nodes in
    ``nodes_missing_value_tracks_true``, and onnx >= 1.17 rejects booleans in an
    int list ("Expected an int, got a boolean").  Coercing the flag to int keeps
    the same meaning (1 = missing values follow the true branch).  The patch is
    installed once and is a no-op for every other tree converter.
    """
    import skl2onnx.common.tree_ensemble as tree_ensemble

    if getattr(tree_ensemble.add_node, "_nocode_int_missing", False):
        return
    original = tree_ensemble.add_node

    def add_node(*args, nodes_missing_value_tracks_true=0, **kwargs):
        return original(
            *args,
            nodes_missing_value_tracks_true=int(nodes_missing_value_tracks_true),
            **kwargs,
        )

    add_node._nocode_int_missing = True
    tree_ensemble.add_node = add_node


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


# Packages bundle_runtime.py needs per modality, pinned to the versions the
# model was trained with (looked up from the running environment).
_RUNTIME_PACKAGES = {
    "image": ["Pillow"],
    "audio": ["Pillow", "torch", "torchaudio", "torchvision", "soundfile", "scipy"],
    "video": ["opencv-python-headless"],
    "text": [],
    "tabular": [],
    "timeseries": [],
}


def _pinned(dist: str) -> str:
    from importlib.metadata import PackageNotFoundError, version

    # Bundles use the headless OpenCV build (servers have no display); pin it
    # to whichever OpenCV build the app itself was trained with.
    lookups = [dist, "opencv-python"] if dist == "opencv-python-headless" else [dist]
    for name in lookups:
        try:
            return f"{dist}=={version(name)}"
        except PackageNotFoundError:
            continue
    return dist


def runtime_requirements(prep: dict, bundle_path: Path) -> list[str]:
    """Pinned requirements for running bundle_runtime.py on this bundle."""
    modality = prep.get("modality", "")
    packages = ["numpy", "onnxruntime", *_RUNTIME_PACKAGES.get(modality, [])]
    if modality == "text" and not prep.get("vocab"):
        packages.append("transformers")
    if not (bundle_path / "model.onnx").is_file() and (bundle_path / "model.joblib").is_file():
        packages += ["joblib", "scikit-learn"]
    return [_pinned(p) for p in packages]


def _example_input(prep: dict) -> tuple[str, str]:
    """(argument shown in the usage line, one-line description) for a modality."""
    modality = prep.get("modality", "")
    if modality == "tabular":
        example = {c: 0.0 for c in prep.get("numeric_cols", [])[:3]}
        for col, cats in list(prep.get("ohe_categories", {}).items())[:2]:
            example[col] = str(cats[0]) if cats else ""
        return f"'{json.dumps(example)}'", "a JSON object of column values (or a .json file)"
    if modality == "timeseries":
        cols = prep.get("raw_feature_columns") or prep.get("feature_columns") or ["x"]
        row = [0.0] * len(cols)
        return f"'{json.dumps([row, row])}'", (
            f"a JSON list of recent rows, oldest first, each [{', '.join(cols)}] (or a .json file)")
    if modality == "text":
        return '"text to classify"', "the text to classify"
    return {"image": "photo.jpg", "audio": "clip.wav", "video": "clip.mp4"}.get(modality, "input"), \
        f"path to a {modality} file"


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
        (bundle_path / "inference.py").write_text(script, encoding="utf-8")
        return

    reqs = runtime_requirements(prep, bundle_path)
    (bundle_path / "requirements.txt").write_text("\n".join(reqs) + "\n", encoding="utf-8")
    example, description = _example_input(prep)
    structured = modality in ("tabular", "timeseries")
    script = f'''\
"""
Command-line inference for this {modality} model.

Preprocessing comes from bundle_runtime.py in this folder: the same code the
NoCode-DL app used, so results match "Try Your Model".

Setup (once):
    pip install -r requirements.txt

Usage:
    python inference.py {example}

INPUT is {description}.
"""
import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from bundle_runtime import Bundle, BundleInputError  # noqa: E402

STRUCTURED_INPUT = {structured!r}


def main() -> int:
    parser = argparse.ArgumentParser(description="Predict with this NoCode-DL model bundle.")
    parser.add_argument("input", help={description!r})
    parser.add_argument("--top-k", type=int, default=3, help="How many classes to list.")
    args = parser.parse_args()

    raw = args.input
    if STRUCTURED_INPUT:
        path = Path(raw)
        raw = json.loads(path.read_text(encoding="utf-8") if path.is_file() else raw)
    try:
        result = Bundle(HERE, trust_pickle=True).predict(raw, top_k=args.top_k)  # your own export
    except BundleInputError as exc:
        print(f"Invalid input: {{exc}}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
'''
    (bundle_path / "inference.py").write_text(script, encoding="utf-8")

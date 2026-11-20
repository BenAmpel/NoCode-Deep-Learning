"""Write a run record into every exported bundle.

``run_record.json`` holds everything needed to describe or repeat a training
run (data fingerprint, split, seed, every setting, device, library versions,
metrics); ``RUN_RECORD.md`` turns it into a methods paragraph and tables a
student can paste into a report and a grader or reviewer can check.

The record describes what actually ran, not only what was asked for: the
training seed that ``prepare_run`` set, the device the run trained on, the
real split sizes and the epochs completed. Nothing here draws from an RNG or
changes the run's state.
"""
from __future__ import annotations

import dataclasses
import datetime as _dt
import hashlib
import inspect
import json
import math
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1
JSON_NAME = "run_record.json"
MARKDOWN_NAME = "RUN_RECORD.md"

# Content hashing reads every byte; past this the fingerprint falls back to
# names and sizes so a large video folder doesn't add minutes to the export.
CONTENT_HASH_LIMIT_BYTES = 4 * 1024**3

_ROOT = Path(__file__).resolve().parent.parent
_LIBRARIES = ("torch", "numpy", "pandas", "scikit-learn", "gradio", "onnx", "onnxruntime", "xgboost")
# Shown first in the methods paragraph when present, in this order.
_HEADLINE_METRICS = ("accuracy", "balanced_accuracy", "f1_macro", "auc_ovr", "mcc", "r2", "rmse", "mae")


# ── helpers ──────────────────────────────────────────────────────────────────

def _jsonable(value: Any) -> Any:
    """Plain JSON types only; NaN and infinities become null."""
    if value is None or isinstance(value, (bool, str)):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(v) for v in value]
    item = getattr(value, "item", None)  # numpy / torch scalars
    if callable(item):
        try:
            return _jsonable(item())
        except (TypeError, ValueError):
            pass
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        return _jsonable(tolist())
    if isinstance(value, Path):
        return str(value)
    return str(value)


def _sha256_file(path: Path, h: "hashlib._Hash") -> None:
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)


def fingerprint_data(data_path: str | None) -> dict:
    """A SHA-256 of the dataset as supplied (a file or a folder).

    Folders are hashed as sorted ``relative path + contents`` pairs, so the
    fingerprint is the same on any machine and ignores timestamps. Hidden
    files (``.DS_Store`` and the like) are skipped.
    """
    if not data_path or not str(data_path).strip():
        return {"method": "unavailable", "reason": "no data path"}
    path = Path(str(data_path)).expanduser()
    if not path.exists():
        return {"method": "unavailable", "reason": "path not found", "name": path.name}

    if path.is_file():
        files = [path]
        base = path.parent
    else:
        files = sorted(
            (p for p in path.rglob("*")
             if p.is_file() and not any(part.startswith(".") for part in p.relative_to(path).parts)),
            key=lambda p: p.relative_to(path).as_posix(),
        )
        base = path
    total = sum(p.stat().st_size for p in files)
    content = total <= CONTENT_HASH_LIMIT_BYTES
    h = hashlib.sha256()
    for p in files:
        rel = p.relative_to(base).as_posix()
        h.update(rel.encode("utf-8") + b"\0")
        if content:
            _sha256_file(p, h)
        else:
            h.update(str(p.stat().st_size).encode() + b"\0")
    return {
        "method": "sha256 of file contents" if content else "sha256 of file names and sizes (data over 4 GB)",
        "sha256": h.hexdigest(),
        "name": path.name,
        "kind": "file" if path.is_file() else "folder",
        "files": len(files),
        "bytes": total,
    }


def _app_version() -> str:
    version_file = _ROOT / ".bundle_version"
    if version_file.is_file():
        return version_file.read_text(encoding="utf-8").strip() or "unknown"
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], cwd=_ROOT,
            capture_output=True, text=True, timeout=5, check=True,
        )
        return f"dev ({out.stdout.strip()})"
    except (OSError, subprocess.SubprocessError):
        return "dev"


def _library_versions() -> dict:
    from importlib.metadata import PackageNotFoundError, version

    found = {}
    for name in _LIBRARIES:
        try:
            found[name] = version(name)
        except PackageNotFoundError:
            continue
    return found


def _split_seed() -> int:
    from data_pipeline.splits import stratified_split

    return int(inspect.signature(stratified_split).parameters["seed"].default)


def _dataset_size(loader: Any) -> int | None:
    try:
        return len(loader.dataset)
    except (AttributeError, TypeError):
        return None


def _parameter_count(model: Any) -> int | None:
    params = getattr(model, "parameters", None)
    if not callable(params):
        return None
    try:
        return int(sum(p.numel() for p in params()))
    except (TypeError, AttributeError, RuntimeError):
        return None


def _to_float(value: Any) -> float | None:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


# ── the record ───────────────────────────────────────────────────────────────

def _split_description(cfg: Any) -> tuple[str, int | None]:
    """How the modality loader split the data (see ``modalities/*.py``), and its seed."""
    if cfg.modality == "timeseries":
        # Validation is the last part of the series; training windows end before it.
        return "chronological (validation is the end of the series, no overlap)", None
    if cfg.modality == "graph":
        return "stratified by class (random if a class has fewer than 2 labelled nodes), by node", int(cfg.subset_seed)
    if cfg.modality == "tabular" and cfg.task == "regression":
        return "random", _split_seed()
    return "stratified by class", _split_seed()


def build_record(ctx: Any, bundle_path: str, *, training_seed: int) -> dict:
    """Everything known about a finished run, as plain JSON types."""
    cfg = ctx.cfg
    history = list(ctx.result.history or [])
    settings = dataclasses.asdict(cfg)
    # Keep only the dataset's name: the record is shared with graders and
    # reviewers, and an absolute path carries the user's account name.
    settings["data_path"] = Path(str(cfg.data_path)).name if cfg.data_path else ""

    split_method, split_seed = _split_description(cfg)

    subset = bool(cfg.use_random_subset)
    record = {
        "schema_version": SCHEMA_VERSION,
        "created_utc": _dt.datetime.now(_dt.timezone.utc).replace(microsecond=0).isoformat(),
        "app": {"name": "NoCode Deep Learning Studio", "version": _app_version()},
        "bundle": Path(bundle_path).name,
        "data": {
            "modality": cfg.modality,
            "fingerprint": fingerprint_data(cfg.data_path),
            "cleaning_applied": bool(cfg.use_data_cleaning),
            "random_subset": (
                {"percent": _jsonable(cfg.subset_percent), "seed": int(cfg.subset_seed)} if subset else None
            ),
            "classes": [str(c) for c in (ctx.classes or [])],
        },
        "split": {
            "method": split_method,
            "seed": split_seed,
            "validation_fraction": _to_float(cfg.val_split),
            "train_samples": _dataset_size(ctx.train_loader),
            "validation_samples": _dataset_size(ctx.val_loader),
            "test_set": None,
        },
        "model": {
            "name": cfg.model_name,
            "training_mode": cfg.training_mode,
            "task": cfg.task,
            "trainable_framework": "scikit-learn" if ctx.sklearn_model else "pytorch",
            "parameters": _parameter_count(ctx.final_model),
        },
        "training": {
            "seed": int(training_seed),
            "epochs_requested": _jsonable(cfg.epochs),
            "epochs_completed": len(history),
            "stopped_early": bool(ctx.stopped_early),
            "history": history,
        },
        "metrics": {
            "computed_on": "validation split" if ctx.val_loader is not None or ctx.sklearn_model else "training data",
            "values": dict(ctx.result.metrics or {}),
        },
        "environment": {
            "detected_device": ctx.device,
            "training_device": ctx.runtime_device,
            "python": platform.python_version(),
            "platform": f"{platform.system()} {platform.release()} ({platform.machine()})",
            "libraries": _library_versions(),
        },
        "settings": settings,
    }
    return _jsonable(record)


def _fmt(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.4g}"
    if value is None or value == "":
        return "—"
    return str(value)


def _article(name: str) -> str:
    """'an' for names read with a vowel sound (an MLP, an LSTM, an Autoencoder), else 'a'."""
    word = str(name).split("-")[0]
    if word[:1].upper() in "AEIOUX":  # X as in "an XGBoost"
        return "an"
    if word.isupper() and word[:1] in "FHLMNRSX":  # acronyms spelled out: "eff", "em", "ess"...
        return "an"
    return "a"


def methods_paragraph(record: dict) -> str:
    """A plain description of the run, ready to adapt for a methods section."""
    data, split, model, train = record["data"], record["split"], record["model"], record["training"]
    env, metrics = record["environment"], record["metrics"]
    fp = data["fingerprint"]
    settings = record["settings"]

    dataset = fp.get("name") or "the dataset"
    digest = f"; SHA-256 {fp['sha256'][:12]}…" if fp.get("sha256") else ""
    parts = [
        f"We trained {_article(model['name'])} {model['name']} model "
        f"({str(model['training_mode']).replace('_', ' ')}) for {model['task']} "
        f"on {data['modality']} data ({dataset}{digest}) using NoCode Deep Learning Studio "
        f"{record['app']['version']}."
    ]
    if split["train_samples"] is not None and split["validation_samples"] is not None:
        frac = split["validation_fraction"]
        held = f"{frac:.0%} held out, " if frac is not None else ""
        seed = f", seed {split['seed']}" if split["seed"] is not None else ""
        parts.append(
            f"The data were split into {split['train_samples']} training and "
            f"{split['validation_samples']} validation samples ({held}{split['method']}{seed})."
        )
    if data["random_subset"]:
        parts.append(
            f"A random {_fmt(data['random_subset']['percent'])}% subset of the data was used "
            f"(seed {data['random_subset']['seed']})."
        )
    if model["trainable_framework"] == "pytorch":
        stop = " (stopped early)" if train["stopped_early"] else ""
        parts.append(
            f"Training used the {settings.get('optimizer')} optimizer with learning rate "
            f"{_fmt(settings.get('lr'))} and batch size {_fmt(settings.get('batch_size'))} for "
            f"{train['epochs_completed']} of {_fmt(train['epochs_requested'])} epochs{stop}, "
            f"with random seed {train['seed']}, on {env['training_device']} "
            f"(PyTorch {env['libraries'].get('torch', 'unknown')})."
        )
    else:
        parts.append(
            f"The model was fit with scikit-learn {env['libraries'].get('scikit-learn', '')}".rstrip()
            + f" (random seed {train['seed']}); its settings are listed below."
        )
    values = metrics["values"]
    shown = [k for k in _HEADLINE_METRICS if values.get(k) is not None][:4]
    if shown:
        listed = ", ".join(f"{k.replace('_', ' ')} {_fmt(values[k])}" for k in shown)
        parts.append(f"On the {metrics['computed_on']}, the model reached {listed}.")
    return " ".join(parts)


def render_markdown(record: dict) -> str:
    data, split, model, train = record["data"], record["split"], record["model"], record["training"]
    env, metrics, fp = record["environment"], record["metrics"], record["data"]["fingerprint"]

    def table(rows: list[tuple[str, Any]]) -> list[str]:
        return ["| Field | Value |", "|---|---|"] + [f"| {k} | {_fmt(v)} |" for k, v in rows]

    lines = [
        f"# Run record: {model['name']}",
        "",
        f"Created {record['created_utc']} by NoCode Deep Learning Studio {record['app']['version']}. "
        f"The full record, with every setting and the per-epoch history, is in `{JSON_NAME}`.",
        "",
        "## Methods paragraph",
        "",
        methods_paragraph(record),
        "",
        "## Data",
        "",
        *table([
            ("Modality", data["modality"]),
            ("Dataset", fp.get("name")),
            ("Fingerprint", fp.get("sha256")),
            ("Fingerprint method", fp.get("method")),
            ("Files / bytes", f"{fp.get('files', '—')} / {fp.get('bytes', '—')}"),
            ("Cleaning applied", data["cleaning_applied"]),
            ("Random subset", f"{data['random_subset']['percent']}% (seed {data['random_subset']['seed']})"
             if data["random_subset"] else "no"),
            ("Classes", ", ".join(data["classes"]) or None),
        ]),
        "",
        "## Split",
        "",
        *table([
            ("Method", split["method"]),
            ("Seed", split["seed"]),
            ("Validation fraction", split["validation_fraction"]),
            ("Training samples", split["train_samples"]),
            ("Validation samples", split["validation_samples"]),
            ("Separate test set", "none"),
        ]),
        "",
        "## Model and training",
        "",
        *table([
            ("Model", model["name"]),
            ("Training mode", model["training_mode"]),
            ("Task", model["task"]),
            ("Parameters", model["parameters"]),
            ("Random seed", train["seed"]),
            ("Epochs completed / requested", f"{train['epochs_completed']} / {_fmt(train['epochs_requested'])}"),
            ("Stopped early", train["stopped_early"]),
        ]),
        "",
        "## Metrics",
        "",
        f"Computed on the {metrics['computed_on']}.",
        "",
        "| Metric | Value |",
        "|---|---|",
        *[f"| {k} | {_fmt(v)} |" for k, v in metrics["values"].items()],
        "",
        "## Environment",
        "",
        *table([
            ("Training device", env["training_device"]),
            ("Detected device", env["detected_device"]),
            ("Python", env["python"]),
            ("Platform", env["platform"]),
            *[(name, ver) for name, ver in env["libraries"].items()],
        ]),
        "",
        "## Notes for readers",
        "",
        f"- Metrics come from the {metrics['computed_on']}. There is no separate held-out test set, "
        "so treat them as validation results, not a final test estimate.",
        "- The same data (matching fingerprint), the same settings and the same app version give the "
        "same split and the same starting seed. Results on GPUs (CUDA, Apple MPS) and with parallel "
        "data loading can still differ slightly from run to run.",
        "",
    ]
    return "\n".join(lines)


def write_run_record(ctx: Any, bundle_path: str, *, training_seed: int) -> Path:
    """Write both files into the bundle folder; returns the JSON path."""
    bundle = Path(bundle_path)
    record = build_record(ctx, bundle_path, training_seed=training_seed)
    json_path = bundle / JSON_NAME
    json_path.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    (bundle / MARKDOWN_NAME).write_text(render_markdown(record), encoding="utf-8")
    return json_path

"""Batch prediction for tabular bundles: score every row of a CSV.

Each row goes through the bundle's own ``Bundle.predict`` (scaling, one-hot
``category_maps`` and the fill-with-zero rule for missing numbers), so a row
scores exactly as it would in "Try Your Model" or an exported server.

Output columns:

- the chosen ID column, read as text so values like ``00123`` survive;
- classification: ``prediction``, ``confidence`` (text such as ``"87.3%"``,
  the same format folder batch prediction writes) and ``confidence_score``
  (the same number as a 0-1 float);
- regression: ``prediction`` (the predicted value, a float);
- tabular Autoencoder: ``anomaly_score`` (mean squared reconstruction error,
  as in the Evaluate tab; higher means more unusual).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROW_COLUMN = "row"


class TabularBatchError(ValueError):
    """A problem with the bundle or CSV that stops the whole batch."""


def read_columns(csv_path: str | Path) -> list[str]:
    """Column names of a CSV/TSV file, read from its header only."""
    from data_pipeline.io_utils import read_structured_file

    return [str(c) for c in read_structured_file(csv_path, nrows=0).columns]


def default_id_column(columns: list[str], prep: dict | None = None) -> str | None:
    """Best guess for the ID column: one named like an ID, else the first non-feature column."""
    features = set(_feature_columns(prep or {}))
    for col in columns:
        low = col.lower()
        if low == "id" or low.endswith("_id") or low.endswith(" id"):
            return col
    return next((c for c in columns if c not in features), None)


def _feature_columns(prep: dict) -> list[str]:
    # Only numeric and categorical columns reach the model; other columns in
    # feature_columns (booleans, dates) were dropped during training.
    return list(prep.get("numeric_cols", [])) + list(prep.get("categorical_cols", []))


def is_autoencoder(prep: dict) -> bool:
    return str(prep.get("model_name", "")) == "Autoencoder" or prep.get("task") == "anomaly"


def predict_csv(bundle_path: str, csv_path: str | Path, id_col: str | None = None) -> tuple[pd.DataFrame, str]:
    """Score every row of ``csv_path`` with a tabular bundle.

    Returns ``(results, notes)`` where ``notes`` is Markdown describing rows
    that needed fallbacks (unseen categories, blank or non-numeric cells).
    Raises ``TabularBatchError`` when the batch cannot run at all, for example
    when the CSV lacks a feature column the model was trained on.
    """
    from data_pipeline.io_utils import read_structured_file
    from ui.inference_helpers import _runtime_bundle

    try:
        bundle = _runtime_bundle(bundle_path)
    except Exception as exc:
        raise TabularBatchError(f"Could not load bundle: {exc}") from exc
    if bundle.modality != "tabular":
        raise TabularBatchError(
            f"This bundle is a **{bundle.modality or 'unknown'}** model; CSV batch prediction needs a tabular bundle."
        )
    prep = bundle.prep
    numeric_cols = list(prep.get("numeric_cols", []))
    categorical_cols = list(prep.get("categorical_cols", []))
    feature_cols = numeric_cols + categorical_cols

    csv_path = Path(csv_path)
    if not csv_path.is_file():
        raise TabularBatchError(f"CSV not found: `{csv_path}`")
    try:
        header = read_columns(csv_path)
    except Exception as exc:
        raise TabularBatchError(f"Could not read the CSV: {exc}") from exc

    missing = [c for c in feature_cols if c not in header]
    if missing:
        raise TabularBatchError(
            "The CSV is missing feature column(s) the model was trained on: "
            + ", ".join(f"`{c}`" for c in missing)
            + ". Add them (the names must match the training data exactly) and run again."
        )
    if id_col and id_col not in header:
        raise TabularBatchError(f"ID column `{id_col}` is not in the CSV.")

    # Categorical values and the ID are read as text: a category column that
    # holds only digits would otherwise parse as numbers and miss every
    # training category, and IDs would lose leading zeros.
    text_cols = {c: str for c in categorical_cols}
    if id_col:
        text_cols[id_col] = str
    try:
        df = read_structured_file(csv_path, dtype=text_cols)
    except Exception as exc:
        raise TabularBatchError(f"Could not read the CSV: {exc}") from exc
    if df.empty:
        raise TabularBatchError("The CSV has a header but no rows to score.")

    notes = _input_notes(df, prep, numeric_cols, categorical_cols)

    features = df[feature_cols].astype(object).where(df[feature_cols].notna(), None)
    records = features.to_dict("records")
    autoencoder = is_autoencoder(prep)
    regression = bundle.task == "regression"

    out: dict[str, list] = {}
    if autoencoder:
        out["anomaly_score"] = []
    else:
        out["prediction"] = []
        if not regression:
            out["confidence"] = []
            out["confidence_score"] = []
    errors = 0

    for raw in records:
        try:
            if autoencoder:
                out["anomaly_score"].append(_reconstruction_error(bundle, raw))
                continue
            result = bundle.predict(raw, top_k=1)
            if regression:
                out["prediction"].append(float(result["value"]))
            else:
                top = result["predictions"][0]
                conf = float(top["confidence"])
                out["prediction"].append(top["label"])
                out["confidence"].append(f"{conf * 100:.1f}%")
                out["confidence_score"].append(round(conf, 6))
        except Exception as exc:
            errors += 1
            msg = f"{type(exc).__name__}: {exc}"
            if autoencoder:
                out["anomaly_score"].append(np.nan)
            elif regression:
                out["prediction"].append(np.nan)
            else:
                out["prediction"].append("ERROR")
                out["confidence"].append(msg)
                out["confidence_score"].append(np.nan)

    if id_col:
        ids = df[id_col].reset_index(drop=True)
        result_df = pd.DataFrame({id_col: ids, **out})
    else:
        result_df = pd.DataFrame({ROW_COLUMN: np.arange(1, len(df) + 1), **out})
    if errors:
        notes.append(f"⚠️ {errors:,} row(s) could not be scored.")
    return result_df, "\n".join(f"- {n}" for n in notes)


def _reconstruction_error(bundle, raw: dict) -> float:
    """Per-row anomaly score, matching ``eval.anomaly_detection.compute_reconstruction_errors``."""
    x = np.asarray(bundle.preprocess(raw), dtype=np.float32).reshape(-1)
    recon = np.asarray(bundle._run(x.reshape(1, -1)), dtype=np.float32).reshape(-1)
    if recon.shape != x.shape:
        raise TabularBatchError(
            f"Autoencoder output has {recon.size} values but the input has {x.size}; "
            "this bundle does not look like a tabular Autoencoder."
        )
    return float(np.mean((x - recon) ** 2))


def _input_notes(df: pd.DataFrame, prep: dict, numeric_cols: list[str], categorical_cols: list[str]) -> list[str]:
    """Describe cells that score with a fallback, so nothing changes silently."""
    notes: list[str] = []
    category_maps = prep.get("category_maps", {})
    for col in categorical_cols:
        known = set(category_maps.get(col, {}))
        values = df[col]
        unseen = values.notna() & ~values.astype(str).isin(known)
        if unseen.any():
            examples = ", ".join(f"`{v}`" for v in values[unseen].astype(str).unique()[:3])
            notes.append(
                f"`{col}`: {int(unseen.sum()):,} row(s) have a value not seen in training ({examples}); "
                "they are scored with every one-hot slot for that column set to 0."
            )
    for col in numeric_cols:
        values = df[col]
        parsed = pd.to_numeric(values, errors="coerce")
        bad = values.notna() & parsed.isna()
        if bad.any():
            examples = ", ".join(f"`{v}`" for v in values[bad].astype(str).unique()[:3])
            notes.append(f"`{col}`: {int(bad.sum()):,} non-numeric value(s) ({examples}) are treated as 0.")
    blank = df[numeric_cols + categorical_cols].isna().any(axis=1)
    if blank.any():
        notes.append(
            f"{int(blank.sum()):,} row(s) have blank feature cells; blank numbers count as 0 and blank "
            "categories as all-zero one-hot, the same as during training."
        )
    return notes

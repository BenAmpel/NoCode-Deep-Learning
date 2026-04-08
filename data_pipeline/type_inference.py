"""
data_pipeline/type_inference.py

CSV column type inference and feature/label suggestion utilities for the
NoCode Deep Learning platform.
"""

from __future__ import annotations

from typing import Dict, List, Optional
import warnings

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Column names that strongly hint at being the target variable
_LABEL_HINTS = {"label", "target", "class", "y", "output", "response", "outcome"}
_ID_HINTS = {"id", "idx", "index", "identifier", "record_id", "user_id", "sample_id"}

# Thresholds
_MAX_CATEGORICAL_UNIQUE = 20       # absolute unique-value count
_MAX_CATEGORICAL_UNIQUE_RATIO = 0.05  # relative unique ratio
_MAX_NUMERIC_CATEGORICAL_RATIO = 0.20
_MIN_NUMERIC_UNIQUE = 10           # minimum unique values to be "numeric"
_MIN_TEXT_AVG_LEN = 20.0           # avg string length for "text" classification
_ID_UNIQUE_RATIO = 0.90            # fraction unique for string "id" detection


# ---------------------------------------------------------------------------
# Core type inference
# ---------------------------------------------------------------------------

def infer_column_types(df: pd.DataFrame) -> Dict[str, str]:
    """Infer the semantic type of each column in a DataFrame.

    Possible type strings
    ---------------------
    ``"constant"``    — Single unique non-null value.
    ``"boolean"``     — Exactly two unique values.
    ``"id"``          — Monotonically increasing integers OR high-cardinality
                        unique-string column (>90 % unique).
    ``"datetime"``    — Values parseable as dates/times.
    ``"numeric"``     — Numeric dtype with > 10 unique values relative to
                        dataset size.
    ``"categorical"`` — String or integer with ≤ 20 unique values or ≤ 5 %
                        unique ratio.
    ``"text"``        — String column with high unique ratio and mean token
                        length > 20 characters.
    ``"unknown"``     — Could not be assigned any of the above types.

    Parameters
    ----------
    df:
        Input DataFrame.  An empty DataFrame returns an empty dict.

    Returns
    -------
    dict mapping column name → type string.
    """
    if df.empty:
        return {}

    n_rows = len(df)
    col_types: Dict[str, str] = {}

    for col in df.columns:
        series = df[col]
        col_name = col.strip().lower()
        n_unique = series.nunique(dropna=True)
        n_non_null = series.count()

        # ---------------------------------------------------------- #
        # Constant
        # ---------------------------------------------------------- #
        if n_unique <= 1:
            col_types[col] = "constant"
            continue

        # ---------------------------------------------------------- #
        # Boolean
        # ---------------------------------------------------------- #
        if n_unique == 2:
            col_types[col] = "boolean"
            continue

        # ---------------------------------------------------------- #
        # Datetime — attempt a lightweight parse on a sample
        # ---------------------------------------------------------- #
        if _looks_like_datetime(series):
            col_types[col] = "datetime"
            continue

        # ---------------------------------------------------------- #
        # Numeric checks
        # ---------------------------------------------------------- #
        if pd.api.types.is_numeric_dtype(series):
            # Integer column: check for monotonically increasing ID pattern
            if pd.api.types.is_integer_dtype(series):
                non_null = series.dropna().sort_values().reset_index(drop=True)
                if (
                    any(hint in col_name for hint in _ID_HINTS)
                    and
                    n_unique == n_non_null
                    and n_non_null > 1
                    and (non_null.diff().dropna() > 0).all()
                ):
                    col_types[col] = "id"
                    continue

            # Categorical-like numeric (e.g. integer codes with few values)
            unique_ratio = n_unique / n_rows if n_rows > 0 else 0.0
            if (
                n_unique <= _MAX_CATEGORICAL_UNIQUE
                and unique_ratio <= _MAX_NUMERIC_CATEGORICAL_RATIO
            ):
                col_types[col] = "categorical"
                continue

            col_types[col] = "numeric"
            continue

        # ---------------------------------------------------------- #
        # Object / string checks
        # ---------------------------------------------------------- #
        str_series = series.dropna().astype(str)
        unique_ratio = n_unique / n_rows if n_rows > 0 else 0.0

        # High-cardinality string → candidate for "id" or "text"
        if unique_ratio >= _ID_UNIQUE_RATIO:
            # Differentiate "id" (short tokens) from "text" (long tokens)
            avg_len = str_series.str.len().mean() if len(str_series) > 0 else 0.0
            if avg_len > _MIN_TEXT_AVG_LEN:
                col_types[col] = "text"
            else:
                col_types[col] = "id"
            continue

        # Low-cardinality string → categorical
        if n_unique <= _MAX_CATEGORICAL_UNIQUE or unique_ratio <= _MAX_CATEGORICAL_UNIQUE_RATIO:
            col_types[col] = "categorical"
            continue

        # Moderate cardinality with long average string → text
        avg_len = str_series.str.len().mean() if len(str_series) > 0 else 0.0
        if avg_len > _MIN_TEXT_AVG_LEN:
            col_types[col] = "text"
            continue

        col_types[col] = "unknown"

    return col_types


def _looks_like_datetime(series: pd.Series, sample_size: int = 200) -> bool:
    """Heuristic check: can the series values be parsed as datetimes?"""
    if pd.api.types.is_datetime64_any_dtype(series):
        return True
    # Only try for object / string columns
    if not pd.api.types.is_object_dtype(series):
        return False

    sample = series.dropna().head(sample_size)
    if len(sample) == 0:
        return False

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            parsed = pd.to_datetime(sample, errors="coerce")
        # Accept if at least 80 % of the sample parses successfully
        return parsed.notna().mean() >= 0.80
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Feature / label suggestion
# ---------------------------------------------------------------------------

def suggest_features_and_label(df: pd.DataFrame) -> Dict:
    """Suggest a label column, feature columns, and columns to drop.

    Heuristics
    ----------
    * If a column is literally named one of "label", "target", "class", "y",
      "output", "response", or "outcome" (case-insensitive) it is preferred as
      the label.
    * Otherwise the last non-id, non-constant column is used as the label.
    * ``id`` and ``constant`` columns are placed in ``drop_cols``.
    * The first ``text`` column is noted separately (for embedding).

    Parameters
    ----------
    df:
        Input DataFrame.

    Returns
    -------
    dict with keys:

    ``suggested_label``
        Column name for the label, or ``None``.
    ``feature_cols``
        Columns suitable as model inputs (excludes label and drop_cols).
    ``drop_cols``
        Columns recommended for removal (id, constant, unknown).
    ``text_col``
        First detected text column, or ``None``.
    ``warnings``
        List of advisory strings.
    """
    if df.empty:
        return {
            "suggested_label": None,
            "feature_cols": [],
            "drop_cols": [],
            "text_col": None,
            "warnings": ["DataFrame is empty."],
        }

    col_types = infer_column_types(df)
    warnings: List[str] = []

    # ------------------------------------------------------------------ #
    # Identify drop candidates
    # ------------------------------------------------------------------ #
    drop_types = {"id", "constant", "unknown"}
    drop_cols = [c for c, t in col_types.items() if t in drop_types]

    # ------------------------------------------------------------------ #
    # Identify text columns
    # ------------------------------------------------------------------ #
    text_cols = [c for c, t in col_types.items() if t == "text"]
    text_col: Optional[str] = text_cols[0] if text_cols else None
    if len(text_cols) > 1:
        warnings.append(
            f"Multiple text columns detected: {text_cols}. "
            f"Using '{text_col}' as the primary text column."
        )

    # ------------------------------------------------------------------ #
    # Suggest label
    # ------------------------------------------------------------------ #
    suggested_label: Optional[str] = None

    # Priority 1: exact name match (case-insensitive)
    for col in df.columns:
        if col.strip().lower() in _LABEL_HINTS:
            suggested_label = col
            break

    # Priority 2: last column that is not id/constant/drop
    if suggested_label is None:
        candidate_cols = [c for c in df.columns if c not in drop_cols]
        if candidate_cols:
            suggested_label = candidate_cols[-1]

    if suggested_label is None:
        warnings.append("Could not identify a suitable label column.")

    # ------------------------------------------------------------------ #
    # Feature columns
    # ------------------------------------------------------------------ #
    excluded = set(drop_cols)
    if suggested_label is not None:
        excluded.add(suggested_label)
    if text_col is not None:
        excluded.add(text_col)

    feature_cols = [c for c in df.columns if c not in excluded]

    if not feature_cols:
        warnings.append("No feature columns identified after filtering.")

    if df.shape[1] == 1:
        warnings.append("DataFrame has only one column — cannot split into features and label.")

    return {
        "suggested_label": suggested_label,
        "feature_cols": feature_cols,
        "drop_cols": drop_cols,
        "text_col": text_col,
        "warnings": warnings,
    }


# ---------------------------------------------------------------------------
# Markdown summary
# ---------------------------------------------------------------------------

def type_inference_markdown(
    col_types: Dict[str, str],
    suggestions: Dict,
) -> str:
    """Render column type inference results and suggestions as Markdown.

    Parameters
    ----------
    col_types:
        Dict as returned by :func:`infer_column_types`.
    suggestions:
        Dict as returned by :func:`suggest_features_and_label`.

    Returns
    -------
    Markdown-formatted string.
    """
    lines: List[str] = [
        "## Column Type Inference",
        "",
        "| Column | Inferred Type |",
        "| --- | --- |",
    ]
    for col, ctype in col_types.items():
        lines.append(f"| `{col}` | {ctype} |")

    lines += [
        "",
        "## Feature & Label Suggestions",
        "",
        f"- **Suggested label column**: `{suggestions['suggested_label']}`",
        f"- **Feature columns** ({len(suggestions['feature_cols'])}): "
        + (
            ", ".join(f"`{c}`" for c in suggestions["feature_cols"])
            if suggestions["feature_cols"]
            else "_none_"
        ),
        f"- **Columns to drop** ({len(suggestions['drop_cols'])}): "
        + (
            ", ".join(f"`{c}`" for c in suggestions["drop_cols"])
            if suggestions["drop_cols"]
            else "_none_"
        ),
        f"- **Text column**: "
        + (f"`{suggestions['text_col']}`" if suggestions["text_col"] else "_none_"),
    ]

    if suggestions["warnings"]:
        lines += ["", "### Warnings", ""]
        for w in suggestions["warnings"]:
            lines.append(f"- {w}")

    return "\n".join(lines)

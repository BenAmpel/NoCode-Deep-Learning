"""
data_pipeline/cleaning.py

Data cleaning utilities for the NoCode Deep Learning platform.
"""

from __future__ import annotations

import logging
import re
from typing import Dict, Literal, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------
CleaningStrategy = Literal["auto", "median", "mean", "drop"]
TimeseriesFillStrategy = Literal["none", "forward_fill", "interpolate"]

_BASIC_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "been", "being", "but", "by",
    "for", "from", "had", "has", "have", "he", "her", "hers", "him", "his",
    "i", "if", "in", "into", "is", "it", "its", "me", "my", "of", "on", "or",
    "our", "ours", "she", "so", "than", "that", "the", "their", "theirs",
    "them", "they", "this", "those", "to", "too", "us", "was", "we", "were",
    "what", "when", "where", "which", "who", "why", "with", "you", "your",
    "yours",
}


# ---------------------------------------------------------------------------
# Main cleaning function
# ---------------------------------------------------------------------------

def clean_dataframe(
    df: pd.DataFrame,
    label_col: str,
    strategy: CleaningStrategy = "auto",
    clip_outliers: bool = False,
) -> Tuple[pd.DataFrame, Dict]:
    """Clean a tabular DataFrame in place and return a report.

    Steps performed (in order)
    --------------------------
    1. Remove fully duplicate rows.
    2. Drop columns where more than 80 % of values are missing.
    3. Drop constant columns (zero-variance), excluding the label column.
    4. Fill missing values:
       - Numeric columns: median (``strategy="auto"`` or ``"median"``),
         mean (``strategy="mean"``), or drop rows (``strategy="drop"``).
       - Categorical / object columns: mode (most frequent value).
    5. Optionally clip numeric outliers to the 1st/99th percentile range.
    6. Build a human-readable summary.

    Parameters
    ----------
    df:
        Input DataFrame.  Not modified in-place; a copy is returned.
    label_col:
        Name of the target / label column.  It is protected from being
        dropped as a constant column.
    strategy:
        Missing-value imputation strategy for numeric columns.
        ``"auto"`` and ``"median"`` both use the column median.

    Returns
    -------
    cleaned_df:
        The cleaned DataFrame with a reset index.
    report_dict:
        Dict with keys ``rows_removed``, ``cols_removed``,
        ``missing_filled``, ``constant_cols_removed``, ``summary``.
    """
    if df.empty:
        report = {
            "rows_removed": 0,
            "cols_removed": 0,
            "missing_filled": 0,
            "constant_cols_removed": 0,
            "summary": "Input DataFrame is empty — nothing to clean.",
        }
        return df.copy(), report

    cleaned = df.copy()
    initial_rows, initial_cols = cleaned.shape

    rows_removed = 0
    cols_removed = 0
    missing_filled = 0
    constant_cols_removed = 0
    dropped_col_names: list[str] = []
    outlier_values_clipped = 0

    # ------------------------------------------------------------------ #
    # 1. Remove duplicate rows
    # ------------------------------------------------------------------ #
    before = len(cleaned)
    cleaned = cleaned.drop_duplicates()
    rows_removed += before - len(cleaned)

    # ------------------------------------------------------------------ #
    # 2. Drop columns with >80 % missing values
    # ------------------------------------------------------------------ #
    missing_fraction = cleaned.isnull().mean()
    high_missing_cols = missing_fraction[missing_fraction > 0.80].index.tolist()
    # Never drop the label column at this stage
    high_missing_cols = [c for c in high_missing_cols if c != label_col]
    if high_missing_cols:
        cleaned = cleaned.drop(columns=high_missing_cols)
        cols_removed += len(high_missing_cols)
        dropped_col_names.extend(high_missing_cols)

    # ------------------------------------------------------------------ #
    # 3. Drop constant columns (excluding label)
    # ------------------------------------------------------------------ #
    feature_cols = [c for c in cleaned.columns if c != label_col]
    const_cols = [c for c in feature_cols if cleaned[c].nunique(dropna=True) <= 1]
    if const_cols:
        cleaned = cleaned.drop(columns=const_cols)
        constant_cols_removed = len(const_cols)
        cols_removed += constant_cols_removed
        dropped_col_names.extend(const_cols)

    # ------------------------------------------------------------------ #
    # 4. Fill missing values
    # ------------------------------------------------------------------ #
    numeric_cols = cleaned.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = cleaned.select_dtypes(exclude=[np.number]).columns.tolist()

    if strategy == "drop":
        before = len(cleaned)
        cleaned = cleaned.dropna()
        rows_removed += before - len(cleaned)
    else:
        # Numeric imputation
        for col in numeric_cols:
            n_missing = int(cleaned[col].isnull().sum())
            if n_missing == 0:
                continue
            if strategy == "mean":
                fill_value = cleaned[col].mean()
            else:  # "auto" | "median"
                fill_value = cleaned[col].median()
            cleaned[col] = cleaned[col].fillna(fill_value)
            missing_filled += n_missing

        # Categorical imputation (mode)
        for col in cat_cols:
            n_missing = int(cleaned[col].isnull().sum())
            if n_missing == 0:
                continue
            mode_series = cleaned[col].mode()
            if mode_series.empty:
                continue
            cleaned[col] = cleaned[col].fillna(mode_series.iloc[0])
            missing_filled += n_missing

    if clip_outliers:
        feature_numeric_cols = [c for c in numeric_cols if c != label_col]
        for col in feature_numeric_cols:
            series = cleaned[col].dropna()
            if series.empty:
                continue
            lower = series.quantile(0.01)
            upper = series.quantile(0.99)
            mask = (cleaned[col] < lower) | (cleaned[col] > upper)
            clipped_count = int(mask.sum())
            if clipped_count:
                cleaned[col] = cleaned[col].clip(lower=lower, upper=upper)
                outlier_values_clipped += clipped_count

    cleaned = cleaned.reset_index(drop=True)

    # ------------------------------------------------------------------ #
    # 5. Build summary
    # ------------------------------------------------------------------ #
    summary_parts = [
        f"Original shape: {initial_rows} rows × {initial_cols} cols.",
        f"Final shape:    {cleaned.shape[0]} rows × {cleaned.shape[1]} cols.",
        f"Duplicate rows removed: {rows_removed}.",
        f"Columns dropped (high-missing or constant): {cols_removed}"
        + (f" ({', '.join(dropped_col_names)})" if dropped_col_names else "") + ".",
        f"Missing values filled: {missing_filled}.",
        f"Missing-value strategy: {strategy}.",
        f"Outlier values clipped: {outlier_values_clipped}." if clip_outliers else "Outlier clipping: disabled.",
    ]
    summary = "  ".join(summary_parts)

    report: Dict = {
        "rows_removed": rows_removed,
        "cols_removed": cols_removed,
        "missing_filled": missing_filled,
        "constant_cols_removed": constant_cols_removed,
        "outlier_values_clipped": outlier_values_clipped,
        "summary": summary,
    }
    return cleaned, report


def clean_text_dataframe(
    df: pd.DataFrame,
    text_col: str,
    *,
    lowercase: bool = True,
    strip_urls: bool = True,
    strip_punctuation: bool = False,
    remove_stopwords: bool = False,
    deduplicate: bool = False,
    apply_stemming: bool = False,
    apply_lemmatization: bool = False,
    drop_empty_rows: bool = True,
) -> Tuple[pd.DataFrame, Dict]:
    """Apply lightweight text normalization and return a report."""
    cleaned = df.copy()
    if text_col not in cleaned.columns:
        raise ValueError(f"Text column '{text_col}' not found.")

    original_rows = len(cleaned)
    series = cleaned[text_col].fillna("").astype(str)

    if strip_urls:
        series = series.str.replace(r"https?://\S+|www\.\S+", " ", regex=True)
    if lowercase:
        series = series.str.lower()
    if strip_punctuation:
        series = series.str.replace(r"[^\w\s]", " ", regex=True)

    if remove_stopwords:
        series = series.apply(
            lambda value: " ".join(
                token for token in value.split() if token not in _BASIC_STOPWORDS
            )
        )

    if apply_lemmatization:
        series = series.apply(
            lambda value: " ".join(_lemmatize_token(token) for token in value.split())
        )
    if apply_stemming:
        series = series.apply(
            lambda value: " ".join(_stem_token(token) for token in value.split())
        )

    series = (
        series.str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )
    cleaned[text_col] = series

    duplicate_rows_removed = 0
    if deduplicate:
        before = len(cleaned)
        cleaned = cleaned.drop_duplicates(subset=[text_col]).copy()
        duplicate_rows_removed = before - len(cleaned)

    rows_removed = 0
    if drop_empty_rows:
        before = len(cleaned)
        cleaned = cleaned[cleaned[text_col].astype(str).str.len() > 0].copy()
        rows_removed = before - len(cleaned)

    report = {
        "rows_removed": rows_removed,
        "duplicate_rows_removed": duplicate_rows_removed,
        "summary": (
            f"Original rows: {original_rows}. Final rows: {len(cleaned)}. "
            f"Lowercase: {'on' if lowercase else 'off'}. "
            f"Strip URLs: {'on' if strip_urls else 'off'}. "
            f"Strip punctuation: {'on' if strip_punctuation else 'off'}. "
            f"Remove stop words: {'on' if remove_stopwords else 'off'}. "
            f"Deduplicate text rows: {'on' if deduplicate else 'off'}. "
            f"Stemming: {'on' if apply_stemming else 'off'}. "
            f"Lemmatization: {'on' if apply_lemmatization else 'off'}."
        ),
    }
    return cleaned.reset_index(drop=True), report


def _stem_token(token: str) -> str:
    token = token.strip()
    for suffix in ("ingly", "edly", "ing", "ed", "ly", "s"):
        if len(token) > len(suffix) + 2 and token.endswith(suffix):
            return token[: -len(suffix)]
    return token


def _lemmatize_token(token: str) -> str:
    token = token.strip()
    irregular = {"children": "child", "mice": "mouse", "geese": "goose", "better": "good"}
    if token in irregular:
        return irregular[token]
    for suffix, replacement in (("ies", "y"), ("ves", "f"), ("men", "man")):
        if len(token) > len(suffix) + 1 and token.endswith(suffix):
            return token[: -len(suffix)] + replacement
    return token


def clean_timeseries_dataframe(
    df: pd.DataFrame,
    label_col: str,
    *,
    time_col: Optional[str] = None,
    sort_by_time: bool = True,
    fill_strategy: TimeseriesFillStrategy = "none",
) -> Tuple[pd.DataFrame, Dict]:
    """Apply basic sorting and missing-value handling for timeseries data."""
    cleaned = df.copy()
    original_rows = len(cleaned)

    if sort_by_time and time_col and time_col in cleaned.columns:
        cleaned = cleaned.sort_values(time_col)

    feature_cols = [c for c in cleaned.columns if c not in {label_col, time_col}]
    numeric_cols = cleaned[feature_cols].select_dtypes(include=[np.number]).columns.tolist()

    missing_before = int(cleaned[feature_cols].isna().sum().sum()) if feature_cols else 0
    if fill_strategy == "forward_fill":
        cleaned[feature_cols] = cleaned[feature_cols].ffill().bfill()
    elif fill_strategy == "interpolate":
        if numeric_cols:
            cleaned[numeric_cols] = cleaned[numeric_cols].interpolate(limit_direction="both")
        non_numeric = [c for c in feature_cols if c not in numeric_cols]
        if non_numeric:
            cleaned[non_numeric] = cleaned[non_numeric].ffill().bfill()

    if feature_cols:
        cleaned = cleaned.dropna(subset=feature_cols)

    report = {
        "rows_removed": original_rows - len(cleaned),
        "summary": (
            f"Original rows: {original_rows}. Final rows: {len(cleaned)}. "
            f"Sorted by time: {'yes' if sort_by_time and time_col else 'no'}. "
            f"Fill strategy: {fill_strategy}. "
            f"Missing feature values before cleaning: {missing_before}."
        ),
    }
    return cleaned.reset_index(drop=True), report


# ---------------------------------------------------------------------------
# Outlier detection
# ---------------------------------------------------------------------------

def detect_outliers(
    df: pd.DataFrame,
    label_col: str,
    method: Literal["iqr", "zscore"] = "iqr",
) -> Dict[str, Dict]:
    """Detect outliers in numeric feature columns.

    Parameters
    ----------
    df:
        DataFrame to analyse.
    label_col:
        Column to exclude from outlier analysis.
    method:
        ``"iqr"``    — flags values outside ``[Q1 − 1.5·IQR, Q3 + 1.5·IQR]``.
        ``"zscore"`` — flags values where ``|z| > 3``.

    Returns
    -------
    dict mapping column name → ``{"n_outliers": int, "pct": float}``.
    ``pct`` is the percentage (0–100) of non-null values flagged.
    """
    result: Dict[str, Dict] = {}
    numeric_cols = [
        c for c in df.select_dtypes(include=[np.number]).columns if c != label_col
    ]

    for col in numeric_cols:
        series = df[col].dropna()
        if series.empty:
            result[col] = {"n_outliers": 0, "pct": 0.0}
            continue

        if method == "iqr":
            q1 = series.quantile(0.25)
            q3 = series.quantile(0.75)
            iqr = q3 - q1
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            mask = (series < lower) | (series > upper)
        elif method == "zscore":
            std = series.std(ddof=0)
            if std == 0.0:
                # Constant column — no outliers by definition
                logger.debug("Column '%s' has zero standard deviation; skipping zscore.", col)
                result[col] = {"n_outliers": 0, "pct": 0.0}
                continue
            z = (series - series.mean()) / std
            mask = z.abs() > 3.0
        else:
            raise ValueError(f"Unknown method '{method}'. Use 'iqr' or 'zscore'.")

        n_outliers = int(mask.sum())
        pct = (n_outliers / len(series)) * 100.0
        result[col] = {"n_outliers": n_outliers, "pct": round(pct, 4)}

    return result


# ---------------------------------------------------------------------------
# Markdown report
# ---------------------------------------------------------------------------

def cleaning_report_markdown(report_dict: Dict) -> str:
    """Render a cleaning report dict as a Markdown-formatted string.

    Parameters
    ----------
    report_dict:
        Dictionary as returned by :func:`clean_dataframe`.

    Returns
    -------
    Markdown string.
    """
    lines = [
        "## Data Cleaning Report",
        "",
        "| Metric | Value |",
        "| --- | --- |",
        f"| Duplicate rows removed | {report_dict['rows_removed']} |",
        f"| Columns removed | {report_dict['cols_removed']} |",
        f"| Constant columns removed | {report_dict['constant_cols_removed']} |",
        f"| Missing values filled | {report_dict['missing_filled']} |",
        "",
        "> ⚠️ **Note:** Imputation statistics (median/mean) are computed on the full dataset "
        "before train/val splitting. This is a minor form of data leakage acceptable for "
        "exploratory work; for rigorous experiments, split first then clean.",
        "",
        "### Summary",
        "",
        report_dict.get("summary", "No summary available."),
    ]
    return "\n".join(lines)

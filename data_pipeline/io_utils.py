from __future__ import annotations

from pathlib import Path


_CSV_ENCODINGS = ("utf-8", "utf-8-sig", "cp1252", "latin-1")


def read_structured_file(path: str | Path, *, nrows: int | None = None):
    import pandas as pd

    path = Path(path)
    suffix = path.suffix.lower()

    if suffix == ".csv":
        last_error = None
        for encoding in _CSV_ENCODINGS:
            try:
                return pd.read_csv(path, nrows=nrows, encoding=encoding)
            except UnicodeDecodeError as exc:
                last_error = exc
                continue
        if last_error is not None:
            raise last_error
        return pd.read_csv(path, nrows=nrows)

    if suffix == ".tsv":
        last_error = None
        for encoding in _CSV_ENCODINGS:
            try:
                return pd.read_csv(path, sep="\t", nrows=nrows, encoding=encoding)
            except UnicodeDecodeError as exc:
                last_error = exc
                continue
        if last_error is not None:
            raise last_error
        return pd.read_csv(path, sep="\t", nrows=nrows)

    if suffix == ".json":
        try:
            return pd.read_json(path, lines=True)
        except ValueError:
            return pd.read_json(path)

    raise ValueError(f"Unsupported structured file type: {path.suffix}")


def drop_missing_label_rows(df, label_col: str):
    if label_col not in df.columns:
        return df, 0
    before = len(df)
    cleaned = df.dropna(subset=[label_col]).copy()
    return cleaned, before - len(cleaned)


def apply_random_subset(df, enabled: bool = False, subset_percent: float = 100.0, subset_seed: int = 42):
    if not enabled:
        return df.copy(), len(df)

    pct = float(subset_percent)
    if pct <= 0:
        raise ValueError("Subset percent must be greater than 0.")
    if pct >= 100 or df.empty:
        return df.copy(), len(df)

    sample_size = max(1, int(round(len(df) * (pct / 100.0))))
    sample_size = min(sample_size, len(df))
    sampled = df.sample(n=sample_size, random_state=int(subset_seed)).copy()
    return sampled, sample_size

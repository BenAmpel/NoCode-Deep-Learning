from __future__ import annotations

from pathlib import Path


_CSV_ENCODINGS = ("utf-8", "utf-8-sig", "cp1252", "latin-1")


MAX_ZIP_TOTAL_BYTES = 20 * 1024 ** 3
MAX_ZIP_FILES = 200_000
MAX_ZIP_RATIO = 500  # uncompressed / compressed, per file; zip bombs reach 1000s


def safe_extract_zip(zip_path: str | Path, dest: str | Path, *,
                     max_total_bytes: int = MAX_ZIP_TOTAL_BYTES,
                     max_files: int = MAX_ZIP_FILES) -> None:
    """Extract an uploaded zip after checking it can't fill the disk or escape ``dest``.

    Everything is checked from the archive's directory before any byte is
    written: total size (and free disk space), file count, per-file
    compression ratio, and that every member resolves inside ``dest``.
    Raises ValueError with a message meant for the user.
    """
    import shutil
    import zipfile

    dest = Path(dest).resolve()
    with zipfile.ZipFile(zip_path) as zf:
        members = [m for m in zf.infolist() if not m.is_dir()]
        if len(members) > max_files:
            raise ValueError(f"The zip has {len(members):,} files; the limit is {max_files:,}.")
        total = sum(m.file_size for m in members)
        free = shutil.disk_usage(dest).free
        if total > max_total_bytes or total > free * 0.9:
            raise ValueError(
                f"The zip expands to {total / 1024 ** 3:.1f} GB, more than the "
                f"{min(max_total_bytes, free * 0.9) / 1024 ** 3:.1f} GB that can be extracted here."
            )
        for m in members:
            if m.compress_size and m.file_size / m.compress_size > MAX_ZIP_RATIO:
                raise ValueError(f"'{m.filename}' is compressed suspiciously well; refusing to extract it.")
            if not (dest / m.filename).resolve().is_relative_to(dest):
                raise ValueError(f"'{m.filename}' would be written outside the extraction folder.")
        zf.extractall(dest)


def _reread_if_other_delimiter(df, path: Path, nrows: int | None, encoding: str, dtype=None):
    """A ".csv" that parsed as one column is usually ;-, tab- or |-separated
    (semicolons are the norm where the decimal mark is a comma). Detect the
    delimiter from the header and re-read; files that parse fine are untouched.
    """
    import csv

    import pandas as pd

    if df.shape[1] != 1:
        return df
    header = str(df.columns[0])
    candidates = [d for d in (";", "\t", "|") if d in header]
    if not candidates:
        return df
    with open(path, encoding=encoding, newline="") as fh:
        sample = fh.read(64 * 1024)
    try:
        delimiter = csv.Sniffer().sniff(sample, delimiters="".join(candidates)).delimiter
    except csv.Error:
        delimiter = max(candidates, key=header.count)
    # ;-separated files come from locales that write decimals as "1,5".
    decimal = "," if delimiter == ";" else "."
    reread = pd.read_csv(path, nrows=nrows, encoding=encoding, sep=delimiter, decimal=decimal, dtype=dtype)
    return reread if reread.shape[1] > 1 else df


def read_structured_file(path: str | Path, *, nrows: int | None = None, dtype=None):
    """Read a structured table file into a DataFrame.

    ``dtype`` is passed to ``pandas.read_csv`` for .csv and .tsv files (for
    example ``{"id": str}`` to keep leading zeros); other formats ignore it.
    """
    import pandas as pd

    path = Path(path)
    suffix = path.suffix.lower()

    if suffix == ".csv":
        last_error = None
        for encoding in _CSV_ENCODINGS:
            try:
                df = pd.read_csv(path, nrows=nrows, encoding=encoding, dtype=dtype)
            except UnicodeDecodeError as exc:
                last_error = exc
                continue
            return _reread_if_other_delimiter(df, path, nrows, encoding, dtype)
        if last_error is not None:
            raise last_error
        return pd.read_csv(path, nrows=nrows, dtype=dtype)

    if suffix == ".tsv":
        last_error = None
        for encoding in _CSV_ENCODINGS:
            try:
                return pd.read_csv(path, sep="\t", nrows=nrows, encoding=encoding, dtype=dtype)
            except UnicodeDecodeError as exc:
                last_error = exc
                continue
        if last_error is not None:
            raise last_error
        return pd.read_csv(path, sep="\t", nrows=nrows, dtype=dtype)

    if suffix == ".json":
        nested = _read_nested_json(path)
        if nested is not None:
            return nested.head(nrows) if nrows is not None else nested
        # Parse as one document first. A one-line array or object also parses as
        # "JSON Lines" (a single row holding everything), so lines=True is only
        # the fallback for files that are not one document.
        try:
            df = pd.read_json(path)
        except ValueError:
            df = pd.read_json(path, lines=True)
        return df.head(nrows) if nrows is not None else df

    raise ValueError(f"Unsupported structured file type: {path.suffix}")


_MAX_NESTED_JSON_BYTES = 200 * 1024 * 1024


def _is_table_member(value) -> bool:
    """A dict of dicts ({date: {currency: rate}}), a dict of equal-length
    lists ({"time": [...], "temp": [...]}), or a list of dicts (records)."""
    if isinstance(value, dict) and value:
        vals = list(value.values())
        if all(isinstance(v, dict) for v in vals):
            return True
        if all(isinstance(v, list) for v in vals):
            return len({len(v) for v in vals}) == 1 and len(vals[0]) > 0
        return False
    if isinstance(value, list) and value:
        return all(isinstance(v, dict) for v in value)
    return False


def _member_rows(value) -> int:
    if isinstance(value, list):
        return len(value)
    vals = list(value.values())
    return len(vals[0]) if isinstance(vals[0], list) else len(vals)


def _table_from_member(name: str, value):
    import pandas as pd

    if isinstance(value, list):
        return pd.json_normalize(value)
    vals = list(value.values())
    if isinstance(vals[0], list):
        return pd.DataFrame(value)
    # {key: {col: value}}: one row per key. Keys that parse as dates (the
    # Frankfurter API shape) become a "date" column.
    df = pd.json_normalize(vals)
    keys = pd.Series([str(k) for k in value.keys()])
    looks_like_dates = keys.str.match(r"^\d{4}-\d{2}-\d{2}").all()
    key_col = "date" if looks_like_dates else name
    if key_col in df.columns:
        key_col = f"{key_col}_key"
    df.insert(0, key_col, keys.to_numpy())
    return df


def _read_nested_json(path: Path):
    """Flatten common API response shapes that ``pandas.read_json`` reads badly.

    Returns a DataFrame for:

    * a top-level object that wraps one table next to scalar metadata, e.g.
      Frankfurter's ``{"base": "EUR", "rates": {date: {currency: rate}}}`` or
      Open-Meteo's ``{"latitude": ..., "hourly": {"time": [...], ...}}``;
    * a list of records with nested objects (flattened to ``a.b`` columns).

    Returns None for everything else (JSON Lines, pandas' own orients, flat
    records), so those keep loading exactly as before.
    """
    import json

    import pandas as pd

    try:
        if path.stat().st_size > _MAX_NESTED_JSON_BYTES:
            return None
        with open(path, encoding="utf-8-sig") as fh:
            obj = json.load(fh)
    except (OSError, ValueError, UnicodeDecodeError):
        return None

    if isinstance(obj, list):
        if obj and all(isinstance(r, dict) for r in obj) and any(
            isinstance(v, dict) for r in obj for v in r.values()
        ):
            return pd.json_normalize(obj)
        return None
    if not isinstance(obj, dict) or not obj:
        return None
    # Every value a dict, or every value a list: pandas' own orients
    # ("columns", "index", "split", ...). Leave those to read_json.
    vals = list(obj.values())
    if all(isinstance(v, dict) for v in vals) or all(isinstance(v, list) for v in vals):
        return None
    members = {k: v for k, v in obj.items() if _is_table_member(v)}
    if not members:
        return None
    name = max(members, key=lambda k: _member_rows(members[k]))
    return _table_from_member(str(name), members[name])


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

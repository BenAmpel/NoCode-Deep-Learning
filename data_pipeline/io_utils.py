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


def _reread_if_other_delimiter(df, path: Path, nrows: int | None, encoding: str):
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
    reread = pd.read_csv(path, nrows=nrows, encoding=encoding, sep=delimiter, decimal=decimal)
    return reread if reread.shape[1] > 1 else df


def read_structured_file(path: str | Path, *, nrows: int | None = None):
    import pandas as pd

    path = Path(path)
    suffix = path.suffix.lower()

    if suffix == ".csv":
        last_error = None
        for encoding in _CSV_ENCODINGS:
            try:
                df = pd.read_csv(path, nrows=nrows, encoding=encoding)
            except UnicodeDecodeError as exc:
                last_error = exc
                continue
            return _reread_if_other_delimiter(df, path, nrows, encoding)
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

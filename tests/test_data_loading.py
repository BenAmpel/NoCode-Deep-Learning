import zipfile

import numpy as np
import pandas as pd
import pytest

from data_pipeline.io_utils import read_structured_file, safe_extract_zip
from data_pipeline.quality_report import build_quality_report
from data_pipeline.type_inference import infer_column_types, suggest_features_and_label


# ---------------------------------------------------------------------------
# Delimiters
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("body, columns, first_row", [
    ("a;b;label\n1,5;2;x\n", ["a", "b", "label"], [1.5, 2, "x"]),      # European CSV
    ("a\tb\tlabel\n1\t2\tx\n", ["a", "b", "label"], [1, 2, "x"]),
    ("a|b|label\n1|2|x\n", ["a", "b", "label"], [1, 2, "x"]),
    ('text,label\n"hi; there",x\n', ["text", "label"], ["hi; there", "x"]),  # comma file left alone
    ("label\nx\n", ["label"], ["x"]),
])
def test_csv_delimiter_detection(tmp_path, body, columns, first_row):
    path = tmp_path / "data.csv"
    path.write_text(body, encoding="utf-8")
    df = read_structured_file(path)
    assert list(df.columns) == columns
    assert df.iloc[0].tolist() == first_row


# ---------------------------------------------------------------------------
# Column types
# ---------------------------------------------------------------------------

def _frame():
    rng = np.random.default_rng(0)
    n = 40
    return pd.DataFrame({
        "humidity": rng.integers(20, 90, n), "width": rng.permutation(n) + 5,
        "valid_score": rng.integers(0, 1000, n), "user_id": rng.permutation(n),
        "recordID": rng.permutation(n) + 1000, "Unnamed: 0": range(n),
        "code": [f"c{i % 15}" for i in range(n)], 7: rng.normal(size=n),
        "label": ["a", "b"] * (n // 2),
    })


def test_id_detection_matches_whole_words_only():
    types = infer_column_types(_frame())
    assert types["humidity"] != "id" and types["width"] != "id" and types["valid_score"] != "id"
    assert types["user_id"] == "id" and types["recordID"] == "id" and types["Unnamed: 0"] == "id"


def test_consecutive_measurements_are_not_a_row_counter():
    df = pd.DataFrame({"floor": range(100, 140), "label": ["a", "b"] * 20})
    assert infer_column_types(df)["floor"] != "id"


def test_integer_column_names_do_not_crash():
    df = _frame()
    types = infer_column_types(df)
    assert types[7] == "numeric"
    assert 7 in suggest_features_and_label(df)["feature_cols"]


def test_moderate_cardinality_codes_are_not_dropped():
    df = _frame()
    assert "code" not in suggest_features_and_label(df)["drop_cols"]


# ---------------------------------------------------------------------------
# Quality report
# ---------------------------------------------------------------------------

def test_cross_mark_validation_messages_block_training(tmp_path):
    stats = {"n_samples": 10, "class_counts": {"a": 10}, "imbalance_ratio": 1.0}
    report = build_quality_report(str(tmp_path), "tabular", "label", stats,
                                  ["❌ Only 1 unique class in 'label'. Need at least 2."])
    assert report["blocking_issues"]
    assert report["status"] != "ready"


def test_leakage_check_ignores_words_containing_id(tmp_path):
    df = _frame()
    stats = {"n_samples": len(df), "class_counts": {"a": 20, "b": 20}, "imbalance_ratio": 1.0}
    report = build_quality_report(str(tmp_path), "tabular", "label", stats, [], preview_df=df)
    text = str(report)
    assert "user_id" in text
    assert "'humidity'" not in text and "'width'" not in text


# ---------------------------------------------------------------------------
# Zip uploads
# ---------------------------------------------------------------------------

def _zip(path, files):
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for name, data in files.items():
            zf.writestr(name, data)
    return path


def test_normal_zip_extracts(tmp_path):
    z = _zip(tmp_path / "ok.zip", {"cats/a.txt": b"meow", "dogs/b.txt": b"woof"})
    (tmp_path / "out").mkdir()
    safe_extract_zip(z, tmp_path / "out")
    assert (tmp_path / "out" / "cats" / "a.txt").read_bytes() == b"meow"


def test_zip_with_too_many_files_is_refused(tmp_path):
    z = _zip(tmp_path / "many.zip", {f"f{i}.txt": b"x" for i in range(20)})
    (tmp_path / "out").mkdir()
    with pytest.raises(ValueError, match="20 files"):
        safe_extract_zip(z, tmp_path / "out", max_files=10)
    assert not any((tmp_path / "out").iterdir())


def test_zip_larger_than_limit_is_refused(tmp_path):
    z = _zip(tmp_path / "big.zip", {"data.bin": np.random.default_rng(0).bytes(4096)})
    (tmp_path / "out").mkdir()
    with pytest.raises(ValueError, match="expands to"):
        safe_extract_zip(z, tmp_path / "out", max_total_bytes=1024)


def test_zip_bomb_is_refused(tmp_path):
    z = _zip(tmp_path / "bomb.zip", {"zeros.bin": b"\0" * (10 * 1024 * 1024)})
    (tmp_path / "out").mkdir()
    with pytest.raises(ValueError, match="compressed suspiciously well"):
        safe_extract_zip(z, tmp_path / "out")
    assert not (tmp_path / "out" / "zeros.bin").exists()


def test_zip_member_escaping_the_folder_is_refused(tmp_path):
    z = _zip(tmp_path / "evil.zip", {"../evil.txt": b"x"})
    (tmp_path / "out").mkdir()
    with pytest.raises(ValueError, match="outside the extraction folder"):
        safe_extract_zip(z, tmp_path / "out")
    assert not (tmp_path / "evil.txt").exists()

import numpy as np
import pandas as pd

from data_pipeline.cleaning import clean_dataframe, cleaning_report_markdown


def _frame(labels):
    return pd.DataFrame({
        "x1": [1.0, 2.0, np.nan, 4.0, 5.0, 6.0],
        "x2": ["a", "b", "a", None, "b", "a"],
        "label": labels,
    })


def test_missing_categorical_labels_are_dropped_not_imputed():
    df = _frame(["a", "b", None, "a", None, "b"])
    cleaned, report = clean_dataframe(df, "label")
    assert cleaned["label"].notna().all()
    assert len(cleaned) == 4
    assert report["missing_label_rows_removed"] == 2
    # The surviving labels are the original ones, in order.
    assert cleaned["label"].tolist() == ["a", "b", "a", "b"]


def test_missing_numeric_labels_are_dropped_not_imputed():
    df = _frame([10.0, np.nan, 30.0, 40.0, 50.0, 60.0])
    cleaned, report = clean_dataframe(df, "label", strategy="median")
    assert report["missing_label_rows_removed"] == 1
    assert 35.0 not in cleaned["label"].tolist()  # median of the others
    assert cleaned["label"].tolist() == [10.0, 30.0, 40.0, 50.0, 60.0]


def test_feature_values_are_still_imputed():
    df = _frame(["a", "b", "a", "b", "a", "b"])
    cleaned, report = clean_dataframe(df, "label", strategy="median")
    assert cleaned[["x1", "x2"]].notna().all().all()
    assert report["missing_filled"] == 2


def test_report_markdown_lists_missing_label_rows():
    df = _frame(["a", None, "a", "b", "a", "b"])
    _, report = clean_dataframe(df, "label")
    assert "Rows dropped for missing label | 1" in cleaning_report_markdown(report)

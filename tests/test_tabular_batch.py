"""Batch prediction from a CSV for tabular bundles (Try Your Model → Batch).

Small untrained models are exported through the real exporter; every batch
row must score exactly like the single-row "Try Your Model" path.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from export.exporter import export_bundle

FEATURES = ["age", "income", "city"]


@pytest.fixture
def train_csv(tmp_path):
    rng = np.random.default_rng(0)
    n = 40
    df = pd.DataFrame({
        "age": rng.integers(18, 70, n).astype(float),
        "income": rng.normal(50_000, 10_000, n),
        "city": rng.choice(["oslo", "lima", "pune"], n),
        "label": rng.choice(["yes", "no", "maybe"], n),
        "price": rng.normal(200, 50, n),
    })
    path = tmp_path / "train.csv"
    df.to_csv(path, index=False)
    return path


def _export_mlp(tmp_path, train_csv, *, task="classification", label="label", name="tab"):
    from models.tabular_models import get_tabular_model
    from modalities.tabular import load_tabular_data

    _, val_loader, classes, prep, n_in, *_ = load_tabular_data(
        str(train_csv), label_col=label, feature_cols=FEATURES, augmentation="none", task=task)
    num_out = 1 if task == "regression" else len(classes)
    model = get_tabular_model("MLP", num_classes=num_out, input_size=n_in, dropout=0.0).eval()
    bundle, _ = export_bundle(model, prep, classes, name, output_dir=str(tmp_path / "out"),
                              sample_batch=next(iter(val_loader))[0])
    return bundle


def _write_batch(tmp_path, rows, name="batch.csv"):
    # Written by hand so IDs like 00123 reach the file exactly as typed.
    cols = list(rows[0])
    lines = [",".join(cols)] + [",".join(str(r[c]) for c in cols) for r in rows]
    path = tmp_path / name
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


BATCH_ROWS = [
    {"applicant_id": "00123", "age": 33, "income": 61000.0, "city": "lima"},
    {"applicant_id": "00456", "age": 52, "income": 42000.5, "city": "oslo"},
    {"applicant_id": "A-9", "age": 21, "income": 75000.0, "city": "pune"},
]


def test_classification_matches_single_row_path_and_saves_numeric_confidence(tmp_path, train_csv, monkeypatch):
    from ui.callbacks import infer
    from ui.inference_helpers import predict_tabular

    bundle = _export_mlp(tmp_path, train_csv)
    csv_path = _write_batch(tmp_path, BATCH_ROWS)

    preview, status, full = infer._batch_predict(bundle, "", str(csv_path), "applicant_id")
    assert status.startswith("✅"), status
    from ui.tabular_batch import class_labels_in_order
    from ui.inference_helpers import _runtime_bundle
    classes = class_labels_in_order(_runtime_bundle(bundle))
    prob_cols = [f"prob_{c}" for c in classes]
    assert list(full.columns) == ["applicant_id", "prediction", "confidence", "confidence_score", *prob_cols]
    assert full["applicant_id"].tolist() == ["00123", "00456", "A-9"]
    assert len(preview) == 3

    for i, row in enumerate(BATCH_ROWS):
        ranked = predict_tabular(bundle, {k: row[k] for k in FEATURES}, top_k=len(classes))
        single = ranked[0]
        assert full["prediction"][i] == single["label"]
        assert full["confidence_score"][i] == pytest.approx(single["confidence"], abs=1e-6)
        assert full["confidence"][i] == f"{single['confidence'] * 100:.1f}%"
        for p in ranked:
            assert full[f"prob_{p['label']}"][i] == pytest.approx(p["confidence"], abs=1e-6)
        assert sum(full[c][i] for c in prob_cols) == pytest.approx(1.0, abs=1e-4)

    # Save writes the full results from state; read the file back.
    monkeypatch.setattr(infer, "_OUTPUTS_ROOT", tmp_path / "saved")
    msg = infer._batch_save_csv(full)
    assert msg.startswith("✅"), msg
    saved_path = next((tmp_path / "saved").glob("batch_predictions_*.csv"))
    saved = pd.read_csv(saved_path, dtype={"applicant_id": str})
    assert saved["applicant_id"].tolist() == ["00123", "00456", "A-9"]
    assert saved_path.name.startswith("batch_predictions_")
    for col in ["confidence_score", *prob_cols]:
        assert pd.api.types.is_float_dtype(saved[col])
        assert saved[col].between(0, 1).all()


def test_every_non_feature_column_is_passed_through(tmp_path, train_csv):
    from ui.tabular_batch import predict_csv

    bundle = _export_mlp(tmp_path, train_csv)
    rows = [
        dict(r, student_code=code, label=lab, prediction=old)
        for r, code, lab, old in zip(BATCH_ROWS, ["007", "042", "100"], ["yes", "no", "maybe"], ["x", "y", "z"])
    ]
    df, _ = predict_csv(bundle, _write_batch(tmp_path, rows), "student_code")
    # ID first, then the other non-feature columns in file order; features are not copied.
    assert list(df.columns[:4]) == ["student_code", "applicant_id", "label", "prediction_input"]
    assert "age" not in df.columns and "city" not in df.columns
    assert df["student_code"].tolist() == ["007", "042", "100"]
    assert df["applicant_id"].tolist() == ["00123", "00456", "A-9"]
    assert df["label"].tolist() == ["yes", "no", "maybe"]
    assert df["prediction_input"].tolist() == ["x", "y", "z"]
    assert set(df["prediction"]) <= {"yes", "no", "maybe"}

    # With no ID chosen the same columns still come through, in file order.
    df2, _ = predict_csv(bundle, _write_batch(tmp_path, rows, name="b2.csv"))
    assert list(df2.columns[:4]) == ["applicant_id", "student_code", "label", "prediction_input"]


def test_regression_writes_predicted_value(tmp_path, train_csv):
    from ui.inference_helpers import predict_tabular
    from ui.tabular_batch import predict_csv

    bundle = _export_mlp(tmp_path, train_csv, task="regression", label="price", name="tabr")
    csv_path = _write_batch(tmp_path, BATCH_ROWS)

    df, _ = predict_csv(bundle, csv_path, "applicant_id")
    assert list(df.columns) == ["applicant_id", "prediction"]
    assert pd.api.types.is_float_dtype(df["prediction"])
    for i, row in enumerate(BATCH_ROWS):
        expected = predict_tabular(bundle, {k: row[k] for k in FEATURES})[0]["value"]
        assert df["prediction"][i] == pytest.approx(expected, abs=1e-5)


def test_unseen_category_is_scored_and_reported(tmp_path, train_csv):
    from export import bundle_runtime as rt
    from ui.inference_helpers import predict_tabular
    from ui.tabular_batch import predict_csv

    bundle = _export_mlp(tmp_path, train_csv)
    rows = [dict(BATCH_ROWS[0], city="tokyo"), BATCH_ROWS[1]]
    df, notes = predict_csv(bundle, _write_batch(tmp_path, rows), "applicant_id")

    assert df["prediction"][0] != "ERROR"
    single = predict_tabular(bundle, {"age": 33, "income": 61000.0, "city": "tokyo"}, top_k=1)[0]
    assert df["confidence_score"][0] == pytest.approx(single["confidence"], abs=1e-6)
    # Every one-hot slot for city is 0 for the unseen value.
    x = rt.Bundle(bundle).preprocess({"age": 33, "income": 61000.0, "city": "tokyo"})
    assert x[0, 2:].sum() == 0
    assert "tokyo" in notes and "`city`" in notes


def test_missing_feature_column_gives_clear_error(tmp_path, train_csv):
    from ui.callbacks import infer
    from ui.tabular_batch import TabularBatchError, predict_csv

    bundle = _export_mlp(tmp_path, train_csv)
    rows = [{k: v for k, v in r.items() if k != "income"} for r in BATCH_ROWS]
    csv_path = _write_batch(tmp_path, rows)

    with pytest.raises(TabularBatchError, match="`income`"):
        predict_csv(bundle, csv_path, "applicant_id")
    preview, status, full = infer._batch_predict(bundle, "", str(csv_path), "applicant_id")
    assert preview is None and full is None
    assert "missing feature column" in status and "`income`" in status


def test_tabular_bundle_without_csv_asks_for_one(tmp_path, train_csv):
    from ui.callbacks import infer

    bundle = _export_mlp(tmp_path, train_csv)
    preview, status, full = infer._batch_predict(bundle, "/some/folder", None, None)
    assert preview is None and full is None
    assert "upload a CSV" in status


def test_digit_only_categories_still_match_training(tmp_path):
    """A category column that was text in training stays text in the batch CSV,
    even when the batch file holds only digits."""
    from ui.inference_helpers import predict_tabular
    from ui.tabular_batch import predict_csv

    rng = np.random.default_rng(1)
    n = 30
    train = pd.DataFrame({
        "x": rng.normal(size=n),
        "zone": rng.choice(["1", "2", "north"], n),
        "label": rng.choice(["a", "b"], n),
    })
    train_path = tmp_path / "zones.csv"
    train.to_csv(train_path, index=False)

    from models.tabular_models import get_tabular_model
    from modalities.tabular import load_tabular_data
    _, val_loader, classes, prep, n_in, *_ = load_tabular_data(
        str(train_path), label_col="label", augmentation="none")
    assert "zone" in prep["categorical_cols"]
    model = get_tabular_model("MLP", num_classes=len(classes), input_size=n_in, dropout=0.0).eval()
    bundle, _ = export_bundle(model, prep, classes, "zones", output_dir=str(tmp_path / "out"),
                              sample_batch=next(iter(val_loader))[0])

    batch = _write_batch(tmp_path, [{"x": 0.5, "zone": 1}, {"x": -0.2, "zone": 2}], name="zb.csv")
    df, notes = predict_csv(bundle, batch)
    assert df.columns[0] == "row" and df["row"].tolist() == [1, 2]
    assert "not seen in training" not in notes
    for i, zone in enumerate(["1", "2"]):
        single = predict_tabular(bundle, {"x": [0.5, -0.2][i], "zone": zone}, top_k=1)[0]
        assert df["confidence_score"][i] == pytest.approx(single["confidence"], abs=1e-6)


def test_autoencoder_bundle_exports_anomaly_scores(tmp_path, train_csv):
    from export import bundle_runtime as rt
    from models.autoencoder import get_autoencoder
    from modalities.tabular import load_tabular_data
    from ui.tabular_batch import predict_csv

    _, val_loader, classes, prep, n_in, *_ = load_tabular_data(
        str(train_csv), label_col="label", feature_cols=FEATURES, augmentation="none")
    prep["model_name"] = "Autoencoder"  # as pipeline/export.py records it
    model = get_autoencoder("tabular", input_size=n_in, latent_dim=4).eval()
    bundle, _ = export_bundle(model, prep, classes, "ae", output_dir=str(tmp_path / "out"),
                              sample_batch=next(iter(val_loader))[0])
    if not (Path(bundle) / "model.onnx").is_file():
        pytest.skip("Autoencoder ONNX export unavailable in this environment")

    df, _ = predict_csv(bundle, _write_batch(tmp_path, BATCH_ROWS), "applicant_id")
    assert list(df.columns) == ["applicant_id", "anomaly_score"]
    for i, row in enumerate(BATCH_ROWS):
        x = rt.Bundle(bundle).preprocess({k: row[k] for k in FEATURES})
        with torch.no_grad():
            recon = model(torch.from_numpy(x)).numpy()
        assert df["anomaly_score"][i] == pytest.approx(float(np.mean((x - recon) ** 2)), rel=1e-4, abs=1e-6)


def test_default_id_column_prefers_id_like_names():
    from ui.tabular_batch import default_id_column

    prep = {"numeric_cols": ["age"], "categorical_cols": ["city"]}
    assert default_id_column(["age", "city", "transaction_id"], prep) == "transaction_id"
    assert default_id_column(["age", "ref", "city"], prep) == "ref"
    assert default_id_column(["age", "city"], prep) is None

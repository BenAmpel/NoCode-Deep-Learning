"""Every exported bundle carries a run record (run_record.json + RUN_RECORD.md)."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from export import run_record
from pipeline.context import RunResult
from pipeline.runner import run
from test_pipeline import TABULAR, _config, _isolated_state  # noqa: F401  (autouse fixture)


def _strict_load(path: Path) -> dict:
    def no_constants(name):
        raise ValueError(f"non-standard JSON constant {name}")

    return json.loads(path.read_text(encoding="utf-8"), parse_constant=no_constants)


def _run(tmp_path, monkeypatch, **changes) -> tuple[RunResult, list]:
    import pipeline.runner as runner

    monkeypatch.setattr(runner, "OUTPUTS_ROOT", tmp_path / "bundles")
    items = list(run(_config(**changes)))
    assert isinstance(items[-1], RunResult)
    return items[-1], items


@pytest.mark.parametrize("model_name,framework", [("MLP", "pytorch"), ("RandomForest", "scikit-learn")])
def test_every_bundle_gets_a_record_of_what_ran(tmp_path, monkeypatch, model_name, framework):
    import config

    result, _ = _run(tmp_path, monkeypatch, model_name=model_name)
    bundle = Path(result.bundle_path)
    record = _strict_load(bundle / run_record.JSON_NAME)
    md = (bundle / run_record.MARKDOWN_NAME).read_text(encoding="utf-8")

    rows = len(pd.read_csv(TABULAR))
    assert record["schema_version"] == run_record.SCHEMA_VERSION
    assert record["model"]["name"] == model_name
    assert record["model"]["trainable_framework"] == framework
    assert record["split"]["train_samples"] + record["split"]["validation_samples"] == rows
    assert record["split"]["method"] == "stratified by class"
    assert record["split"]["seed"] == 42
    assert record["training"]["seed"] == config.DEFAULTS["seed"]
    assert record["environment"]["training_device"] in {"cpu", "mps", "cuda"}
    assert record["metrics"]["values"] == pytest.approx(result.metrics)
    assert record["metrics"]["computed_on"] == "validation split"
    assert len(record["data"]["fingerprint"]["sha256"]) == 64
    assert record["settings"]["lr"] == pytest.approx(1e-3)
    if framework == "pytorch":
        assert record["model"]["parameters"] > 0
        assert record["training"]["epochs_completed"] == len(result.history) == 1

    # Shared with graders and reviewers: no absolute paths (they carry the account name).
    for text in (json.dumps(record), md):
        assert str(Path(TABULAR).parent) not in text
    assert record["settings"]["data_path"] == Path(TABULAR).name

    assert model_name in run_record.methods_paragraph(record)
    assert "## Methods paragraph" in md
    assert "no separate held-out test set" in md


def test_a_failing_record_never_fails_the_run(tmp_path, monkeypatch):
    def boom(*_a, **_k):
        raise RuntimeError("disk full")

    monkeypatch.setattr(run_record, "write_run_record", boom)
    result, _ = _run(tmp_path, monkeypatch, model_name="RandomForest")
    assert Path(result.bundle_path, "labels.json").is_file()
    assert not Path(result.bundle_path, run_record.JSON_NAME).exists()


def test_fingerprint_is_stable_and_tracks_content(tmp_path):
    data = tmp_path / "images"
    (data / "cat").mkdir(parents=True)
    (data / "dog").mkdir()
    (data / "cat" / "1.png").write_bytes(b"cat-one")
    (data / "dog" / "1.png").write_bytes(b"dog-one")

    first = run_record.fingerprint_data(str(data))
    assert first == run_record.fingerprint_data(str(data))
    assert (first["kind"], first["files"], first["bytes"]) == ("folder", 2, 14)

    (data / ".DS_Store").write_bytes(b"finder noise")
    assert run_record.fingerprint_data(str(data))["sha256"] == first["sha256"]

    (data / "dog" / "1.png").write_bytes(b"dog-two")
    assert run_record.fingerprint_data(str(data))["sha256"] != first["sha256"]

    (data / "dog" / "1.png").write_bytes(b"dog-one")
    (data / "dog" / "1.png").rename(data / "dog" / "2.png")
    assert run_record.fingerprint_data(str(data))["sha256"] != first["sha256"]


def test_fingerprint_of_a_single_file_and_of_nothing(tmp_path):
    csv = tmp_path / "data.csv"
    csv.write_text("a,b\n1,2\n")
    fp = run_record.fingerprint_data(str(csv))
    assert (fp["kind"], fp["name"], fp["files"]) == ("file", "data.csv", 1)
    assert run_record.fingerprint_data(str(tmp_path / "gone.csv"))["method"] == "unavailable"
    assert run_record.fingerprint_data("")["method"] == "unavailable"


def test_large_data_falls_back_to_names_and_sizes(tmp_path, monkeypatch):
    csv = tmp_path / "data.csv"
    csv.write_text("a,b\n1,2\n")
    monkeypatch.setattr(run_record, "CONTENT_HASH_LIMIT_BYTES", 1)
    assert "names and sizes" in run_record.fingerprint_data(str(csv))["method"]


def test_non_finite_numbers_become_null():
    import numpy as np

    out = run_record._jsonable({"mape": float("nan"), "r2": np.float32(0.5), "inf": float("inf"), "arr": np.arange(2)})
    assert out == {"mape": None, "r2": 0.5, "inf": None, "arr": [0, 1]}
    json.dumps(out, allow_nan=False)


_TS = str(Path(TABULAR).with_name("timeseries_stress.csv"))
_TS_COLS = dict(modality="timeseries", data_path=_TS, time_col="timestamp", label_col="target",
                feature_cols=["sensor_a", "sensor_b", "sensor_c"], model_name="LSTM")
_REG_COLS = dict(task="regression", label_col="project_score",
                 feature_cols=["study_hours", "attendance_pct", "practice_quizzes"])


@pytest.mark.parametrize("changes,split_method,has_seed", [
    (_REG_COLS, "random", True),
    (dict(task="clustering", model_name="Autoencoder"), "stratified by class", True),
    (dict(task="anomaly", model_name="Autoencoder"), "stratified by class", True),
    (_TS_COLS, "chronological", False),
], ids=["regression", "clustering", "anomaly-autoencoder", "timeseries"])
def test_record_describes_the_split_each_path_really_uses(tmp_path, monkeypatch, changes, split_method, has_seed):
    result, _ = _run(tmp_path, monkeypatch, **changes)
    bundle = Path(result.bundle_path)
    record = _strict_load(bundle / run_record.JSON_NAME)
    assert record["split"]["method"].startswith(split_method)
    assert (record["split"]["seed"] is not None) == has_seed
    assert record["split"]["train_samples"] and record["split"]["validation_samples"]
    paragraph = run_record.methods_paragraph(record)
    assert ("seed 42)" in paragraph) == has_seed
    assert "## Methods paragraph" in (bundle / run_record.MARKDOWN_NAME).read_text(encoding="utf-8")

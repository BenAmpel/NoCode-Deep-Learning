"""Exported bundles must serve the same predictions as the app and the model.

For each modality a small untrained model is exported through the real
exporter, then three answers are compared on the same raw input:

- the generated FastAPI server (main.py, via TestClient),
- the app's "Try Your Model" functions (ui.inference_helpers),
- the original PyTorch / scikit-learn model run on the preprocessed input.

Preprocessing parity with training itself is covered in
test_bundle_runtime_parity.py.
"""
import importlib.util
import json
import shutil
import subprocess
import sys
import uuid
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from fastapi.testclient import TestClient

import config
from export import bundle_runtime as rt
from export.exporter import export_bundle
from export.fastapi_generator import generate_fastapi_app

ROOT = Path(__file__).resolve().parents[1]
IMAGES = ROOT / "fixtures" / "image_stress_dataset"
AUDIO = ROOT / "fixtures" / "audio_stress_dataset"
VIDEO = ROOT / "fixtures" / "object_detection_test" / "bus_loop.mp4"


@pytest.fixture(autouse=True)
def _no_loader_workers(monkeypatch):
    monkeypatch.setitem(config.DEFAULTS, "num_workers", 0)
    torch.manual_seed(0)


def _serve(bundle: Path) -> TestClient:
    generate_fastapi_app(str(bundle))
    spec = importlib.util.spec_from_file_location(f"served_{uuid.uuid4().hex}", bundle / "main.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return TestClient(module.app)


def _torch_scores(model, x):
    model.eval()
    with torch.no_grad():
        if isinstance(x, dict):
            out = model(torch.from_numpy(x["input_ids"]), torch.from_numpy(x["attention_mask"]))
        else:
            out = model(torch.from_numpy(x))
    return out.numpy().reshape(-1)


def _check_classification(served: dict, in_app: list[dict], expected_probs: np.ndarray, labels: list[str]):
    assert served["task"] == "classification"
    best = int(np.argmax(expected_probs))
    assert served["predictions"][0]["label"] == labels[best]
    assert served["predictions"][0]["confidence"] == pytest.approx(float(expected_probs[best]), abs=1e-4)
    assert [p["label"] for p in in_app] == [p["label"] for p in served["predictions"]]
    for a, b in zip(in_app, served["predictions"]):
        assert a["confidence"] == pytest.approx(b["confidence"], abs=1e-6)


def _softmax(x):
    e = np.exp(x - x.max())
    return e / e.sum()


# ---------------------------------------------------------------------------
# File modalities
# ---------------------------------------------------------------------------

def test_image_bundle_serves_like_app_and_model(tmp_path):
    from models.image_models import get_image_model
    from modalities.image import load_image_data
    from ui.inference_helpers import predict_image

    _, val_loader, classes, prep, val_samples = load_image_data(
        str(IMAGES), mode="from_scratch", image_size=64, augmentation="none", force_grayscale=True)
    model = get_image_model("TinyCNN", num_classes=len(classes), mode="from_scratch").eval()
    bundle, _ = export_bundle(model, prep, classes, "img", output_dir=str(tmp_path),
                              sample_batch=next(iter(val_loader))[0])
    sample = val_samples[0][0]

    served = _serve(Path(bundle)).post("/predict", files={"file": ("x.png", Path(sample).read_bytes())}).json()
    probs = _softmax(_torch_scores(model, rt.Bundle(bundle).preprocess(sample)))
    _check_classification(served, predict_image(bundle, sample), probs, classes)


def test_audio_bundle_serves_like_app_and_model(tmp_path):
    from models.image_models import get_image_model
    from modalities.audio import load_audio_data
    from ui.inference_helpers import predict_audio

    _, val_loader, classes, prep = load_audio_data(str(AUDIO), sample_rate=16000, image_size=64, augmentation="none")
    model = get_image_model("TinyCNN", num_classes=len(classes), mode="from_scratch").eval()
    bundle, _ = export_bundle(model, prep, classes, "aud", output_dir=str(tmp_path),
                              sample_batch=next(iter(val_loader))[0])
    sample = val_loader.dataset.samples[0][0]

    served = _serve(Path(bundle)).post("/predict", files={"file": ("x.wav", Path(sample).read_bytes())}).json()
    probs = _softmax(_torch_scores(model, rt.Bundle(bundle).preprocess(sample)))
    _check_classification(served, predict_audio(bundle, sample), probs, classes)


def test_video_bundle_serves_like_model(tmp_path):
    from models.video_models import get_video_model
    from modalities.video import load_video_data

    data = tmp_path / "videos"
    for cls in ("a", "b"):
        for i in range(3):
            (data / cls).mkdir(parents=True, exist_ok=True)
            shutil.copy(VIDEO, data / cls / f"v{i}.mp4")
    _, val_loader, classes, prep = load_video_data(str(data), n_frames=4, frame_size=32, batch_size=1)
    model = get_video_model("TinyR3D", num_classes=len(classes), mode="from_scratch").eval()
    bundle, _ = export_bundle(model, prep, classes, "vid", output_dir=str(tmp_path / "out"),
                              sample_batch=next(iter(val_loader))[0])

    served = _serve(Path(bundle)).post("/predict", files={"file": ("x.mp4", VIDEO.read_bytes())}).json()
    probs = _softmax(_torch_scores(model, rt.Bundle(bundle).preprocess(str(VIDEO))))
    assert served["predictions"][0]["confidence"] == pytest.approx(float(probs.max()), abs=1e-4)


# ---------------------------------------------------------------------------
# Structured modalities
# ---------------------------------------------------------------------------

@pytest.fixture
def tabular_csv(tmp_path):
    rng = np.random.default_rng(0)
    n = 40
    df = pd.DataFrame({
        "age": rng.integers(18, 70, n).astype(float),
        "income": rng.normal(50_000, 10_000, n),
        "city": rng.choice(["oslo", "lima", "pune"], n),
        "label": rng.choice(["yes", "no", "maybe"], n),
        "price": rng.normal(200, 50, n),
    })
    path = tmp_path / "t.csv"
    df.to_csv(path, index=False)
    return path


RAW_ROW = {"age": 33, "income": 61000.0, "city": "lima"}


def test_tabular_classification_mlp(tmp_path, tabular_csv):
    from models.tabular_models import get_tabular_model
    from modalities.tabular import load_tabular_data
    from ui.inference_helpers import predict_tabular

    _, val_loader, classes, prep, n_in, *_ = load_tabular_data(
        str(tabular_csv), label_col="label", feature_cols=["age", "income", "city"], augmentation="none")
    model = get_tabular_model("MLP", num_classes=len(classes), input_size=n_in, dropout=0.0).eval()
    bundle, _ = export_bundle(model, prep, classes, "tabc", output_dir=str(tmp_path),
                              sample_batch=next(iter(val_loader))[0])

    served = _serve(Path(bundle)).post("/predict", json={"features": RAW_ROW}).json()
    probs = _softmax(_torch_scores(model, rt.Bundle(bundle).preprocess(RAW_ROW)))
    _check_classification(served, predict_tabular(bundle, RAW_ROW), probs, classes)


def test_tabular_regression_returns_a_value(tmp_path, tabular_csv):
    from models.tabular_models import get_tabular_model
    from modalities.tabular import load_tabular_data
    from ui.inference_helpers import predict_tabular

    _, val_loader, classes, prep, n_in, *_ = load_tabular_data(
        str(tabular_csv), label_col="price", feature_cols=["age", "income", "city"],
        augmentation="none", task="regression")
    model = get_tabular_model("MLP", num_classes=1, input_size=n_in, dropout=0.0).eval()
    bundle, _ = export_bundle(model, prep, classes, "tabr", output_dir=str(tmp_path),
                              sample_batch=next(iter(val_loader))[0])

    served = _serve(Path(bundle)).post("/predict", json={"features": RAW_ROW}).json()
    expected = float(_torch_scores(model, rt.Bundle(bundle).preprocess(RAW_ROW))[0])
    assert served == {"task": "regression", "value": pytest.approx(expected, abs=1e-4)}
    in_app = predict_tabular(bundle, RAW_ROW)
    assert in_app[0]["value"] == pytest.approx(expected, abs=1e-4)  # no softmax over one output


def test_tabular_sklearn_onnx_probabilities(tmp_path, tabular_csv):
    from sklearn.ensemble import RandomForestClassifier

    from modalities.tabular import load_tabular_data

    _, _, classes, prep, _, Xtr, ytr, *_ = load_tabular_data(
        str(tabular_csv), label_col="label", feature_cols=["age", "income", "city"], augmentation="none")
    model = RandomForestClassifier(n_estimators=10, random_state=0).fit(Xtr, ytr)
    bundle, warnings = export_bundle(model, prep, classes, "rf", output_dir=str(tmp_path), is_sklearn=True)
    assert (Path(bundle) / "model.onnx").is_file(), warnings

    served = _serve(Path(bundle)).post("/predict", json={"features": RAW_ROW})
    assert served.status_code == 200, served.text
    expected = model.predict_proba(rt.Bundle(bundle).preprocess(RAW_ROW))[0]
    top = served.json()["predictions"][0]
    assert top["label"] == classes[int(np.argmax(expected))]
    assert top["confidence"] == pytest.approx(float(expected.max()), abs=1e-4)


def test_timeseries_regression_with_lag_features(tmp_path):
    from models.timeseries_models import get_timeseries_model
    from modalities.timeseries import load_timeseries_data
    from ui.inference_helpers import predict_timeseries

    n = 80
    df = pd.DataFrame({"a": np.sin(np.arange(n) / 4), "b": np.cos(np.arange(n) / 5),
                       "y": np.arange(n, dtype=float)})
    csv = tmp_path / "s.csv"
    df.to_csv(csv, index=False)
    _, val_loader, classes, prep, n_in, *_ = load_timeseries_data(
        str(csv), label_col="y", feature_cols=["a", "b"], window_size=6, task="regression",
        augmentation="none", lag_steps=2, rolling_window=3)
    model = get_timeseries_model("LSTM", num_classes=1, input_size=n_in, hidden_size=8).eval()
    bundle, _ = export_bundle(model, prep, classes, "ts", output_dir=str(tmp_path / "out"),
                              sample_batch=next(iter(val_loader))[0])
    rows = df[["a", "b"]].tail(12).to_numpy().tolist()

    served = _serve(Path(bundle)).post("/predict", json={"rows": rows}).json()
    expected = float(_torch_scores(model, rt.Bundle(bundle).preprocess(rows))[0])
    assert served == {"task": "regression", "value": pytest.approx(expected, abs=1e-4)}
    assert predict_timeseries(bundle, rows)[0]["value"] == pytest.approx(expected, abs=1e-4)


def test_text_vocab_model(tmp_path):
    from models.text_models import get_text_model
    from modalities.text import load_text_data
    from ui.inference_helpers import predict_text

    texts = ["great course loved it", "terrible and confusing", "ok overall", "great pacing!"] * 5
    csv = tmp_path / "t.csv"
    pd.DataFrame({"text": texts, "label": ["pos", "neg", "neu", "pos"] * 5}).to_csv(csv, index=False)
    _, val_loader, classes, prep, vocab_size, _ = load_text_data(
        str(csv), text_col="text", label_col="label", model_name="LSTM", max_length=16)
    model = get_text_model("LSTM", num_classes=len(classes), mode="from_scratch",
                           vocab_size=vocab_size, hidden_size=16).eval()
    bundle, _ = export_bundle(model, prep, classes, "txt", output_dir=str(tmp_path / "out"),
                              sample_batch=next(iter(val_loader))[0])
    text = "Loved the great pacing!"

    served = _serve(Path(bundle)).post("/predict", json={"text": text}).json()
    probs = _softmax(_torch_scores(model, rt.Bundle(bundle).preprocess(text)))
    _check_classification(served, predict_text(bundle, text), probs, classes)


# ---------------------------------------------------------------------------
# CLI and server hardening
# ---------------------------------------------------------------------------

@pytest.fixture
def tabular_bundle(tmp_path, tabular_csv):
    from models.tabular_models import get_tabular_model
    from modalities.tabular import load_tabular_data

    _, val_loader, classes, prep, n_in, *_ = load_tabular_data(
        str(tabular_csv), label_col="label", feature_cols=["age", "income", "city"], augmentation="none")
    model = get_tabular_model("MLP", num_classes=len(classes), input_size=n_in, dropout=0.0).eval()
    bundle, _ = export_bundle(model, prep, classes, "cli", output_dir=str(tmp_path / "out"),
                              sample_batch=next(iter(val_loader))[0])
    return Path(bundle)


def test_inference_cli_runs_from_any_directory(tmp_path, tabular_bundle):
    proc = subprocess.run(
        [sys.executable, str(tabular_bundle / "inference.py"), json.dumps(RAW_ROW)],
        capture_output=True, text=True, cwd=tmp_path, timeout=120,
    )
    assert proc.returncode == 0, proc.stderr
    assert json.loads(proc.stdout) == rt.Bundle(tabular_bundle).predict(RAW_ROW)
    reqs = (tabular_bundle / "requirements.txt").read_text().split()
    assert any(r.startswith("onnxruntime==") for r in reqs) and any(r.startswith("numpy==") for r in reqs)


def test_server_requires_api_key_when_configured(tabular_bundle, monkeypatch):
    monkeypatch.setenv("NOCODE_DL_API_KEY", "s3cret")
    client = _serve(tabular_bundle)
    assert client.post("/predict", json={"features": RAW_ROW}).status_code == 401
    assert client.post("/predict", json={"features": RAW_ROW}, headers={"X-API-Key": "nope"}).status_code == 401
    assert client.post("/predict", json={"features": RAW_ROW}, headers={"X-API-Key": "s3cret"}).status_code == 200
    assert client.get("/health").status_code == 200


def test_server_rejects_bad_input_without_leaking_internals(tabular_bundle):
    client = _serve(tabular_bundle)
    resp = client.post("/predict", json={"features": ["not", "a", "dict"]})
    assert resp.status_code == 422
    assert "Traceback" not in resp.text


def test_upload_size_limit(tmp_path, monkeypatch):
    from models.image_models import get_image_model
    from modalities.image import load_image_data

    _, val_loader, classes, prep, _ = load_image_data(str(IMAGES), mode="from_scratch", image_size=32, augmentation="none")
    model = get_image_model("TinyCNN", num_classes=len(classes), mode="from_scratch").eval()
    bundle, _ = export_bundle(model, prep, classes, "lim", output_dir=str(tmp_path),
                              sample_batch=next(iter(val_loader))[0])
    monkeypatch.setenv("NOCODE_DL_MAX_UPLOAD_MB", "0.001")  # ~1 KB
    resp = _serve(Path(bundle)).post("/predict", files={"file": ("big.png", b"0" * 5000)})
    assert resp.status_code == 413

"""The bundle runtime must build exactly the inputs the training pipeline builds.

Each test runs a sample through the real training data path (validation
transforms, no augmentation) and through export/bundle_runtime.py, and
requires the arrays to match. If training preprocessing changes without the
runtime following, exported models silently get different inputs.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image

from export import bundle_runtime as rt

ROOT = Path(__file__).resolve().parents[1]
IMAGE = ROOT / "fixtures" / "image_stress_dataset" / "red_square" / "red_square_00.png"
AUDIO = ROOT / "fixtures" / "audio_stress_dataset" / "low_tone" / "low_tone_00.wav"


@pytest.mark.parametrize("force_grayscale", [False, True])
def test_image_matches_training_validation_transform(force_grayscale):
    from data_pipeline.augmentation import get_image_transforms

    norm = {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]}
    _, val_tf = get_image_transforms("none", image_size=64, normalization=norm, force_grayscale=force_grayscale)
    expected = val_tf(Image.open(IMAGE).convert("RGB")).unsqueeze(0).numpy()
    prep = {"resize": [64, 64], **norm, "force_grayscale": force_grayscale}
    got = rt.image_to_model_input(str(IMAGE), prep)
    np.testing.assert_allclose(got, expected, atol=1e-5)


@pytest.mark.parametrize("normalize_waveform", [False, True])
def test_audio_matches_training_dataset(normalize_waveform):
    from torchvision import transforms

    from modalities.audio import AudioDataset

    size = 96
    tf = transforms.Compose([
        transforms.Resize((size, size), antialias=True),
        transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3),
    ])
    ds = AudioDataset([(str(AUDIO), 0)], sample_rate=16000, n_mels=64, n_fft=1024, hop_length=512,
                      max_duration_sec=2, transform=tf, normalize_waveform=normalize_waveform)
    expected = ds[0][0].unsqueeze(0).numpy()
    prep = {"sample_rate": 16000, "n_mels": 64, "n_fft": 1024, "hop_length": 512,
            "max_duration_sec": 2, "resize": [size, size], "mean": [0.5] * 3, "std": [0.5] * 3,
            "normalize_waveform": normalize_waveform}
    got = rt.audio_to_model_input(str(AUDIO), prep)
    np.testing.assert_allclose(got, expected, atol=1e-5)
    assert -1.01 <= got.min() and got.max() <= 1.01  # scaled, not raw dB


@pytest.mark.parametrize("scaling", ["standard", "minmax", "robust", "none"])
def test_tabular_matches_training_rows(tmp_path, scaling):
    from modalities.tabular import load_tabular_data

    rng = np.random.default_rng(0)
    n = 30
    df = pd.DataFrame({
        "age": rng.integers(18, 70, n).astype(float),
        "income": rng.normal(50_000, 10_000, n),
        "city": rng.choice(["oslo", "lima", "pune"], n),
        "label": rng.choice(["yes", "no"], n),
    })
    df.loc[3, "income"] = np.nan  # training fills numeric NaN with 0
    path = tmp_path / "t.csv"
    df.to_csv(path, index=False)
    _, _, _, prep, _, X_train, _, X_val, _ = load_tabular_data(
        str(path), label_col="label", augmentation="none", scaling_strategy=scaling, val_split=0.2)

    from data_pipeline.splits import stratified_split

    classes = sorted(set(df["label"].astype(str)))
    train_idx, val_idx = stratified_split([classes.index(str(v)) for v in df["label"]], 0.2)
    X_by_row = dict(zip(train_idx + val_idx, np.concatenate([X_train, X_val])))
    for i in range(n):
        raw = df.drop(columns=["label"]).iloc[i].to_dict()
        got = rt.tabular_to_model_input(raw, prep)[0]
        np.testing.assert_allclose(got, X_by_row[i], atol=1e-5)


def test_timeseries_matches_training_windows(tmp_path):
    from modalities.timeseries import prepare_timeseries_windows

    rng = np.random.default_rng(1)
    n = 60
    df = pd.DataFrame({"a": np.sin(np.arange(n) / 3) + rng.normal(0, 0.1, n),
                       "b": np.arange(n, dtype=float), "label": (np.arange(n) // 10) % 2})
    path = tmp_path / "s.csv"
    df.to_csv(path, index=False)
    window, lag, roll = 5, 2, 3
    windows, _, _, prep, _, starts = prepare_timeseries_windows(
        str(path), label_col="label", feature_cols=["a", "b"], window_size=window,
        lag_steps=lag, rolling_window=roll, return_window_starts=True)
    trim = max(lag, roll - 1)
    raw = df[["a", "b"]].to_numpy().tolist()
    for i in (0, len(starts) // 2, len(starts) - 1):
        history = raw[: trim + int(starts[i]) + window]  # everything up to the window's last row
        got = rt.timeseries_to_model_input(history, prep)[0]
        np.testing.assert_allclose(got, windows[i], atol=1e-5)


def test_text_vocab_encoding_matches_training(tmp_path):
    from modalities.text import load_text_data

    texts = ["Great course, LOVED it!", "terrible... and confusing", "ok overall", "great pacing"] * 5
    path = tmp_path / "t.csv"
    pd.DataFrame({"text": texts, "label": ["pos", "neg", "neu", "pos"] * 5}).to_csv(path, index=False)
    _, val_loader, _, prep, _, val_texts = load_text_data(
        str(path), text_col="text", label_col="label", model_name="LSTM", max_length=12, val_split=0.25)
    enc = val_loader.dataset.encodings
    for i, text in enumerate(val_texts):
        got = rt.text_to_model_inputs(text, prep)
        assert got["input_ids"][0].tolist() == enc["input_ids"][i].tolist()
        assert got["attention_mask"][0].tolist() == enc["attention_mask"][i].tolist()


def test_text_cleaning_matches_training_cleaning_step():
    from data_pipeline.cleaning import clean_text_dataframe

    texts = ["Visit https://x.io NOW!!", "The children were running quickly", "  geese   and mice  "]
    opts = dict(lowercase=True, strip_urls=True, strip_punctuation=True, remove_stopwords=True,
                apply_stemming=True, apply_lemmatization=True)
    cleaned, _ = clean_text_dataframe(pd.DataFrame({"t": texts}), "t", drop_empty_rows=False, **opts)
    for raw, expected in zip(texts, cleaned["t"]):
        assert rt.clean_text(raw, **opts) == expected

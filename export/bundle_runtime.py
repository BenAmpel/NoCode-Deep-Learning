"""
NoCode-DL bundle runtime: load an exported model bundle and predict from raw input.

This one file is the only implementation of inference preprocessing. The app's
"Try Your Model" tab uses it, and the exporter copies it into every bundle as
``bundle_runtime.py`` for the generated ``main.py`` (FastAPI) and
``inference.py`` (CLI). Keeping one implementation is what stops served
predictions from drifting away from what the model saw in training; parity
tests compare it against the training pipelines.

It must stay importable on its own: no imports from the rest of the app.

Dependencies: numpy and onnxruntime always. Per modality, only when used:
Pillow (image, audio), torch + torchaudio + soundfile + scipy (audio), transformers
(pretrained text), opencv-python-headless (video), joblib + scikit-learn
(sklearn bundles saved without ONNX).
"""
from __future__ import annotations

import json
import re
import shutil
import subprocess
import tempfile
import warnings
from pathlib import Path
from typing import Any

import numpy as np

RUNTIME_VERSION = 1

_DEFAULT_IMAGE_SIZE = 224
_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD = [0.229, 0.224, 0.225]


class BundleInputError(ValueError):
    """The input could not be turned into a model input (bad shape, file, JSON...)."""


class UntrustedModelError(PermissionError):
    """The bundle's model is a pickle (model.joblib) and the bundle isn't trusted."""


# ---------------------------------------------------------------------------
# Text cleaning (also used by data_pipeline.cleaning during training)
# ---------------------------------------------------------------------------

BASIC_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "been", "being", "but", "by",
    "for", "from", "had", "has", "have", "he", "her", "hers", "him", "his",
    "i", "if", "in", "into", "is", "it", "its", "me", "my", "of", "on", "or",
    "our", "ours", "she", "so", "than", "that", "the", "their", "theirs",
    "them", "they", "this", "those", "to", "too", "us", "was", "we", "were",
    "what", "when", "where", "which", "who", "why", "with", "you", "your",
    "yours",
}


def stem_token(token: str) -> str:
    token = token.strip()
    for suffix in ("ingly", "edly", "ing", "ed", "ly", "s"):
        if len(token) > len(suffix) + 2 and token.endswith(suffix):
            return token[: -len(suffix)]
    return token


def lemmatize_token(token: str) -> str:
    token = token.strip()
    irregular = {"children": "child", "mice": "mouse", "geese": "goose", "better": "good"}
    if token in irregular:
        return irregular[token]
    for suffix, replacement in (("ies", "y"), ("ves", "f"), ("men", "man")):
        if len(token) > len(suffix) + 1 and token.endswith(suffix):
            return token[: -len(suffix)] + replacement
    return token


def clean_text(
    value: str,
    *,
    lowercase: bool = False,
    strip_urls: bool = False,
    strip_punctuation: bool = False,
    remove_stopwords: bool = False,
    apply_stemming: bool = False,
    apply_lemmatization: bool = False,
) -> str:
    """Normalize one text the way training's text-cleaning step does."""
    value = str(value)
    if strip_urls:
        value = re.sub(r"https?://\S+|www\.\S+", " ", value)
    if lowercase:
        value = value.lower()
    if strip_punctuation:
        value = re.sub(r"[^\w\s]", " ", value)
    if remove_stopwords:
        value = " ".join(t for t in value.split() if t not in BASIC_STOPWORDS)
    if apply_lemmatization:
        value = " ".join(lemmatize_token(t) for t in value.split())
    if apply_stemming:
        value = " ".join(stem_token(t) for t in value.split())
    return re.sub(r"\s+", " ", value).strip()


def _clean_text_like_training(text: str, cleaning: dict) -> str:
    # Training only runs the cleaning step when at least one option is set.
    if not any(bool(v) for v in (cleaning or {}).values()):
        return str(text)
    return clean_text(
        text,
        lowercase=bool(cleaning.get("lowercase")),
        strip_urls=bool(cleaning.get("strip_urls")),
        strip_punctuation=bool(cleaning.get("strip_punctuation")),
        remove_stopwords=bool(cleaning.get("remove_stopwords")),
        apply_stemming=bool(cleaning.get("apply_stemming")),
        apply_lemmatization=bool(cleaning.get("apply_lemmatization")),
    )


def simple_tokenize(text: str) -> list[str]:
    """Tokenizer for from-scratch text models (LSTM/GRU/CNN/Transformer)."""
    return re.findall(r"\w+|[^\w\s]", text.lower())


# ---------------------------------------------------------------------------
# Time-series features (also used by modalities.timeseries during training)
# ---------------------------------------------------------------------------

def build_temporal_feature_matrix(
    values: np.ndarray,
    *,
    lag_steps: int = 0,
    rolling_window: int = 0,
) -> tuple[np.ndarray, int]:
    """Expand base time-series features with lag and rolling statistics.

    Returns the engineered feature matrix and the number of leading rows that
    must be discarded to avoid NaNs caused by lagging/rolling windows.
    """
    engineered = [values]
    trim = 0

    if int(lag_steps) > 0:
        trim = max(trim, int(lag_steps))
        for lag in range(1, int(lag_steps) + 1):
            lagged = np.roll(values, shift=lag, axis=0)
            lagged[:lag, :] = np.nan
            engineered.append(lagged)

    if int(rolling_window) > 1:
        window = int(rolling_window)
        trim = max(trim, window - 1)
        means = np.full_like(values, np.nan, dtype=np.float32)
        stds = np.full_like(values, np.nan, dtype=np.float32)
        mins = np.full_like(values, np.nan, dtype=np.float32)
        maxs = np.full_like(values, np.nan, dtype=np.float32)
        for idx in range(window - 1, len(values)):
            chunk = values[idx - window + 1: idx + 1]
            means[idx] = chunk.mean(axis=0)
            stds[idx] = chunk.std(axis=0)
            mins[idx] = chunk.min(axis=0)
            maxs[idx] = chunk.max(axis=0)
        engineered.extend([means, stds, mins, maxs])

    matrix = np.concatenate(engineered, axis=1).astype(np.float32)
    return matrix, trim


# ---------------------------------------------------------------------------
# Audio decoding and spectrograms (also used by modalities.audio)
# ---------------------------------------------------------------------------

def _read_wav_with_scipy(path: str):
    import torch
    from scipy.io import wavfile

    sample_rate, data = wavfile.read(path)
    arr = np.asarray(data)
    arr = arr[np.newaxis, :] if arr.ndim == 1 else arr.T
    if np.issubdtype(arr.dtype, np.integer):
        if arr.dtype == np.uint8:
            waveform = (arr.astype(np.float32) - 128.0) / 128.0
        else:
            waveform = arr.astype(np.float32) / max(float(np.iinfo(arr.dtype).max), 1.0)
    else:
        waveform = arr.astype(np.float32)
    return torch.from_numpy(waveform), int(sample_rate)


def _decode_audio_with_ffmpeg(path: str):
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("ffmpeg is not installed")
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        subprocess.run(
            [ffmpeg, "-y", "-loglevel", "error", "-i", path, tmp_path],
            check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True,
        )
        return _read_wav_with_scipy(tmp_path)
    except subprocess.CalledProcessError as exc:
        detail = exc.stderr.strip() or "ffmpeg could not decode the file"
        raise RuntimeError(f"ffmpeg decode failed: {detail}") from exc
    finally:
        Path(tmp_path).unlink(missing_ok=True)


# soundfile's bundled libsndfile decodes these without ffmpeg. Anything else
# (.m4a / AAC) needs the ffmpeg command-line tool.
SOUNDFILE_EXTS = {".wav", ".flac", ".mp3", ".ogg"}


def needs_ffmpeg(path: str) -> bool:
    """True if decoding ``path`` depends on ffmpeg and ffmpeg isn't installed."""
    return Path(path).suffix.lower() not in SOUNDFILE_EXTS and shutil.which("ffmpeg") is None


def _read_with_soundfile(path: str):
    import soundfile
    import torch

    data, sample_rate = soundfile.read(path, dtype="float32", always_2d=True)
    return torch.from_numpy(np.ascontiguousarray(data.T)), int(sample_rate)


def _read_with_torchaudio(path: str):
    import torchaudio

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return torchaudio.load(path)


def load_audio_waveform(path: str, target_sr: int | None = None):
    """Decode an audio file to a (channels, samples) float tensor, resampled if asked.

    Tries soundfile (wav, flac, mp3, ogg), then torchaudio (works only when
    torchcodec + FFmpeg libraries are installed), then scipy for wav and the
    ffmpeg command-line tool for everything else.
    """
    import torchaudio

    suffix = Path(path).suffix.lower()
    decoders = [("soundfile", _read_with_soundfile), ("torchaudio", _read_with_torchaudio)]
    decoders.append(("scipy", _read_wav_with_scipy) if suffix == ".wav"
                    else ("ffmpeg", _decode_audio_with_ffmpeg))

    waveform = sample_rate = None
    errors: list[str] = []
    for name, decode in decoders:
        try:
            waveform, sample_rate = decode(path)
            break
        except Exception as exc:
            errors.append(f"{name}: {exc}")

    if waveform is None or sample_rate is None:
        hint = ""
        if needs_ffmpeg(path):
            hint = (f" {suffix or 'This'} audio needs ffmpeg, which isn't installed. Install ffmpeg, "
                    "or convert the file to WAV, MP3, FLAC or OGG.")
        raise RuntimeError(f"Could not decode audio file {path} ({'; '.join(errors)}).{hint}")

    if target_sr is not None and sample_rate != target_sr:
        waveform = torchaudio.functional.resample(waveform, sample_rate, target_sr)
        sample_rate = target_sr
    return waveform, int(sample_rate)


def scale_spectrogram(mel_db):
    """Min-max scale a dB spectrogram to [0, 1] (raw dB spans roughly -100..+40)."""
    lo, hi = mel_db.min(), mel_db.max()
    if hi > lo:
        return (mel_db - lo) / (hi - lo)
    return mel_db * 0.0


def audio_to_model_input(path: str, prep: dict) -> np.ndarray:
    """Audio file -> (1, 3, H, W) float32, matching the training validation pipeline."""
    import torch
    import torchaudio.transforms as AT
    from torchvision.transforms import functional as TF

    sample_rate = int(prep.get("sample_rate", 22050))
    max_samples = int(prep.get("max_duration_sec", 10)) * sample_rate
    h, w = _resize_hw(prep)

    waveform, sr = load_audio_waveform(path, target_sr=sample_rate)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(0, keepdim=True)
    if prep.get("normalize_waveform"):
        peak = waveform.abs().max()
        if peak > 0:
            waveform = waveform / peak
    n = waveform.shape[1]
    if n > max_samples:
        waveform = waveform[:, :max_samples]
    elif n < max_samples:
        waveform = torch.nn.functional.pad(waveform, (0, max_samples - n))

    mel = AT.MelSpectrogram(
        sample_rate=sr,
        n_mels=int(prep.get("n_mels", 64)),
        n_fft=int(prep.get("n_fft", 1024)),
        hop_length=int(prep.get("hop_length", 512)),
    )(waveform)
    img = scale_spectrogram(AT.AmplitudeToDB()(mel)).repeat(3, 1, 1)
    img = TF.resize(img, [h, w], antialias=True)
    img = TF.normalize(img, mean=prep.get("mean", [0.5] * 3), std=prep.get("std", [0.5] * 3))
    return img.unsqueeze(0).numpy().astype(np.float32)


# ---------------------------------------------------------------------------
# Per-modality preprocessing
# ---------------------------------------------------------------------------

def _resize_hw(prep: dict) -> tuple[int, int]:
    resize = prep.get("resize", [_DEFAULT_IMAGE_SIZE, _DEFAULT_IMAGE_SIZE])
    if isinstance(resize, (list, tuple)) and len(resize) >= 2:
        return int(resize[0]), int(resize[1])
    return _DEFAULT_IMAGE_SIZE, _DEFAULT_IMAGE_SIZE


def _open_image(image: Any):
    from PIL import Image

    if isinstance(image, Image.Image):
        return image.convert("RGB")
    if isinstance(image, (bytes, bytearray)):
        import io
        return Image.open(io.BytesIO(image)).convert("RGB")
    if isinstance(image, (str, Path)):
        return Image.open(image).convert("RGB")
    arr = np.asarray(image)
    if arr.dtype != np.uint8:
        arr = arr.astype(np.float32)
        if arr.max() <= 1.0:
            arr = arr * 255.0
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    return Image.fromarray(arr).convert("RGB")


def image_to_model_input(image: Any, prep: dict) -> np.ndarray:
    """Image -> (1, 3, H, W) float32, matching torchvision's validation transform:
    Resize (PIL bilinear) -> optional Grayscale(3) -> ToTensor -> Normalize."""
    from PIL import Image

    h, w = _resize_hw(prep)
    img = _open_image(image).resize((w, h), Image.BILINEAR)
    if prep.get("force_grayscale"):
        img = img.convert("L").convert("RGB")
    arr = np.asarray(img, dtype=np.float32) / 255.0
    mean = np.asarray(prep.get("mean", _IMAGENET_MEAN), dtype=np.float32)
    std = np.asarray(prep.get("std", _IMAGENET_STD), dtype=np.float32)
    arr = (arr - mean) / std
    return arr.transpose(2, 0, 1)[np.newaxis].astype(np.float32)


def video_to_model_input(path: str, prep: dict) -> np.ndarray:
    """Video file -> (1, 3, T, H, W) float32, sampling frames like training."""
    import cv2

    n_frames = int(prep.get("n_frames", 8))
    size = int(prep.get("frame_size", 112))
    cap = cv2.VideoCapture(str(path))
    total = max(1, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
    if total <= n_frames:
        idxs = list(range(total)) + [total - 1] * (n_frames - total)
    else:
        idxs = [int(i * total / n_frames) for i in range(n_frames)]
    frames = []
    for fi in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ok, frame = cap.read()
        if not ok:
            frame = np.zeros((size, size, 3), dtype=np.uint8)
        frame = cv2.resize(frame, (size, size))
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    arr = np.stack(frames).transpose(3, 0, 1, 2).astype(np.float32) / 255.0
    mean = np.asarray(prep.get("mean"), dtype=np.float32).reshape(3, 1, 1, 1)
    std = np.asarray(prep.get("std"), dtype=np.float32).reshape(3, 1, 1, 1)
    return ((arr - mean) / std)[np.newaxis].astype(np.float32)


def tabular_to_model_input(features: dict[str, Any], prep: dict) -> np.ndarray:
    """Raw column values -> (1, n_features) float32: scale numerics, one-hot categoricals.

    Missing or non-numeric numeric values become 0.0, as training's fillna(0)
    does; an unseen category leaves all of that column's one-hot slots at 0.
    """
    if not isinstance(features, dict):
        raise BundleInputError("Tabular input must be an object mapping column names to values.")
    numeric_cols = prep.get("numeric_cols", [])
    values = []
    for col in numeric_cols:
        try:
            v = float(features.get(col, 0.0))
            values.append(0.0 if np.isnan(v) else v)
        except (TypeError, ValueError):
            values.append(0.0)
    num = np.asarray(values, dtype=np.float32).reshape(1, -1)

    if num.shape[1]:
        strategy = prep.get("scaling_strategy", "standard")
        if strategy == "minmax":
            lo = np.asarray(prep["min_vals"], dtype=np.float32)
            hi = np.asarray(prep["max_vals"], dtype=np.float32)
            num = (num - lo) / np.where((hi - lo) == 0, 1.0, hi - lo)
        elif strategy == "robust":
            med = np.asarray(prep["median_vals"], dtype=np.float32)
            iqr = np.asarray(prep["iqr"], dtype=np.float32)
            num = (num - med) / np.where(iqr == 0, 1.0, iqr)
        elif strategy != "none":
            mean = np.asarray(prep["scaler_mean"], dtype=np.float32)
            scale = np.asarray(prep["scaler_scale"], dtype=np.float32)
            num = (num - mean) / np.where(scale == 0, 1.0, scale)

    dummy_names = prep.get("dummy_feature_names", [])
    category_maps = prep.get("category_maps", {})
    dummies = {name: 0.0 for name in dummy_names}
    for col in prep.get("categorical_cols", []):
        name = category_maps.get(col, {}).get(str(features.get(col)))
        if name in dummies:
            dummies[name] = 1.0
    cat = np.asarray([dummies[n] for n in dummy_names], dtype=np.float32).reshape(1, -1)
    return np.concatenate([num, cat], axis=1).astype(np.float32)


def timeseries_to_model_input(rows: Any, prep: dict) -> np.ndarray:
    """Recent rows (oldest first) -> (1, window_size, n_features) float32.

    Rows are lists of raw feature values or objects keyed by feature name.
    Short inputs are padded by repeating the first row, as before.
    """
    if isinstance(rows, str):
        try:
            rows = json.loads(rows)
        except json.JSONDecodeError as exc:
            raise BundleInputError(f"Time-series input is not valid JSON: {exc}") from exc
    if not isinstance(rows, list) or not rows:
        raise BundleInputError("Time-series input must be a non-empty list of rows.")

    feature_cols = prep.get("raw_feature_columns") or prep.get("feature_columns", [])
    window_size = int(prep.get("window_size", 1))
    lag_steps = int(prep.get("lag_steps", 0))
    rolling_window = int(prep.get("rolling_window", 0))

    if isinstance(rows[0], dict):
        cols = feature_cols or list(rows[0].keys())
        arr = np.asarray([[float(r.get(c, 0.0)) for c in cols] for r in rows], dtype=np.float32)
    else:
        arr = np.asarray([[float(v) for v in r] for r in rows], dtype=np.float32)
    if arr.ndim != 2:
        raise BundleInputError("Time-series input must be 2-D: [time, features].")
    if feature_cols and arr.shape[1] != len(feature_cols):
        raise BundleInputError(f"Expected {len(feature_cols)} features per row, got {arr.shape[1]}.")

    min_rows = window_size + max(lag_steps, max(0, rolling_window - 1))
    if arr.shape[0] < min_rows:
        arr = np.concatenate([np.repeat(arr[:1], min_rows - arr.shape[0], axis=0), arr], axis=0)
    if lag_steps > 0 or rolling_window > 1:
        arr, trim = build_temporal_feature_matrix(arr, lag_steps=lag_steps, rolling_window=rolling_window)
        arr = arr[trim:]
    arr = arr[-window_size:]

    mean = np.asarray(prep.get("mean", []), dtype=np.float32)
    std = np.asarray(prep.get("std", []), dtype=np.float32)
    if len(mean) == arr.shape[1] and len(std) == arr.shape[1]:
        arr = (arr - mean) / np.where(std == 0, 1.0, std)
    return arr[np.newaxis].astype(np.float32)


def text_to_model_inputs(text: str, prep: dict, tokenizer=None) -> dict[str, np.ndarray]:
    """Text -> {"input_ids", "attention_mask"} int64 arrays of shape (1, max_length).

    From-scratch models use the vocabulary saved in ``prep``; pretrained
    transformers need the Hugging Face ``tokenizer`` they were trained with.
    """
    text = _clean_text_like_training(str(text), prep.get("cleaning", {}))
    max_length = int(prep.get("max_length", 128))
    vocab = prep.get("vocab")
    if vocab:
        ids = [vocab.get(t, vocab.get("<unk>", 1)) for t in simple_tokenize(text)[:max_length]]
        mask = [1] * len(ids) + [0] * (max_length - len(ids))
        ids = ids + [0] * (max_length - len(ids))
    else:
        if tokenizer is None:
            raise BundleInputError("This text model needs its tokenizer to preprocess input.")
        enc = tokenizer(text, max_length=max_length, padding="max_length",
                        truncation=True, return_tensors="np")
        ids, mask = enc["input_ids"][0].tolist(), enc["attention_mask"][0].tolist()
    return {
        "input_ids": np.asarray([ids], dtype=np.int64),
        "attention_mask": np.asarray([mask], dtype=np.int64),
    }


# ---------------------------------------------------------------------------
# Bundle
# ---------------------------------------------------------------------------

def _softmax(x: np.ndarray) -> np.ndarray:
    z = x - x.max()
    e = np.exp(z)
    return e / e.sum()


class Bundle:
    """A model bundle folder: labels.json, preprocessing.json and model.onnx/.joblib."""

    def __init__(self, path: str | Path, *, trust_pickle: bool = False):
        """``trust_pickle``: allow loading ``model.joblib`` (a pickle, which can
        run arbitrary code when loaded). Only set it for bundles you created;
        ONNX models never need it."""
        self.path = Path(path)
        self.trust_pickle = bool(trust_pickle)
        with open(self.path / "labels.json", encoding="utf-8") as fh:
            self.labels: dict[str, str] = {str(k): str(v) for k, v in json.load(fh).items()}
        with open(self.path / "preprocessing.json", encoding="utf-8") as fh:
            self.prep: dict = json.load(fh)
        self.modality: str = self.prep.get("modality", "")
        self.onnx_path = self.path / "model.onnx" if (self.path / "model.onnx").is_file() else None
        joblib_path = next((self.path / n for n in ("model.joblib", "model.pkl") if (self.path / n).is_file()), None)
        self.joblib_path = joblib_path
        if self.onnx_path is None and self.joblib_path is None:
            raise FileNotFoundError(f"No model.onnx or model.joblib in {self.path}")
        self._session = None
        self._sk_model = None
        self._tokenizer = None

    # -- metadata ---------------------------------------------------------

    @property
    def task(self) -> str:
        task = str(self.prep.get("task", "classification"))
        if task in ("regression", "forecasting"):
            return "regression"
        if len(self.labels) == 1 and list(self.labels.values()) == ["target"]:
            return "regression"
        return "classification"

    @property
    def classes(self) -> list[str]:
        return [self.labels[k] for k in sorted(self.labels, key=lambda k: int(k))]

    # -- model execution --------------------------------------------------

    def _onnx(self):
        if self._session is None:
            import onnxruntime as ort
            self._session = ort.InferenceSession(str(self.onnx_path), providers=["CPUExecutionProvider"])
        return self._session

    def _run(self, inputs: dict[str, np.ndarray] | np.ndarray) -> np.ndarray:
        """Run the model and return 1-D outputs: class scores or a regression value."""
        if self.onnx_path is not None:
            sess = self._onnx()
            declared = sess.get_inputs()
            if isinstance(inputs, dict):
                feeds = {i.name: inputs[i.name] for i in declared if i.name in inputs}
                if len(feeds) != len(declared):
                    feeds = {i.name: v for i, v in zip(declared, inputs.values())}
            else:
                x = inputs
                if len(declared[0].shape) == 2 and x.ndim > 2:
                    x = x.reshape(x.shape[0], -1)  # sklearn models take flattened windows
                feeds = {declared[0].name: x}
            outputs = sess.run(None, feeds)
            names = [o.name for o in sess.get_outputs()]
            if "probabilities" in names:  # skl2onnx classifier (zipmap=False)
                return np.asarray(outputs[names.index("probabilities")], dtype=np.float32).reshape(-1)
            if "variable" in names:  # skl2onnx regressor
                return np.asarray(outputs[names.index("variable")], dtype=np.float32).reshape(-1)
            return np.asarray(outputs[0], dtype=np.float32).reshape(-1)

        if self._sk_model is None:
            if not self.trust_pickle:
                raise UntrustedModelError(
                    f"{self.joblib_path.name} is a pickle file, and loading a pickle can run any "
                    "code it contains. Only open bundles you created yourself; this one is not "
                    "marked as trusted."
                )
            import joblib
            self._sk_model = joblib.load(self.joblib_path)
        x = inputs if not isinstance(inputs, dict) else next(iter(inputs.values()))
        x = x.reshape(x.shape[0], -1)
        if self.task == "classification" and hasattr(self._sk_model, "predict_proba"):
            return np.asarray(self._sk_model.predict_proba(x), dtype=np.float32).reshape(-1)
        return np.asarray(self._sk_model.predict(x), dtype=np.float32).reshape(-1)

    def _text_inputs(self, text: str) -> dict[str, np.ndarray]:
        if not self.prep.get("vocab") and self._tokenizer is None:
            from transformers import AutoTokenizer
            local = self.path / "tokenizer"
            source = str(local) if local.is_dir() else self.prep.get("tokenizer", "distilbert-base-uncased")
            self._tokenizer = AutoTokenizer.from_pretrained(source)
        return text_to_model_inputs(text, self.prep, tokenizer=self._tokenizer)

    # -- public API -------------------------------------------------------

    def preprocess(self, raw: Any):
        m = self.modality
        try:
            if m == "image":
                return image_to_model_input(raw, self.prep)
            if m == "audio":
                return audio_to_model_input(str(raw), self.prep)
            if m == "video":
                return video_to_model_input(str(raw), self.prep)
            if m == "tabular":
                return tabular_to_model_input(raw, self.prep)
            if m == "timeseries":
                return timeseries_to_model_input(raw, self.prep)
            if m == "text":
                return self._text_inputs(raw)
        except BundleInputError:
            raise
        except (OSError, ValueError, KeyError, TypeError) as exc:
            raise BundleInputError(f"Could not read {m} input: {exc}") from exc
        raise BundleInputError(f"Inference is not supported for '{m}' bundles.")

    def predict(self, raw: Any, top_k: int = 3) -> dict:
        """Predict from raw input.

        Classification: {"task": "classification", "predictions": [{"label",
        "confidence", "rank"}, ...]} sorted by confidence.
        Regression: {"task": "regression", "value": float}.
        """
        out = self._run(self.preprocess(raw))
        if self.task == "regression":
            return {"task": "regression", "value": float(out[0])}

        scores = out[: len(self.labels)] if len(out) >= len(self.labels) else out
        looks_like_probs = scores.min() >= 0.0 and scores.max() <= 1.0 and abs(scores.sum() - 1.0) < 1e-3
        probs = scores if looks_like_probs else _softmax(scores)
        # Stable sort on -probs: ties go to the lower class index, like argmax
        # (and training's metrics). argsort(...)[::-1] picked the higher one.
        order = np.argsort(-probs, kind="stable")[: max(1, min(int(top_k), len(probs)))]
        return {
            "task": "classification",
            "predictions": [
                {"label": self.labels.get(str(int(i)), f"class_{int(i)}"),
                 "confidence": float(probs[i]), "rank": rank}
                for rank, i in enumerate(order, start=1)
            ],
        }

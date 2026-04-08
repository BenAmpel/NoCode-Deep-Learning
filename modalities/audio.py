"""
Audio modality pipeline.

Strategy: audio → mono waveform → mel spectrogram → 3-channel "image"
This lets us reuse the same image model zoo with no extra model code.

Expected data format:
    data_path/
        class_a/
            clip1.wav
            ...
        class_b/
            ...
"""
import random
import shutil
import subprocess
import tempfile
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from config import DEFAULTS

AUDIO_EXTS = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}


def _read_wav_with_scipy(path: str) -> tuple[torch.Tensor, int]:
    from scipy.io import wavfile

    sample_rate, data = wavfile.read(path)
    arr = np.asarray(data)

    if arr.ndim == 1:
        arr = arr[np.newaxis, :]
    else:
        arr = arr.T

    if np.issubdtype(arr.dtype, np.integer):
        if arr.dtype == np.uint8:
            waveform = (arr.astype(np.float32) - 128.0) / 128.0
        else:
            scale = float(np.iinfo(arr.dtype).max)
            waveform = arr.astype(np.float32) / max(scale, 1.0)
    else:
        waveform = arr.astype(np.float32)

    return torch.from_numpy(waveform), int(sample_rate)


def _decode_audio_with_ffmpeg(path: str) -> tuple[torch.Tensor, int]:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("ffmpeg is not installed")

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        subprocess.run(
            [
                ffmpeg,
                "-y",
                "-loglevel",
                "error",
                "-i",
                path,
                tmp_path,
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
        )
        return _read_wav_with_scipy(tmp_path)
    except subprocess.CalledProcessError as exc:
        detail = exc.stderr.strip() or "ffmpeg could not decode the file"
        raise RuntimeError(f"ffmpeg decode failed: {detail}") from exc
    finally:
        try:
            Path(tmp_path).unlink(missing_ok=True)
        except Exception:
            pass


def load_audio_waveform(path: str, target_sr: int | None = None) -> tuple[torch.Tensor, int]:
    import torchaudio

    waveform = None
    sample_rate = None
    load_errors: list[str] = []
    suffix = Path(path).suffix.lower()

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            waveform, sample_rate = torchaudio.load(path)
    except Exception as exc:
        load_errors.append(f"torchaudio.load failed: {exc}")

    if waveform is None or sample_rate is None:
        if suffix == ".wav":
            try:
                waveform, sample_rate = _read_wav_with_scipy(path)
            except Exception as exc:
                load_errors.append(f"scipy wavfile.read failed: {exc}")
        else:
            try:
                waveform, sample_rate = _decode_audio_with_ffmpeg(path)
            except Exception as exc:
                load_errors.append(str(exc))

    if waveform is None or sample_rate is None:
        hint = ""
        if suffix in {".mp3", ".m4a", ".ogg", ".flac"} and not shutil.which("ffmpeg"):
            hint = " Install ffmpeg to enable fallback decoding for compressed audio."
        raise RuntimeError(
            ("; ".join(load_errors) or f"Unable to decode audio file: {path}") + hint
        )

    if target_sr is not None and sample_rate != target_sr:
        waveform = torchaudio.functional.resample(waveform, sample_rate, target_sr)
        sample_rate = target_sr

    return waveform, int(sample_rate)


class AudioDataset(Dataset):
    def __init__(
        self,
        samples: list[tuple[str, int]],
        sample_rate: int = 22050,
        n_mels: int = 64,
        n_fft: int = 1024,
        hop_length: int = 512,
        max_duration_sec: int = 10,
        transform=None,
        waveform_transform=None,
        spectrogram_transform=None,
        normalize_waveform: bool = False,
    ):
        import torchaudio.transforms as AT
        self.samples         = samples
        self.target_sr       = sample_rate
        self.max_samples     = max_duration_sec * sample_rate
        self.mel_transform     = AT.MelSpectrogram(
            sample_rate=sample_rate,
            n_mels=n_mels,
            n_fft=n_fft,
            hop_length=hop_length,
        )
        self.to_db             = AT.AmplitudeToDB()
        self.transform         = transform
        self.waveform_transform = waveform_transform  # applied before mel conversion
        self.spectrogram_transform = spectrogram_transform
        self.normalize_waveform = normalize_waveform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        waveform, sr = load_audio_waveform(path, target_sr=self.target_sr)

        # Force mono
        if waveform.shape[0] > 1:
            waveform = waveform.mean(0, keepdim=True)

        if self.normalize_waveform:
            peak = waveform.abs().max()
            if peak > 0:
                waveform = waveform / peak

        # Pad / trim to fixed length
        n = waveform.shape[1]
        if n > self.max_samples:
            waveform = waveform[:, : self.max_samples]
        elif n < self.max_samples:
            waveform = F.pad(waveform, (0, self.max_samples - n))

        # Apply waveform-level augmentation (e.g. noise injection, time shift)
        if self.waveform_transform is not None:
            try:
                waveform = self.waveform_transform(waveform)
            except Exception:
                pass  # Augmentation errors should never crash training

        mel    = self.mel_transform(waveform)       # (1, n_mels, time)
        if self.spectrogram_transform is not None:
            try:
                mel = self.spectrogram_transform(mel)
            except Exception:
                pass
        mel_db = self.to_db(mel)
        img    = mel_db.repeat(3, 1, 1)             # (3, n_mels, time) — 3-ch "image"

        if self.transform:
            img = self.transform(img)

        return img, label


def _filter_valid_audio_samples(samples: list[tuple[str, int]]) -> list[tuple[str, int]]:
    valid: list[tuple[str, int]] = []
    for path, label in samples:
        try:
            waveform, _ = load_audio_waveform(path)
            if waveform.numel() == 0:
                continue
            valid.append((path, label))
        except Exception:
            continue
    return valid


def load_audio_data(
    data_path: str,
    batch_size: int = 16,
    val_split: float = 0.2,
    sample_rate: int = 22050,
    n_mels: int = 64,
    augmentation: str = "light",
    verify_files: bool = False,
    normalize_waveform: bool = False,
    image_size: int = 224,
    augmentation_options: dict | None = None,
) -> tuple[DataLoader, DataLoader, list[str], dict]:
    data_path = Path(data_path)
    classes   = sorted(d.name for d in data_path.iterdir() if d.is_dir())
    if not classes:
        raise ValueError(f"No class subdirectories found in {data_path}")
    class_to_idx = {c: i for i, c in enumerate(classes)}

    samples = [
        (str(f), class_to_idx[cls])
        for cls in classes
        for f in (data_path / cls).iterdir()
        if f.suffix.lower() in AUDIO_EXTS
    ]
    if verify_files:
        samples = _filter_valid_audio_samples(samples)
    if not samples:
        raise ValueError(f"No audio files found in {data_path}")

    random.seed(42)
    random.shuffle(samples)
    n_val         = max(1, int(len(samples) * val_split))
    train_samples = samples[n_val:]
    val_samples   = samples[:n_val]

    normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    resize    = transforms.Resize((image_size, image_size), antialias=True)
    tf        = transforms.Compose([resize, normalize])

    # Wire augmentation pipeline for waveform-level transforms
    try:
        from data_pipeline.augmentation import get_audio_transforms
        train_wav_aug, _, train_spec_aug, _ = get_audio_transforms(augmentation, options=augmentation_options)
    except Exception:
        train_wav_aug = None
        train_spec_aug = None

    nw = DEFAULTS["num_workers"]
    pm = DEFAULTS["pin_memory"]
    ds_kwargs = dict(
        sample_rate=sample_rate,
        n_mels=n_mels,
        normalize_waveform=normalize_waveform,
    )
    train_loader = DataLoader(
        AudioDataset(train_samples, **ds_kwargs, transform=tf,
                     waveform_transform=train_wav_aug,
                     spectrogram_transform=train_spec_aug),
        batch_size=batch_size, shuffle=True,
        num_workers=nw, pin_memory=pm,
        persistent_workers=(nw > 0),
        prefetch_factor=2 if nw > 0 else None,
    )
    val_loader = DataLoader(
        AudioDataset(val_samples, **ds_kwargs, transform=tf,
                     waveform_transform=None,
                     spectrogram_transform=None),  # no augmentation on validation
        batch_size=batch_size, shuffle=False,
        num_workers=nw, pin_memory=pm,
        persistent_workers=(nw > 0),
        prefetch_factor=2 if nw > 0 else None,
    )

    preprocessing_config = {
        "modality": "audio",
        "sample_rate": sample_rate,
        "n_mels": n_mels,
        "n_fft": 1024,
        "hop_length": 512,
        "max_duration_sec": 10,
        "resize": [image_size, image_size],
        "mean": [0.5, 0.5, 0.5],
        "std": [0.5, 0.5, 0.5],
        "verify_files": verify_files,
        "normalize_waveform": normalize_waveform,
        "augmentation_level": augmentation,
        "augmentation_options": augmentation_options or {},
    }

    return train_loader, val_loader, classes, preprocessing_config

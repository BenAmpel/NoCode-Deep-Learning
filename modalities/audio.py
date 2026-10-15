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
from pathlib import Path

import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from config import DEFAULTS

AUDIO_EXTS = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}


# Decoding lives in the bundle runtime so training and exported inference
# read audio identically.
from export.bundle_runtime import load_audio_waveform, needs_ffmpeg, scale_spectrogram  # noqa: E402


def audio_needs_ffmpeg_message(paths: list[str]) -> str:
    """Explain that these files can't be decoded until ffmpeg is installed."""
    exts = sorted({Path(p).suffix.lower() for p in paths})
    return (f"{len(paths)} audio file(s) ({', '.join(exts)}) need ffmpeg to decode, and ffmpeg "
            "isn't installed. Install ffmpeg, or convert them to WAV, MP3, FLAC or OGG.")


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
        # Scale dB (roughly -100..+40) to [0, 1] before Normalize(0.5, 0.5).
        # Without this the network saw values far outside the [-1, 1] range
        # inference (and pretrained image backbones) assume.
        mel_db = scale_spectrogram(self.to_db(mel))
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
    else:
        # Fail before training starts, not on the first undecodable batch.
        blocked = [path for path, _ in samples if needs_ffmpeg(path)]
        if blocked:
            raise ValueError(audio_needs_ffmpeg_message(blocked) + (
                " Or tick 'Skip unreadable audio files' to train without them."))
    if not samples:
        raise ValueError(f"No audio files found in {data_path}")

    from data_pipeline.splits import stratified_split
    train_idx, val_idx = stratified_split([label for _, label in samples], val_split)
    train_samples = [samples[i] for i in train_idx]
    val_samples   = [samples[i] for i in val_idx]

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

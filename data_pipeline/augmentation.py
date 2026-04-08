"""
Returns torchvision / torchaudio transform pipelines per modality and augmentation level.
Levels: "none", "light", "medium", "heavy"
"""
from __future__ import annotations

AUGMENTATION_LEVELS = ["none", "light", "medium", "heavy"]


def _resolve_aug_flags(level: str, explicit: dict | None, defaults: dict[str, dict[str, bool]]) -> dict[str, bool]:
    resolved = dict(defaults.get(level, defaults.get("none", {})))
    if explicit:
        for key, value in explicit.items():
            if value is not None:
                resolved[key] = bool(value)
    return resolved


def get_image_transforms(
    level: str,
    image_size: int = 224,
    options: dict | None = None,
    normalization: dict | None = None,
    force_grayscale: bool = False,
):
    """Returns (train_transform, val_transform) for images."""
    from torchvision import transforms
    MEAN = (normalization or {}).get("mean", [0.485, 0.456, 0.406])
    STD  = (normalization or {}).get("std", [0.229, 0.224, 0.225])

    prefix_ops = []
    if force_grayscale:
        prefix_ops.extend([transforms.Grayscale(num_output_channels=3)])

    base_val = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        *prefix_ops,
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])

    default_flags = {
        "none": {"horizontal_flip": False, "vertical_flip": False, "rotation": False, "color_jitter": False, "grayscale": False, "perspective": False},
        "light": {"horizontal_flip": True, "vertical_flip": False, "rotation": False, "color_jitter": False, "grayscale": False, "perspective": False},
        "medium": {"horizontal_flip": True, "vertical_flip": False, "rotation": True, "color_jitter": True, "grayscale": False, "perspective": False},
        "heavy": {"horizontal_flip": True, "vertical_flip": True, "rotation": True, "color_jitter": True, "grayscale": True, "perspective": True},
    }
    flags = _resolve_aug_flags(level, options, default_flags)

    aug_ops = []
    if flags.get("horizontal_flip"):
        aug_ops.append(transforms.RandomHorizontalFlip(p=0.5))
    if flags.get("vertical_flip"):
        aug_ops.append(transforms.RandomVerticalFlip(p=0.2))
    if flags.get("color_jitter"):
        jitter = (0.2, 0.2, 0.1, 0.0) if level in {"light", "medium"} else (0.4, 0.4, 0.2, 0.1)
        aug_ops.append(
            transforms.ColorJitter(
                brightness=jitter[0],
                contrast=jitter[1],
                saturation=jitter[2],
                hue=jitter[3],
            )
        )
    if flags.get("rotation"):
        aug_ops.append(transforms.RandomRotation(degrees=10 if level != "heavy" else 30))
    if flags.get("grayscale"):
        aug_ops.append(transforms.RandomGrayscale(p=0.1))
    if flags.get("perspective"):
        aug_ops.append(transforms.RandomPerspective(distortion_scale=0.2, p=0.3))

    if not aug_ops:
        return base_val, base_val

    train_tf = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        *prefix_ops,
        *aug_ops,
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    return train_tf, base_val


def get_audio_transforms(level: str, options: dict | None = None):
    """Returns (train_waveform_aug, val_waveform_aug, train_spec_aug, val_spec_aug)."""
    import torch
    import torchaudio.transforms as AT

    def add_noise(waveform: "torch.Tensor", snr_db: float = 20.0) -> "torch.Tensor":
        noise = torch.randn_like(waveform)
        signal_power = waveform.pow(2).mean()
        noise_power  = noise.pow(2).mean()
        scale = (signal_power / (noise_power * 10 ** (snr_db / 10) + 1e-9)).sqrt()
        return (waveform + scale * noise).clamp(-1, 1)

    def time_shift(waveform: "torch.Tensor", shift_frac: float = 0.1) -> "torch.Tensor":
        shift = int(waveform.shape[-1] * shift_frac)
        return torch.roll(waveform, shifts=shift, dims=-1)

    def gain_jitter(waveform: "torch.Tensor", amount: float = 0.15) -> "torch.Tensor":
        gain = 1.0 + (torch.rand(1, device=waveform.device) * 2 - 1) * amount
        return (waveform * gain).clamp(-1, 1)

    default_flags = {
        "none": {"noise": False, "time_shift": False, "gain_jitter": False, "time_mask": False, "freq_mask": False},
        "light": {"noise": True, "time_shift": False, "gain_jitter": False, "time_mask": False, "freq_mask": False},
        "medium": {"noise": True, "time_shift": True, "gain_jitter": True, "time_mask": True, "freq_mask": False},
        "heavy": {"noise": True, "time_shift": True, "gain_jitter": True, "time_mask": True, "freq_mask": True},
    }
    flags = _resolve_aug_flags(level, options, default_flags)

    time_mask_param = 12 if level != "heavy" else 20
    freq_mask_param = 8 if level != "heavy" else 14
    time_mask_tf = AT.TimeMasking(time_mask_param=time_mask_param)
    freq_mask_tf = AT.FrequencyMasking(freq_mask_param=freq_mask_param)

    def train_waveform_aug(x):
        if flags.get("noise"):
            x = add_noise(x, snr_db=25 if level == "light" else (20 if level == "medium" else 15))
        if flags.get("time_shift"):
            x = time_shift(x, 0.05 if level != "heavy" else 0.1)
        if flags.get("gain_jitter"):
            x = gain_jitter(x, amount=0.12 if level != "heavy" else 0.2)
        return x

    def train_spec_aug(spec):
        if flags.get("time_mask"):
            spec = time_mask_tf(spec)
        if flags.get("freq_mask"):
            spec = freq_mask_tf(spec)
        return spec

    waveform_identity = lambda x: x
    spec_identity = lambda x: x
    has_waveform_aug = any(flags.get(name) for name in ("noise", "time_shift", "gain_jitter"))
    has_spec_aug = any(flags.get(name) for name in ("time_mask", "freq_mask"))
    return (
        train_waveform_aug if has_waveform_aug else waveform_identity,
        waveform_identity,
        train_spec_aug if has_spec_aug else spec_identity,
        spec_identity,
    )


def get_tabular_augmentation(level: str):
    """Returns a function that adds Gaussian noise to feature vectors (numpy arrays)."""
    import numpy as np
    noise_map = {"none": 0.0, "light": 0.01, "medium": 0.05, "heavy": 0.1}
    sigma = noise_map.get(level, 0.0)

    def augment(X: "np.ndarray") -> "np.ndarray":
        if sigma == 0.0:
            return X
        return X + np.random.normal(0, sigma, X.shape).astype(X.dtype)

    return augment


def get_timeseries_augmentation(level: str):
    """Returns a function that jitters and scales time-series windows (numpy arrays)."""
    import numpy as np
    params = {
        "none":   (0.0,  1.0),
        "light":  (0.01, 0.02),
        "medium": (0.05, 0.1),
        "heavy":  (0.1,  0.2),
    }
    jitter_sigma, scale_sigma = params.get(level, (0.0, 0.0))

    def augment(x: "np.ndarray") -> "np.ndarray":
        if jitter_sigma:
            x = x + np.random.normal(0, jitter_sigma, x.shape).astype(x.dtype)
        if scale_sigma:
            scale = 1 + np.random.normal(0, scale_sigma)
            x = (x * scale).astype(x.dtype)
        return x

    return augment

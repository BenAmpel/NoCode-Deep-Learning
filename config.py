import os
import torch


def get_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


DEVICE = get_device()


def _num_workers() -> int:
    """
    Safe default: half of available CPU cores, capped at 8.
    Returns 0 on Windows (fork-based multiprocessing causes issues there).
    """
    import platform
    if platform.system() == "Windows":
        return 0
    try:
        return min(8, max(0, (os.cpu_count() or 1) // 2))
    except Exception:
        return 0


# Compute once at import time
_NW = _num_workers()

DEFAULTS = {
    "batch_size":               16,
    "epochs":                   10,
    "learning_rate":            1e-3,
    "dropout":                  0.3,
    "optimizer":                "adam",
    # >0 enables parallel data loading for file-based modalities (image/audio/video).
    # In-memory loaders (tabular/text/timeseries) keep num_workers=0 — MP overhead
    # outweighs any gain when data is already in RAM.
    "num_workers":              _NW,
    # pin_memory speeds up CPU→GPU transfers on CUDA; unsupported on MPS/CPU.
    "pin_memory":               (DEVICE == "cuda"),
    "early_stopping_patience":  3,
    "val_split":                0.2,
    "seed":                     42,
}

VIDEO_DEFAULTS = {
    "n_frames":   8,
    "frame_size": 112,
    "batch_size": 4,   # smaller default for video — heavier batches
}

AUDIO_DEFAULTS = {
    "sample_rate":      22050,
    "n_mels":           64,
    "hop_length":       512,
    "n_fft":            1024,
    "max_duration_sec": 10,
}

TEXT_DEFAULTS = {
    "max_length": 128,
}

TIMESERIES_DEFAULTS = {
    "window_size": 50,
    "stride":      1,
}

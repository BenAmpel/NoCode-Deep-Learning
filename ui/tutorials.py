"""
Download and prepare built-in tutorial datasets for each modality.

Each ``prepare_*`` function is idempotent — if the destination directory
already exists and is non-empty it returns immediately.  All functions write
data into the directory they receive so callers control the location.
"""
from __future__ import annotations

import csv
import math
import random
import tempfile
import urllib.request
import zipfile
from pathlib import Path


# ---------------------------------------------------------------------------
# Image — MNIST handwritten digits
# ---------------------------------------------------------------------------

def prepare_mnist_tutorial(dest: Path) -> None:
    """300 PNG images per class (digits 0-9) from the MNIST training split."""
    if dest.exists() and _non_empty(dest):
        return
    from torchvision.datasets import MNIST  # project dependency

    dest.mkdir(parents=True, exist_ok=True)
    for cls in range(10):
        (dest / str(cls)).mkdir(exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp:
        dataset = MNIST(root=tmp, train=True, download=True, transform=None)
        counts: dict[int, int] = {i: 0 for i in range(10)}
        per_class = 300
        for img, label in dataset:
            if counts[label] >= per_class:
                if all(v >= per_class for v in counts.values()):
                    break
                continue
            counts[label] += 1
            img.save(dest / str(label) / f"train_{counts[label]:05d}.png")


# ---------------------------------------------------------------------------
# Tabular — Iris species classification
# ---------------------------------------------------------------------------

def prepare_iris_tutorial(dest: Path) -> None:
    """Iris dataset: 150 rows, 4 numeric features, 3 species classes → iris.csv."""
    if dest.exists() and _non_empty(dest):
        return
    from sklearn.datasets import load_iris  # project dependency

    iris = load_iris()
    dest.mkdir(parents=True, exist_ok=True)
    csv_path = dest / "iris.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["sepal_length", "sepal_width", "petal_length", "petal_width", "label"])
        for row, target in zip(iris.data, iris.target):
            writer.writerow([*row, iris.target_names[target]])


# ---------------------------------------------------------------------------
# Text — 20 Newsgroups topic classification
# ---------------------------------------------------------------------------

def prepare_newsgroups_tutorial(dest: Path) -> None:
    """3 newsgroup topics (~900 docs total) cleaned and saved as newsgroups.csv."""
    if dest.exists() and _non_empty(dest):
        return
    from sklearn.datasets import fetch_20newsgroups  # project dependency

    categories = ["sci.space", "rec.sport.hockey", "talk.politics.misc"]
    data = fetch_20newsgroups(
        subset="train",
        categories=categories,
        remove=("headers", "footers", "quotes"),
    )
    dest.mkdir(parents=True, exist_ok=True)
    csv_path = dest / "newsgroups.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["text", "label"])
        for text, target in zip(data.data, data.target):
            cleaned = " ".join(text.split())[:2000]
            if len(cleaned) < 20:
                continue
            writer.writerow([cleaned, data.target_names[target]])


# ---------------------------------------------------------------------------
# Audio — Free Spoken Digit Dataset (FSDD)
# ---------------------------------------------------------------------------

def prepare_audio_tutorial(dest: Path) -> None:
    """
    Digits 0-9 spoken by multiple speakers (~50 WAV clips per class).
    Downloads the Free Spoken Digit Dataset from GitHub (~10 MB).
    """
    if dest.exists() and _non_empty(dest):
        return

    url = (
        "https://github.com/Jakobovski/free-spoken-digit-dataset"
        "/archive/refs/heads/master.zip"
    )
    dest.mkdir(parents=True, exist_ok=True)
    for d in range(10):
        (dest / str(d)).mkdir(exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp:
        zip_path = Path(tmp) / "fsdd.zip"
        urllib.request.urlretrieve(url, zip_path)
        with zipfile.ZipFile(zip_path) as zf:
            for name in zf.namelist():
                if not name.endswith(".wav"):
                    continue
                fname = Path(name).name          # e.g. "0_george_0.wav"
                parts = fname.split("_")
                if len(parts) < 2 or not parts[0].isdigit():
                    continue
                digit = parts[0]
                (dest / digit / fname).write_bytes(zf.read(name))


# ---------------------------------------------------------------------------
# Timeseries — synthetic sinusoidal signals
# ---------------------------------------------------------------------------

def prepare_timeseries_tutorial(dest: Path) -> None:
    """
    3 signal classes (low-freq sine, high-freq sine, linear ramp),
    200 series each, 50 timesteps — saved as timeseries.csv.
    """
    if dest.exists() and _non_empty(dest):
        return

    dest.mkdir(parents=True, exist_ok=True)
    rng = random.Random(42)
    n_steps = 50
    n_per_class = 200

    configs = [
        ("low_freq_sine",  lambda t: math.sin(2 * math.pi * t / n_steps)),
        ("high_freq_sine", lambda t: math.sin(6 * math.pi * t / n_steps)),
        ("linear_ramp",    lambda t: t / n_steps),
    ]

    rows: list[list] = []
    for label, fn in configs:
        for _ in range(n_per_class):
            noise = [rng.gauss(0, 0.05) for _ in range(n_steps)]
            values = [round(fn(t) + noise[t], 4) for t in range(n_steps)]
            rows.append(values + [label])

    rng.shuffle(rows)
    col_names = [f"t{i}" for i in range(n_steps)] + ["label"]
    csv_path = dest / "timeseries.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(col_names)
        writer.writerows(rows)


# ---------------------------------------------------------------------------
# Video — synthetic shape-motion clips
# ---------------------------------------------------------------------------

def prepare_video_tutorial(dest: Path) -> None:
    """
    3 shape classes (circle, square, triangle) with moving coloured shapes,
    30 MP4 clips each at 64×64 px.  Uses OpenCV (project dependency).
    """
    if dest.exists() and _non_empty(dest):
        return

    import cv2
    import numpy as np

    classes = {
        "circle":   _draw_circle,
        "square":   _draw_square,
        "triangle": _draw_triangle,
    }
    n_per_class = 30
    fps = 10
    n_frames = 20
    size = 64

    dest.mkdir(parents=True, exist_ok=True)
    rng = random.Random(42)

    for cls_name, draw_fn in classes.items():
        (dest / cls_name).mkdir(exist_ok=True)
        for i in range(n_per_class):
            path = str(dest / cls_name / f"{cls_name}_{i:03d}.mp4")
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out = cv2.VideoWriter(path, fourcc, fps, (size, size))
            color = (rng.randint(100, 255), rng.randint(100, 255), rng.randint(100, 255))
            cx, cy = rng.randint(20, 44), rng.randint(20, 44)
            dx, dy = rng.choice([-1, 1]) * 2, rng.choice([-1, 1]) * 2
            for _ in range(n_frames):
                frame = np.zeros((size, size, 3), dtype=np.uint8)
                draw_fn(frame, cx, cy, 12, color)
                out.write(frame)
                cx = max(12, min(size - 12, cx + dx))
                cy = max(12, min(size - 12, cy + dy))
            out.release()


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _non_empty(path: Path) -> bool:
    """Return True if *path* contains at least one file anywhere in its tree."""
    return any(True for _ in path.rglob("*") if _.is_file())


def _draw_circle(frame, cx: int, cy: int, r: int, color: tuple) -> None:
    import cv2
    cv2.circle(frame, (cx, cy), r, color, -1)


def _draw_square(frame, cx: int, cy: int, r: int, color: tuple) -> None:
    import cv2
    cv2.rectangle(frame, (cx - r, cy - r), (cx + r, cy + r), color, -1)


def _draw_triangle(frame, cx: int, cy: int, r: int, color: tuple) -> None:
    import cv2
    import numpy as np
    pts = np.array([[cx, cy - r], [cx - r, cy + r], [cx + r, cy + r]])
    cv2.fillPoly(frame, [pts], color)

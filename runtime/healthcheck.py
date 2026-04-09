from __future__ import annotations

import importlib
import shutil
import socket
import sys

from runtime.runtime_setup import PROJECT_ROOT


REQUIRED_IMPORTS = [
    "gradio",
    "torch",
    "torchvision",
    "torchaudio",
    "numpy",
    "pandas",
    "sklearn",
    "matplotlib",
    "transformers",
    "onnxruntime",
    "fastapi",
]


def _port_is_available(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind((host, port))
        except OSError:
            return False
    return True


def run_healthcheck(server_port: int = 7860) -> list[str]:
    messages: list[str] = []

    if sys.version_info < (3, 10):
        raise RuntimeError("Python 3.10 or newer is required.")
    messages.append(f"Python {sys.version.split()[0]}")

    for module_name in REQUIRED_IMPORTS:
        importlib.import_module(module_name)
    messages.append("Core dependencies import successfully")

    from config import DEVICE

    outputs_dir = PROJECT_ROOT / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)
    probe = outputs_dir / ".healthcheck_write_test"
    probe.write_text("ok\n")
    probe.unlink()
    messages.append(f"Outputs directory writable: {outputs_dir.resolve()}")

    if not _port_is_available("127.0.0.1", server_port):
        raise RuntimeError(f"Port {server_port} is already in use.")
    messages.append(f"Port {server_port} is available")

    ffmpeg_path = shutil.which("ffmpeg")
    if ffmpeg_path:
        messages.append(f"Compressed audio support ready: ffmpeg found at {ffmpeg_path}")
    else:
        messages.append(
            "Compressed audio note: ffmpeg not found. WAV audio will work, but mp3/m4a/ogg/flac files may not decode on this machine."
        )

    messages.append(f"Detected torch device: {DEVICE}")
    return messages

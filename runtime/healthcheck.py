from __future__ import annotations

import importlib
import importlib.util
import shutil
import socket
import sys

from runtime.runtime_setup import PROJECT_ROOT


REQUIRED_IMPORTS = [
    "gradio",
    "torch",
    "torchvision",
    "torchaudio",
    "soundfile",
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


def run_healthcheck(server_port: int = 7860, import_all: bool = False) -> list[str]:
    """Checks to run before the server starts.

    A normal launch only confirms each required package is installed
    (find_spec reads no package code). Importing them all here cost about
    3 seconds on every launch, mostly torchvision, scikit-learn and
    transformers, which the app itself loads only when a task needs them.
    ``import_all`` imports every one, so a broken wheel fails loudly; the
    smoke test and --health-check-only use it, and so does first-launch CI.
    """
    messages: list[str] = []

    if not ((3, 11) <= sys.version_info[:2] <= (3, 13)):
        raise RuntimeError("Python 3.11-3.13 is required (numpy 2.4 / scikit-learn 1.8 need >= 3.11).")
    messages.append(f"Python {sys.version.split()[0]}")

    if import_all:
        for module_name in REQUIRED_IMPORTS:
            importlib.import_module(module_name)
        messages.append("Core dependencies import successfully")
    else:
        missing = [name for name in REQUIRED_IMPORTS if importlib.util.find_spec(name) is None]
        if missing:
            raise RuntimeError(
                f"Missing packages: {', '.join(missing)}. Delete the '.venv' folder next to the app "
                "and launch again to reinstall."
            )
        messages.append("Core dependencies are installed")

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
        messages.append(f"Audio support: all formats (ffmpeg found at {ffmpeg_path})")
    else:
        messages.append(
            "Audio support: wav/mp3/flac/ogg. ffmpeg not found, so .m4a files won't decode on this machine."
        )

    messages.append(f"Detected torch device: {DEVICE}")
    return messages

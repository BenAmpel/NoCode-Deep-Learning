from __future__ import annotations

import hashlib
import json
import os
import platform
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_VENV_DIR = PROJECT_ROOT / ".venv"
REQUIREMENTS_FILE = PROJECT_ROOT / "requirements.txt"


def resolve_venv_dir(raw: str | None = None) -> Path:
    if not raw:
        return DEFAULT_VENV_DIR
    candidate = Path(raw)
    return candidate if candidate.is_absolute() else (PROJECT_ROOT / candidate).resolve()


def install_state_path(venv_dir: Path) -> Path:
    return venv_dir / "install_state.json"


def requirements_hash() -> str:
    return hashlib.sha256(REQUIREMENTS_FILE.read_bytes()).hexdigest()


def requirement_names() -> list[str]:
    """Lower-cased package names pinned in requirements.txt."""
    names = []
    for line in REQUIREMENTS_FILE.read_text().splitlines():
        line = line.split("#")[0].strip()
        if line:
            names.append(line.split("==")[0].split("[")[0].strip().lower())
    return names


def load_install_state(venv_dir: Path) -> dict | None:
    state_path = install_state_path(venv_dir)
    if not state_path.is_file():
        return None
    try:
        return json.loads(state_path.read_text())
    except json.JSONDecodeError:
        return None


def _python_minor(version: str) -> str:
    return ".".join(str(version).split(".")[:2])


def save_install_state(venv_dir: Path, python_path: Path | None = None) -> dict:
    """Record a finished install. ``python_path`` is the interpreter the
    packages went into; it defaults to the venv's own python and is the
    system python for the Windows no-venv fallback."""
    state = {
        "requirements_hash": requirements_hash(),
        "python_version": platform.python_version(),
        "machine": platform.machine(),
        "platform": platform.platform(),
        "venv_python": str(python_path or venv_python_path(venv_dir)),
        "requirements": requirement_names(),
    }
    install_state_path(venv_dir).write_text(json.dumps(state, indent=2) + "\n")
    return state


def install_is_current(venv_dir: Path) -> tuple[bool, str]:
    """Is the environment in ``venv_dir`` usable as-is?

    Besides the requirements hash, catch environments that broke underneath
    the marker: the base Python was removed or upgraded (the venv's python
    becomes a dangling link), or the folder moved to a different CPU
    architecture (e.g. migrated from an Intel Mac). Any of these would
    otherwise fail at launch forever without triggering a reinstall.
    """
    state = load_install_state(venv_dir)
    if not state:
        return False, "Install marker not found."
    if state.get("requirements_hash") != requirements_hash():
        return False, "Pinned dependencies changed since the last install."
    recorded_python = Path(state.get("venv_python") or venv_python_path(venv_dir))
    if not recorded_python.is_file():  # is_file() is False for a dangling link
        return False, "The environment's Python is missing (was Python upgraded or removed?)."
    if _python_minor(state.get("python_version", "")) != _python_minor(platform.python_version()):
        return False, "Python version changed since the last install."
    if state.get("machine") != platform.machine():
        return False, "CPU architecture changed since the last install."
    return True, "Install marker is current."


# install.py exits with this code when it cannot create a virtual environment
# at all (as opposed to a pip/network failure). Only then is the Windows
# bootstrap's install-into-system-Python fallback appropriate.
VENV_UNAVAILABLE_EXIT = 3

TORCH_CPU_INDEX = "https://download.pytorch.org/whl/cpu"
_PIP_NETWORK_FLAGS = ["--retries", "5", "--timeout", "60", "--progress-bar", "off"]


def pip_install_commands(python: Path | str) -> list[list[str]]:
    """pip commands that install requirements.txt into ``python``'s environment.

    macOS and Windows get CPU/MPS torch wheels from PyPI, so only PyPI is
    used there. On Linux, PyPI's torch wheels bundle CUDA, so the torch
    packages alone come from PyTorch's CPU index first. Mixing indexes in
    one command (--extra-index-url) would let either index supply *any*
    package, a dependency-confusion risk.
    """
    base = [str(python), "-m", "pip", "install", *_PIP_NETWORK_FLAGS]
    commands = []
    if sys.platform.startswith("linux"):
        torch_pins = [
            line.strip() for line in REQUIREMENTS_FILE.read_text().splitlines()
            if line.strip().split("==")[0] in {"torch", "torchvision", "torchaudio"}
        ]
        commands.append([*base, "--index-url", TORCH_CPU_INDEX, *torch_pins])
    commands.append([*base, "-r", str(REQUIREMENTS_FILE)])
    return commands


def find_free_port(start: int = 7860, attempts: int = 20, host: str = "127.0.0.1") -> int:
    """First port from ``start`` that can be bound, so a leftover server
    (or another app) on 7860 doesn't break or hijack the launch."""
    import socket

    for port in range(start, start + attempts):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            try:
                sock.bind((host, port))
            except OSError:
                continue
        return port
    raise RuntimeError(f"No free port between {start} and {start + attempts - 1}.")


def venv_python_path(venv_dir: Path) -> Path:
    if os.name == "nt":
        return venv_dir / "Scripts" / "python.exe"
    return venv_dir / "bin" / "python"


def running_inside_venv(venv_dir: Path) -> bool:
    current = Path(sys.executable).resolve()
    expected = venv_python_path(venv_dir).resolve()
    return current == expected

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


def load_install_state(venv_dir: Path) -> dict | None:
    state_path = install_state_path(venv_dir)
    if not state_path.is_file():
        return None
    try:
        return json.loads(state_path.read_text())
    except json.JSONDecodeError:
        return None


def save_install_state(venv_dir: Path) -> dict:
    state = {
        "requirements_hash": requirements_hash(),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "venv_python": str(venv_python_path(venv_dir)),
    }
    install_state_path(venv_dir).write_text(json.dumps(state, indent=2) + "\n")
    return state


def install_is_current(venv_dir: Path) -> tuple[bool, str]:
    state = load_install_state(venv_dir)
    if not state:
        return False, "Install marker not found."
    if state.get("requirements_hash") != requirements_hash():
        return False, "Pinned dependencies changed since the last install."
    return True, "Install marker is current."


def venv_python_path(venv_dir: Path) -> Path:
    if os.name == "nt":
        return venv_dir / "Scripts" / "python.exe"
    return venv_dir / "bin" / "python"


def running_inside_venv(venv_dir: Path) -> bool:
    current = Path(sys.executable).resolve()
    expected = venv_python_path(venv_dir).resolve()
    return current == expected

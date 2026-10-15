"""
Windows bootstrap — mirrors the macOS runtime bootstrap.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path


APP_NAME = "NoCode-DL"
USER_ROOT = Path(os.environ.get("LOCALAPPDATA", Path.home())) / APP_NAME
RUNTIME_APP_ROOT = USER_ROOT / "app"
VENV_DIR = USER_ROOT / ".venv"
VERSION_FILE = ".bundle_version"
SYNC_EXCLUDES = {
    "__pycache__",
    ".venv",
    "outputs",
    "build",
    "dist",
}
# User data inside the runtime app folder that must survive an upgrade
# (trained models, checkpoints, exports, run history, project state).
PRESERVE_ON_UPGRADE = {"outputs"}
SERVER_PORT = 7860


def _bundled_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _read_version(root: Path) -> str:
    version_path = root / VERSION_FILE
    if not version_path.is_file():
        raise RuntimeError(f"Missing bundle version file at {version_path}")
    return version_path.read_text(encoding="utf-8").strip()


def _sync_bundle_to_runtime(source_root: Path, target_root: Path) -> None:
    source_version = _read_version(source_root)
    target_version = None
    if (target_root / VERSION_FILE).is_file():
        target_version = (target_root / VERSION_FILE).read_text(encoding="utf-8").strip()

    if target_root.exists() and source_version == target_version:
        return

    # Replace the previous version's files but keep user data. Deleting the
    # whole folder here used to wipe every trained model on each upgrade.
    if target_root.exists():
        for child in target_root.iterdir():
            if child.name in PRESERVE_ON_UPGRADE:
                continue
            if child.is_dir() and not child.is_symlink():
                shutil.rmtree(child)
            else:
                child.unlink()

    def ignore(_dir: str, names: list[str]) -> set[str]:
        return {name for name in names if name in SYNC_EXCLUDES}

    shutil.copytree(source_root, target_root, ignore=ignore, dirs_exist_ok=True)


def _run(cmd: list[str], *, cwd: Path) -> None:
    subprocess.run(cmd, cwd=cwd, check=True)


def _bootstrap_without_venv(app_root: Path, venv_dir: Path) -> None:
    """Install into the launching Python when it cannot create a venv.

    Only used when install.py reports VENV_UNAVAILABLE_EXIT. pip comes from the
    interpreter's own bundled ensurepip rather than a downloaded get-pip.py.
    """
    from runtime.runtime_setup import pip_install_commands, save_install_state

    python = sys.executable
    try:
        subprocess.run([python, "-m", "pip", "--version"], capture_output=True, check=True)
        print("[bootstrap] pip already available", flush=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("[bootstrap] installing pip with ensurepip ...", flush=True)
        subprocess.run([python, "-m", "ensurepip", "--upgrade"], cwd=app_root, check=True)

    for cmd in pip_install_commands(python):
        subprocess.run(cmd, cwd=app_root, check=True)

    venv_dir.mkdir(parents=True, exist_ok=True)
    save_install_state(venv_dir, python_path=Path(python))


def main() -> int:
    source_root = _bundled_root()
    print(f"[bootstrap] syncing bundled app from {source_root}", flush=True)
    USER_ROOT.mkdir(parents=True, exist_ok=True)
    _sync_bundle_to_runtime(source_root, RUNTIME_APP_ROOT)

    if str(RUNTIME_APP_ROOT) not in sys.path:
        sys.path.insert(0, str(RUNTIME_APP_ROOT))

    from runtime.runtime_setup import (
        VENV_UNAVAILABLE_EXIT,
        find_free_port,
        install_is_current,
        load_install_state,
    )

    install_script = RUNTIME_APP_ROOT / "install.py"
    run_script = RUNTIME_APP_ROOT / "run_local.py"

    if not install_script.is_file() or not run_script.is_file():
        raise RuntimeError("Bundled app is missing install.py or run_local.py.")

    is_current, reason = install_is_current(VENV_DIR)
    if not is_current:
        print(f"[bootstrap] {reason} Installing dependencies into {VENV_DIR}", flush=True)
        code = subprocess.run(
            [sys.executable, str(install_script), "--venv-dir", str(VENV_DIR)], cwd=RUNTIME_APP_ROOT,
        ).returncode
        if code == VENV_UNAVAILABLE_EXIT:
            # This Python can't make virtual environments at all.
            print("[bootstrap] virtual environments unavailable; installing into this Python", flush=True)
            _bootstrap_without_venv(RUNTIME_APP_ROOT, VENV_DIR)
        elif code != 0:
            # A pip/network failure: retrying the same ~2 GB install elsewhere
            # would only hide the real error.
            raise RuntimeError(
                f"Installing dependencies failed (exit code {code}). See the messages above; "
                "check your internet connection or proxy, then launch again."
            )

    state = load_install_state(VENV_DIR) or {}
    python = Path(state.get("venv_python") or VENV_DIR / "Scripts" / "python.exe")
    port = find_free_port(SERVER_PORT)

    print(f"[bootstrap] launching app with {python} on port {port}", flush=True)
    # subprocess.run, not os.execv: on Windows execv does not quote arguments
    # (paths with spaces split) and the PowerShell launcher sees exit code 0
    # immediately, hiding failures. Here the app's exit code is passed through.
    return subprocess.run(
        [str(python), str(run_script), "--venv-dir", str(VENV_DIR), "--server-port", str(port)],
        cwd=RUNTIME_APP_ROOT,
    ).returncode


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit as _se:
        if _se.code not in (None, 0):
            print(f"\n[bootstrap] exiting with code {_se.code}", flush=True)
        raise
    except Exception:
        import traceback

        tb = traceback.format_exc()
        log_path = Path(__file__).parent / "NoCode-DL-error.log"
        try:
            log_path.write_text(tb, encoding="utf-8")
        except Exception:
            pass
        print("\n" + "=" * 60, flush=True)
        print("FATAL ERROR — NoCode-DL failed to start", flush=True)
        print("=" * 60, flush=True)
        print(tb, flush=True)
        print(f"Log saved to: {log_path}", flush=True)
        print("=" * 60, flush=True)
        input("Press Enter to close...")
        raise SystemExit(1)

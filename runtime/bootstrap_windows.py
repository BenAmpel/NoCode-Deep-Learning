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

    if target_root.exists():
        shutil.rmtree(target_root)

    def ignore(_dir: str, names: list[str]) -> set[str]:
        return {name for name in names if name in SYNC_EXCLUDES}

    shutil.copytree(source_root, target_root, ignore=ignore)


def _run(cmd: list[str], *, cwd: Path) -> None:
    subprocess.run(cmd, cwd=cwd, check=True)


def _bootstrap_without_venv(app_root: Path, venv_dir: Path) -> None:
    from data_pipeline.network_utils import download_file

    python = sys.executable
    requirements = app_root / "requirements.txt"

    try:
        subprocess.run([python, "-m", "pip", "--version"], capture_output=True, check=True)
        print("[bootstrap] pip already available", flush=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("[bootstrap] downloading get-pip.py ...", flush=True)
        get_pip = app_root / "get-pip.py"
        download_file(
            "https://bootstrap.pypa.io/get-pip.py",
            get_pip,
            allowed_hosts={"bootstrap.pypa.io"},
        )
        subprocess.run([python, str(get_pip)], cwd=app_root, check=True)

    subprocess.run([python, "-m", "pip", "install", "--upgrade", "pip"], cwd=app_root, check=True)
    subprocess.run(
        [
            python, "-m", "pip", "install",
            "-r", str(requirements),
            "--extra-index-url", "https://download.pytorch.org/whl/cpu",
        ],
        cwd=app_root,
        check=True,
    )

    venv_dir.mkdir(parents=True, exist_ok=True)
    scripts_dir = venv_dir / "Scripts"
    scripts_dir.mkdir(exist_ok=True)

    from runtime.runtime_setup import save_install_state
    save_install_state(venv_dir)


def main() -> int:
    source_root = _bundled_root()
    print(f"[bootstrap] syncing bundled app from {source_root}", flush=True)
    USER_ROOT.mkdir(parents=True, exist_ok=True)
    _sync_bundle_to_runtime(source_root, RUNTIME_APP_ROOT)

    if str(RUNTIME_APP_ROOT) not in sys.path:
        sys.path.insert(0, str(RUNTIME_APP_ROOT))

    from runtime.runtime_setup import install_is_current

    install_script = RUNTIME_APP_ROOT / "install.py"
    run_script = RUNTIME_APP_ROOT / "run_local.py"

    if not install_script.is_file() or not run_script.is_file():
        raise RuntimeError("Bundled app is missing install.py or run_local.py.")

    is_current, _reason = install_is_current(VENV_DIR)

    if is_current and not (VENV_DIR / "Scripts" / "python.exe").is_file():
        try:
            import gradio  # noqa: F401
        except ImportError:
            print("[bootstrap] install state is stale — packages missing; reinstalling", flush=True)
            is_current = False
            state_file = VENV_DIR / "install_state.json"
            if state_file.exists():
                state_file.unlink()

    if not is_current:
        print(f"[bootstrap] creating fresh environment in {VENV_DIR}", flush=True)
        try:
            _run([sys.executable, str(install_script), "--venv-dir", str(VENV_DIR)], cwd=RUNTIME_APP_ROOT)
        except subprocess.CalledProcessError:
            print("[bootstrap] venv creation failed — using pip install fallback", flush=True)
            _bootstrap_without_venv(RUNTIME_APP_ROOT, VENV_DIR)

    venv_python = VENV_DIR / "Scripts" / "python.exe"
    if not venv_python.is_file():
        venv_python = Path(sys.executable)

    print(f"[bootstrap] launching app with {venv_python}", flush=True)
    os.execv(
        str(venv_python),
        [
            str(venv_python),
            str(run_script),
            "--venv-dir", str(VENV_DIR),
            "--server-port", str(SERVER_PORT),
        ],
    )


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

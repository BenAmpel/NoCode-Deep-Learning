from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import venv
from pathlib import Path

from runtime_setup import PROJECT_ROOT, REQUIREMENTS_FILE, resolve_venv_dir, save_install_state, venv_python_path


def run(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True, cwd=PROJECT_ROOT)


def build_pip_install_command(venv_python: Path) -> list[str]:
    cmd = [str(venv_python), "-m", "pip", "install", "-r", str(REQUIREMENTS_FILE)]
    # PyTorch wheels live on a separate index
    cmd.extend(["--extra-index-url", "https://download.pytorch.org/whl/cpu"])
    package_cache = PROJECT_ROOT / "packages"
    if package_cache.is_dir():
        cmd.extend(["--find-links", str(package_cache)])
    return cmd


def main() -> int:
    parser = argparse.ArgumentParser(description="Create a fresh local environment for NoCode-DL.")
    parser.add_argument("--venv-dir", default=".venv", help="Virtual environment directory to create.")
    args = parser.parse_args()

    if sys.version_info < (3, 10):
        raise SystemExit("Python 3.10 or newer is required.")

    venv_dir = resolve_venv_dir(args.venv_dir)
    if venv_dir.exists():
        shutil.rmtree(venv_dir)

    print(f"Creating fresh virtual environment at {venv_dir} ...")
    venv.EnvBuilder(with_pip=True, clear=False).create(venv_dir)

    venv_python = venv_python_path(venv_dir)
    run([str(venv_python), "-m", "pip", "install", "--upgrade", "pip"])
    run(build_pip_install_command(venv_python))
    save_install_state(venv_dir)

    print("")
    print("Install complete.")
    print("Next: python run_local.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

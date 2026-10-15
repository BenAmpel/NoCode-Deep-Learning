from __future__ import annotations

import argparse
import platform
import shutil
import subprocess
import sys
import venv
from pathlib import Path

from runtime.runtime_setup import (
    PROJECT_ROOT,
    VENV_UNAVAILABLE_EXIT,
    load_install_state,
    pip_install_commands,
    requirement_names,
    resolve_venv_dir,
    save_install_state,
    venv_python_path,
)

SUPPORTED_PYTHON = ((3, 11), (3, 13))  # numpy 2.4 / scikit-learn 1.8 need >= 3.11


def run(cmd: list[str]) -> None:
    print("$", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=PROJECT_ROOT)


def _can_reuse(venv_dir: Path) -> bool:
    """Keep the existing environment if it was built by this Python on this CPU.

    Upgrading it in place means a failed or interrupted install leaves a
    working environment behind (the install marker isn't updated, so the
    next launch retries) and unchanged packages aren't downloaded again.
    """
    state = load_install_state(venv_dir) or {}
    same_python = ".".join(state.get("python_version", "").split(".")[:2]) == \
        f"{sys.version_info.major}.{sys.version_info.minor}"
    # pip never uninstalls packages dropped from requirements.txt, and some
    # swaps leave conflicting files behind (opencv-python vs -headless both
    # provide cv2), so a removal means starting fresh.
    nothing_removed = set(state.get("requirements", [])) <= set(requirement_names())
    return (
        venv_python_path(venv_dir).is_file()
        and same_python
        and state.get("machine") == platform.machine()
        and "requirements" in state
        and nothing_removed
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Create or update the local environment for NoCode-DL.")
    parser.add_argument("--venv-dir", default=".venv", help="Virtual environment directory to use.")
    parser.add_argument("--fresh", action="store_true", help="Delete and recreate the environment.")
    args = parser.parse_args()

    lo, hi = SUPPORTED_PYTHON
    if not (lo <= sys.version_info[:2] <= hi):
        raise SystemExit(
            f"Python {lo[0]}.{lo[1]}-{hi[0]}.{hi[1]} is required; this is Python {platform.python_version()}."
        )

    venv_dir = resolve_venv_dir(args.venv_dir)
    if args.fresh or not _can_reuse(venv_dir):
        print(f"Creating virtual environment at {venv_dir} ...", flush=True)
        if venv_dir.exists():
            shutil.rmtree(venv_dir)
        try:
            venv.EnvBuilder(with_pip=True).create(venv_dir)
        except Exception as exc:  # e.g. a Python without the venv/ensurepip modules
            print(f"Could not create a virtual environment: {exc}", file=sys.stderr, flush=True)
            shutil.rmtree(venv_dir, ignore_errors=True)
            return VENV_UNAVAILABLE_EXIT
    else:
        print(f"Updating existing environment at {venv_dir} ...", flush=True)

    python = venv_python_path(venv_dir)
    try:
        run([str(python), "-m", "pip", "install", "--upgrade", "pip", "--retries", "5", "--timeout", "60"])
        for cmd in pip_install_commands(python):
            run(cmd)
    except subprocess.CalledProcessError as exc:
        print(
            f"\nDependency install failed (exit code {exc.returncode}). Check your internet "
            "connection or proxy settings, then launch again; finished downloads are reused.",
            file=sys.stderr, flush=True,
        )
        return exc.returncode or 1

    save_install_state(venv_dir)
    print("")
    print("Install complete.")
    print("Next: python run_local.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

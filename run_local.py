from __future__ import annotations

import argparse
import os
import sys
import time

from runtime.runtime_setup import install_is_current, resolve_venv_dir, running_inside_venv, venv_python_path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the local NoCode-DL app after a fresh install.")
    parser.add_argument("--venv-dir", default=".venv", help="Virtual environment directory to use.")
    parser.add_argument("--server-port", type=int, default=7860, help="Port for the local Gradio app.")
    parser.add_argument("--health-check-only", action="store_true", help="Run environment checks and exit.")
    parser.add_argument("--smoke-test", action="store_true", help="Launch briefly without opening a browser, then exit.")
    parser.add_argument("--no-browser", action="store_true", help="Do not auto-open a browser tab.")
    return parser.parse_args()


def _reexec_into_venv(venv_dir) -> None:
    venv_python = venv_python_path(venv_dir)
    if not venv_python.is_file():
        # No venv python.exe — we may be running from an embeddable/system Python
        # that had packages installed directly into its site-packages (Windows fallback).
        # If the key package is importable we can continue without re-execing.
        try:
            import gradio  # noqa: F401
            return
        except ImportError:
            raise SystemExit(
                "Packages not found. Delete the '.venv' folder next to the app and "
                "re-launch to trigger a fresh install."
            )
    script_path = os.path.abspath(__file__)
    os.execv(str(venv_python), [str(venv_python), script_path, *sys.argv[1:]])


def main() -> int:
    args = _parse_args()
    venv_dir = resolve_venv_dir(args.venv_dir)

    is_current, reason = install_is_current(venv_dir)
    if not is_current:
        raise SystemExit(f"{reason} Run `python install.py --venv-dir {args.venv_dir}` first.")

    if not running_inside_venv(venv_dir):
        _reexec_into_venv(venv_dir)

    from runtime.healthcheck import run_healthcheck

    for line in run_healthcheck(server_port=args.server_port):
        print(f"[health] {line}")

    if args.health_check_only:
        return 0

    from app import demo, launch_app

    if args.smoke_test:
        launch_app(
            inbrowser=False,
            server_port=args.server_port,
            prevent_thread_lock=True,
            frontend=False,
            show_api=False,
        )
        try:
            time.sleep(2)
        finally:
            demo.close()
        print("[smoke] Launch and shutdown succeeded.")
        return 0

    launch_app(inbrowser=not args.no_browser, server_port=args.server_port)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit as _se:
        if _se.code not in (None, 0):
            print(f"\n[run_local] ERROR: {_se}", flush=True)
            print("Check the window above for details.", flush=True)
        raise
    except Exception:
        import traceback
        traceback.print_exc()
        raise SystemExit(1)

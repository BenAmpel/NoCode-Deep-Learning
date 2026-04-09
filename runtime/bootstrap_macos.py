from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import threading
import time
from http.client import HTTPConnection
from pathlib import Path


APP_NAME = "NoCode-DL"
USER_ROOT = Path.home() / "Library" / "Application Support" / APP_NAME
RUNTIME_APP_ROOT = USER_ROOT / "app"
VENV_DIR = USER_ROOT / ".venv"
VERSION_FILE = ".bundle_version"
SYNC_EXCLUDES = {
    ".DS_Store",
    ".venv",
    "__pycache__",
    "outputs",
    "build",
    "dist",
}
SERVER_URL = "http://127.0.0.1:7860"
POLL_INTERVAL = 0.5
POLL_TIMEOUT = 180
SERVER_TYPICAL_SECS = 45


def _bundled_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _read_version(root: Path) -> str:
    version_path = root / VERSION_FILE
    if not version_path.is_file():
        raise RuntimeError(f"Missing bundle version file at {version_path}")
    return version_path.read_text(encoding="utf-8").strip()


def _sync_bundle_to_runtime(source_root: Path, target_root: Path) -> None:
    source_version = _read_version(source_root)
    target_version = (
        (target_root / VERSION_FILE).read_text(encoding="utf-8").strip()
        if (target_root / VERSION_FILE).is_file()
        else None
    )

    if target_root.exists() and source_version == target_version:
        return

    if target_root.exists():
        shutil.rmtree(target_root)

    def ignore(_dir: str, names: list[str]) -> set[str]:
        return {name for name in names if name in SYNC_EXCLUDES}

    shutil.copytree(source_root, target_root, ignore=ignore)


def _fmt_elapsed(secs: float) -> str:
    secs = int(secs)
    if secs < 60:
        return f"{secs}s"
    return f"{secs // 60}m {secs % 60:02d}s"


def _fmt_eta(elapsed: float, typical: float) -> str:
    remaining = typical - elapsed
    if remaining > 5:
        return f"~{_fmt_elapsed(remaining)} remaining"
    if remaining > -10:
        return "almost ready\u2026"
    return "taking longer than usual\u2026"


def _wait_for_server(timeout: float = POLL_TIMEOUT) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        conn = None
        try:
            conn = HTTPConnection("127.0.0.1", 7860, timeout=1)
            conn.request("GET", "/")
            response = conn.getresponse()
            if 200 <= response.status < 500:
                return True
        except Exception:
            time.sleep(POLL_INTERVAL)
        finally:
            if conn is not None:
                try:
                    conn.close()
                except Exception:
                    pass
    return False


def main() -> int:
    try:
        import tkinter as tk
        has_tk = True
    except ImportError:
        has_tk = False

    source_root = _bundled_root()
    USER_ROOT.mkdir(parents=True, exist_ok=True)

    error_holder: list[Exception] = []
    status_holder: list[str] = ["Preparing\u2026"]
    detail_holder: list[str] = [""]
    phase_start: list[float] = [time.time()]
    phase_typical: list[float] = [-1.0]
    done_event = threading.Event()

    def set_phase(msg: str, typical_secs: float = -1.0) -> None:
        status_holder[0] = msg
        detail_holder[0] = ""
        phase_start[0] = time.time()
        phase_typical[0] = typical_secs

    def bootstrap_thread() -> None:
        try:
            set_phase("Syncing app files\u2026")
            _sync_bundle_to_runtime(source_root, RUNTIME_APP_ROOT)

            if str(RUNTIME_APP_ROOT) not in sys.path:
                sys.path.insert(0, str(RUNTIME_APP_ROOT))

            from runtime.runtime_setup import install_is_current

            install_script = RUNTIME_APP_ROOT / "install.py"
            run_script = RUNTIME_APP_ROOT / "run_local.py"

            if not install_script.is_file() or not run_script.is_file():
                raise RuntimeError("Bundled app is missing install.py or run_local.py.")

            is_current, _reason = install_is_current(VENV_DIR)
            if not is_current:
                set_phase("Installing dependencies\u2026", typical_secs=300.0)
                detail_holder[0] = "First launch \u2014 this may take a few minutes"
                proc = subprocess.Popen(
                    [sys.executable, str(install_script), "--venv-dir", str(VENV_DIR)],
                    cwd=RUNTIME_APP_ROOT,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                )
                pkg_pattern = re.compile(
                    r"(?:Collecting|Downloading|Installing collected packages:|Successfully installed)\s+(.+)"
                )
                for line in proc.stdout:
                    line = line.rstrip()
                    match = pkg_pattern.match(line)
                    if match:
                        raw = match.group(1).strip()
                        if line.startswith("Installing collected") or line.startswith("Successfully"):
                            pkgs = [pkg.strip() for pkg in raw.split(",")]
                            summary = ", ".join(pkgs[:3])
                            if len(pkgs) > 3:
                                summary += f" +{len(pkgs) - 3} more"
                            verb = "Installed" if line.startswith("Successfully") else "Finalizing"
                            detail_holder[0] = f"{verb}: {summary}"
                        else:
                            pkg_name = raw.split("==")[0].split(">=")[0].split("~=")[0].strip()
                            detail_holder[0] = f"Installing: {pkg_name}"
                proc.wait()
                if proc.returncode != 0:
                    raise RuntimeError(f"install.py exited with code {proc.returncode}")

            venv_python = VENV_DIR / "bin" / "python"
            if not venv_python.is_file():
                raise RuntimeError(f"Expected virtualenv Python at {venv_python}")

            set_phase("Starting server\u2026", typical_secs=SERVER_TYPICAL_SECS)
            detail_holder[0] = "Loading models and initialising Gradio\u2026"
            server_proc = subprocess.Popen(
                [
                    str(venv_python),
                    str(run_script),
                    "--venv-dir", str(VENV_DIR),
                    "--no-browser",
                ],
                cwd=RUNTIME_APP_ROOT,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )

            server_ready = threading.Event()

            def _stream_server() -> None:
                skip = re.compile(r"^\s*$|uvicorn|httpx|HTTP Request")
                for line in server_proc.stdout:
                    if server_ready.is_set():
                        break
                    line = line.rstrip()
                    if line and not skip.search(line):
                        clean = re.sub(r"^\S+\s+\S+\s+\S+\s+", "", line).strip()
                        if clean:
                            detail_holder[0] = clean[:80]

            threading.Thread(target=_stream_server, daemon=True).start()

            ready = _wait_for_server()
            server_ready.set()

            if ready:
                status_holder[0] = "Opening browser\u2026"
                detail_holder[0] = ""
                phase_typical[0] = -1.0
                import webbrowser
                webbrowser.open(SERVER_URL)
                status_holder[0] = "Done"
            else:
                status_holder[0] = "Server did not start in time."
                detail_holder[0] = "Check ~/Library/Logs/NoCode-DL/launcher.log for details."

        except Exception as exc:
            error_holder.append(exc)
            status_holder[0] = f"Error: {exc}"
            detail_holder[0] = ""
        finally:
            done_event.set()

    thread = threading.Thread(target=bootstrap_thread, daemon=True)
    thread.start()

    if has_tk:
        root = tk.Tk()
        root.title("NoCode-DL")
        root.resizable(False, False)
        root.configure(bg="#1a1a2e")

        width, height = 460, 200
        sw = root.winfo_screenwidth()
        sh = root.winfo_screenheight()
        root.geometry(f"{width}x{height}+{(sw - width) // 2}+{(sh - height) // 2}")

        tk.Label(root, text="NoCode-DL", font=("Helvetica Neue", 22, "bold"), fg="#ffffff", bg="#1a1a2e").pack(pady=(22, 4))
        status_var = tk.StringVar(value="Preparing\u2026")
        detail_var = tk.StringVar(value="")
        tk.Label(root, textvariable=status_var, font=("Helvetica Neue", 13, "bold"), fg="#e9fff8", bg="#1a1a2e").pack(pady=(10, 2))
        tk.Label(root, textvariable=detail_var, font=("Helvetica Neue", 11), fg="#b7d3cb", bg="#1a1a2e", wraplength=400, justify="center").pack(pady=(0, 12))
        progress = tk.ttk.Progressbar(root, orient="horizontal", mode="indeterminate", length=360)
        progress.pack(pady=(0, 16))
        progress.start(10)

        def _poll() -> None:
            status_var.set(status_holder[0])
            elapsed = time.time() - phase_start[0]
            if phase_typical[0] > 0 and not detail_holder[0]:
                detail_var.set(_fmt_eta(elapsed, phase_typical[0]))
            else:
                detail_var.set(detail_holder[0])
            if done_event.is_set():
                progress.stop()
                if error_holder:
                    status_var.set(f"Error: {error_holder[0]}")
                root.after(1200, root.destroy)
                return
            root.after(200, _poll)

        root.after(50, _poll)
        root.mainloop()
    else:
        while not done_event.is_set():
            time.sleep(0.2)

    if error_holder:
        raise RuntimeError(str(error_holder[0]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

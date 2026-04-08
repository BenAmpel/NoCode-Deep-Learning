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
SERVER_TYPICAL_SECS = 45  # used for ETA display


def _bundled_root() -> Path:
    return Path(__file__).resolve().parent


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
    # Past expected time — just show elapsed, no ETA
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
        from tkinter import ttk
        has_tk = True
    except ImportError:
        has_tk = False

    source_root = _bundled_root()
    USER_ROOT.mkdir(parents=True, exist_ok=True)

    error_holder: list[Exception] = []
    # [0] = main status line, [1] = detail/timing line, [2] = phase start timestamp or -1
    status_holder: list[str] = ["Preparing\u2026"]
    detail_holder: list[str] = [""]
    phase_start: list[float] = [time.time()]
    phase_typical: list[float] = [-1.0]  # -1 means no ETA
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

            from runtime_setup import install_is_current

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
                    m = pkg_pattern.match(line)
                    if m:
                        raw = m.group(1).strip()
                        # Shorten long package lists
                        if line.startswith("Installing collected") or line.startswith("Successfully"):
                            pkgs = [p.strip() for p in raw.split(",")]
                            summary = ", ".join(pkgs[:3])
                            if len(pkgs) > 3:
                                summary += f" +{len(pkgs) - 3} more"
                            verb = "Installed" if line.startswith("Successfully") else "Finalizing"
                            detail_holder[0] = f"{verb}: {summary}"
                        else:
                            # Trim version specifiers for display
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

            # Stream server output so the user can see what it's doing
            _server_ready = threading.Event()

            def _stream_server() -> None:
                skip = re.compile(r"^\s*$|uvicorn|httpx|HTTP Request")
                for line in server_proc.stdout:
                    if _server_ready.is_set():
                        break
                    line = line.rstrip()
                    if line and not skip.search(line):
                        # Strip log prefix (timestamp + level + logger)
                        clean = re.sub(r"^\S+\s+\S+\s+\S+\s+", "", line).strip()
                        if clean:
                            detail_holder[0] = clean[:80]

            threading.Thread(target=_stream_server, daemon=True).start()

            ready = _wait_for_server()
            _server_ready.set()

            if ready:
                status_holder[0] = "Opening browser\u2026"
                detail_holder[0] = ""
                phase_typical[0] = -1.0
                import webbrowser
                webbrowser.open(SERVER_URL)
                status_holder[0] = "Done"
            else:
                status_holder[0] = "Server did not start in time."
                detail_holder[0] = f"Check ~/Library/Logs/NoCode-DL/launcher.log for details."

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

        w, h = 460, 200
        sw = root.winfo_screenwidth()
        sh = root.winfo_screenheight()
        root.geometry(f"{w}x{h}+{(sw - w) // 2}+{(sh - h) // 2}")

        tk.Label(
            root,
            text="NoCode-DL",
            font=("Helvetica Neue", 22, "bold"),
            fg="#ffffff",
            bg="#1a1a2e",
        ).pack(pady=(22, 4))

        status_var = tk.StringVar(value="Preparing\u2026")
        tk.Label(
            root,
            textvariable=status_var,
            font=("Helvetica Neue", 13, "bold"),
            fg="#ccccee",
            bg="#1a1a2e",
            wraplength=420,
        ).pack()

        detail_var = tk.StringVar(value="")
        tk.Label(
            root,
            textvariable=detail_var,
            font=("Helvetica Neue", 11),
            fg="#7777aa",
            bg="#1a1a2e",
            wraplength=420,
        ).pack(pady=(2, 10))

        style = ttk.Style(root)
        style.theme_use("clam")
        style.configure(
            "NCD.Horizontal.TProgressbar",
            troughcolor="#2d2d4e",
            background="#6c63ff",
            darkcolor="#6c63ff",
            lightcolor="#6c63ff",
            bordercolor="#1a1a2e",
            thickness=8,
        )
        bar = ttk.Progressbar(
            root,
            mode="indeterminate",
            style="NCD.Horizontal.TProgressbar",
            length=400,
        )
        bar.pack(pady=(0, 16))
        bar.start(10)

        def poll() -> None:
            elapsed = time.time() - phase_start[0]
            typical = phase_typical[0]

            status_var.set(status_holder[0])

            # Build detail line: user-set detail + timing suffix
            base_detail = detail_holder[0]
            if typical > 0:
                timing = f"{_fmt_elapsed(elapsed)} elapsed \u2014 {_fmt_eta(elapsed, typical)}"
                detail_var.set(f"{base_detail}\n{timing}" if base_detail else timing)
            else:
                detail_var.set(base_detail)

            if done_event.is_set():
                bar.stop()
                if error_holder:
                    import tkinter.messagebox as mb
                    mb.showerror("NoCode-DL Error", str(error_holder[0]))
                root.destroy()
                return
            root.after(500, poll)

        root.after(500, poll)
        root.mainloop()
    else:
        done_event.wait()
        if error_holder:
            raise error_holder[0]

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

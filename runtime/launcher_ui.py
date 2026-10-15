"""
First-run progress window shared by the macOS and Windows bootstraps.

The bootstrap does its work (sync, dependency install, server start) on a
worker thread and reports through a ``Progress`` object; ``run_window`` shows
it in a small Tk window on the main thread. Without Tk (or without a display)
the same work runs headless.

This module must only use the standard library: it runs under the launching
Python before any dependency is installed.
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
import threading
import time
from http.client import HTTPConnection
from pathlib import Path
from typing import Callable

POLL_INTERVAL = 0.5

# Environment variable the launchers pass through for CI: run the first
# launch for real, but start the app with --smoke-test and never wait on a
# person (no browser, no dialogs left open).
SMOKE_ENV = "NOCODE_DL_SMOKE_TEST"

_PIP_LINE = re.compile(
    r"(?:Collecting|Downloading|Installing collected packages:|Successfully installed)\s+(.+)"
)


def smoke_mode() -> bool:
    return os.environ.get(SMOKE_ENV, "").strip() not in ("", "0")


def fmt_elapsed(secs: float) -> str:
    secs = int(secs)
    if secs < 60:
        return f"{secs}s"
    return f"{secs // 60}m {secs % 60:02d}s"


def fmt_eta(elapsed: float, typical: float) -> str:
    remaining = typical - elapsed
    if remaining > 5:
        return f"~{fmt_elapsed(remaining)} remaining"
    if remaining > -10:
        return "almost ready…"
    return "taking longer than usual…"


def pip_detail(line: str) -> str | None:
    """A short "Installing: torch" style summary of one line of pip output."""
    match = _PIP_LINE.match(line)
    if not match:
        return None
    raw = match.group(1).strip()
    if line.startswith("Installing collected") or line.startswith("Successfully"):
        pkgs = [pkg.strip() for pkg in (raw.split(",") if "," in raw else raw.split())]
        summary = ", ".join(pkgs[:3])
        if len(pkgs) > 3:
            summary += f" +{len(pkgs) - 3} more"
        verb = "Installed" if line.startswith("Successfully") else "Finalizing"
        return f"{verb}: {summary}"
    pkg_name = re.split(r"[=<>~!\[;\s]", raw, maxsplit=1)[0].strip()
    return f"Installing: {pkg_name}"


def wait_for_server(port: int, timeout: float, proc: subprocess.Popen | None = None) -> bool:
    """True once the app answers HTTP on ``port``; False on timeout or if
    ``proc`` (the server) exits first."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        if proc is not None and proc.poll() is not None:
            return False
        conn = None
        try:
            conn = HTTPConnection("127.0.0.1", port, timeout=1)
            conn.request("GET", "/")
            if 200 <= conn.getresponse().status < 500:
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


def kill_tree(proc: subprocess.Popen) -> None:
    """Stop ``proc`` and its children (pip under install.py, for example)."""
    if proc.poll() is not None:
        return
    if os.name == "nt":
        subprocess.run(
            ["taskkill", "/F", "/T", "/PID", str(proc.pid)],
            capture_output=True,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
    else:
        proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()


def open_path(path: Path) -> None:
    try:
        if os.name == "nt":
            os.startfile(str(path))  # type: ignore[attr-defined]
        elif sys.platform == "darwin":
            subprocess.Popen(["open", str(path)])
        else:
            subprocess.Popen(["xdg-open", str(path)])
    except Exception:
        pass


def show_error_box(title: str, message: str) -> None:
    """Last-resort error dialog for when Tk itself is unavailable."""
    if smoke_mode():
        return
    try:
        if os.name == "nt":
            import ctypes

            ctypes.windll.user32.MessageBoxW(None, message, title, 0x10)  # MB_ICONERROR
        elif sys.platform == "darwin":
            text = message.replace("\\", "\\\\").replace('"', '\\"')
            script = f'display dialog "{text}" buttons {{"OK"}} default button "OK" with icon stop'
            subprocess.run(["osascript", "-e", script], capture_output=True)
    except Exception:
        pass


class Progress:
    """State the worker thread writes and the window polls."""

    def __init__(self) -> None:
        self.status = "Preparing…"
        self.detail = ""
        self.phase_start = time.time()
        self.typical = -1.0
        self.error: str | None = None
        self.url: str | None = None  # set while the app runs and the window stays up
        self.done = threading.Event()
        self.cancelled = threading.Event()
        self.window_shown = False

    def phase(self, status: str, typical_secs: float = -1.0, detail: str = "") -> None:
        self.status = status
        self.detail = detail
        self.phase_start = time.time()
        self.typical = typical_secs
        print(f"[bootstrap] {status}", flush=True)

    def fail(self, message: str) -> None:
        self.error = message
        self.status = "NoCode-DL could not start"
        self.detail = ""


def run_window(
    progress: Progress,
    worker: Callable[[], None],
    *,
    log_path: Path,
    on_quit: Callable[[], None] | None = None,
    interactive: bool = True,
) -> None:
    """Run ``worker`` on a thread while showing its progress. Returns when it
    finishes or the user closes the window; after an error, only once the
    user has read it and pressed Close.

    ``on_quit`` runs when the user closes the window or presses Quit (stop
    the install or the server). With ``interactive`` False (smoke tests),
    errors close the window instead of waiting for a click.
    """

    def _target() -> None:
        try:
            worker()
        except Exception as exc:  # shown in the window; the traceback goes to the log
            import traceback

            traceback.print_exc()
            progress.fail(str(exc))
        finally:
            progress.done.set()

    thread = threading.Thread(target=_target, daemon=True)
    thread.start()

    root = None
    try:
        import tkinter as tk
        from tkinter import ttk  # submodule: `import tkinter` alone does not load it

        root = tk.Tk()
    except Exception as exc:  # no Tk, or no display
        print(f"[bootstrap] progress window unavailable ({exc}); continuing without it", flush=True)

    if root is None:
        thread.join()
        if progress.error and interactive:
            show_error_box("NoCode-DL", f"{progress.error}\n\nDetails: {log_path}")
        return

    progress.window_shown = True
    print("[bootstrap] progress window shown", flush=True)

    font = "Segoe UI" if os.name == "nt" else "Helvetica Neue"
    bg, fg, sub = "#1a1a2e", "#e9fff8", "#b7d3cb"
    root.title("NoCode-DL")
    root.resizable(False, False)
    root.configure(bg=bg)
    width, height = 480, 230
    sw, sh = root.winfo_screenwidth(), root.winfo_screenheight()
    root.geometry(f"{width}x{height}+{(sw - width) // 2}+{(sh - height) // 2}")

    def _raise() -> None:
        # Launched from a hidden console the window can open behind others.
        try:
            root.lift()
            root.attributes("-topmost", True)
            root.after(1500, lambda: root.attributes("-topmost", False))
        except Exception:
            pass

    _raise()
    tk.Label(root, text="NoCode-DL", font=(font, 20, "bold"), fg="#ffffff", bg=bg).pack(pady=(20, 4))
    status_var = tk.StringVar(value=progress.status)
    detail_var = tk.StringVar(value="")
    status_label = tk.Label(root, textvariable=status_var, font=(font, 12, "bold"), fg=fg, bg=bg,
                            wraplength=440, justify="center")
    status_label.pack(pady=(8, 2))
    tk.Label(root, textvariable=detail_var, font=(font, 10), fg=sub, bg=bg,
             wraplength=440, justify="center").pack(pady=(0, 10))
    bar = ttk.Progressbar(root, orient="horizontal", mode="indeterminate", length=380)
    bar.pack(pady=(0, 12))
    bar.start(10)
    buttons = tk.Frame(root, bg=bg)
    buttons.pack(pady=(0, 12))
    mode = {"value": "working"}

    def _quit() -> None:
        progress.cancelled.set()
        if on_quit is not None:
            on_quit()
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", _quit)

    def _show_running() -> None:
        mode["value"] = "running"
        bar.stop()
        bar.pack_forget()
        import webbrowser

        tk.Button(buttons, text="Open in browser", width=16,
                  command=lambda: webbrowser.open(progress.url or "")).pack(side="left", padx=6)
        tk.Button(buttons, text="Quit NoCode-DL", width=16, command=_quit).pack(side="left", padx=6)

    def _show_error() -> None:
        mode["value"] = "error"
        for child in buttons.winfo_children():
            child.destroy()
        bar.stop()
        bar.pack_forget()
        status_var.set(progress.status)
        status_label.configure(fg="#ff9b9b")
        detail_var.set(f"{progress.error}\n\nDetails are in {log_path}")
        root.geometry(f"{width}x{height + 110}")
        tk.Button(buttons, text="Open log", width=12,
                  command=lambda: open_path(log_path)).pack(side="left", padx=6)
        tk.Button(buttons, text="Close", width=12, command=root.destroy).pack(side="left", padx=6)
        root.protocol("WM_DELETE_WINDOW", root.destroy)
        _raise()

    def _poll() -> None:
        if progress.done.is_set():
            if progress.error and interactive:
                if mode["value"] != "error":
                    _show_error()
            else:
                status_var.set(progress.status)
                root.after(800, root.destroy)
            return
        if progress.url and mode["value"] == "working":
            _show_running()
        status_var.set(progress.status)
        if mode["value"] == "running":
            detail_var.set(f"Running at {progress.url}\nClosing this window stops NoCode-DL.")
        elif progress.typical > 0 and not progress.detail:
            detail_var.set(fmt_eta(time.time() - progress.phase_start, progress.typical))
        else:
            detail_var.set(progress.detail)
        root.after(200, _poll)

    root.after(50, _poll)
    root.mainloop()
    # After Quit, give the worker a moment to record the server's exit.
    thread.join(timeout=15 if progress.cancelled.is_set() else None)


class Tee:
    """Write to a log file and, when there is one, the original stream
    (``None`` under pythonw)."""

    def __init__(self, log_file, stream) -> None:
        self._log = log_file
        self._stream = stream

    def write(self, text: str) -> int:
        self._log.write(text)
        self._log.flush()
        if self._stream is not None:
            try:
                self._stream.write(text)
                self._stream.flush()
            except Exception:
                pass
        return len(text)

    def flush(self) -> None:
        self._log.flush()

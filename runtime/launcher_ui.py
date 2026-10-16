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

# While nothing is installing (sync, server start), the details line rotates
# through these every FUN_INTERVAL seconds. Real news always wins: a fresh
# detail from the bootstrap shows instead, and errors replace the line.
FUN_INTERVAL = 5.0
FUN_MESSAGES = (
    "Warming up the tensors…",
    "Teaching tensors to behave…",
    "Counting neurons (all of them)…",
    "Polishing the gradients…",
    "Convincing the GPU to join in…",
    "Stacking layers neatly…",
    "Brewing a fresh batch of weights…",
    "Lining up the learning rates…",
    "Untangling the attention heads…",
    "Almost there, stretching the epochs…",
)

# A first install pulls in about 110 distributions; pip only says how many
# once it has downloaded them all, so progress before that is measured
# against this estimate.
EXPECTED_PACKAGES = 100
# Below this much measured progress the rate is too noisy to extrapolate,
# so the time left counts down from the phase's typical duration instead.
MEASURED_ETA_MIN = 0.1

_WHEEL = re.compile(r"(?P<name>[A-Za-z0-9_.]+?)-(?P<version>\d[^-]*)-.*\.whl")
_SDIST = re.compile(r"(?P<name>[A-Za-z0-9_.-]+?)-(?P<version>\d[\w.]*)\.(?:tar\.gz|zip)")
_SIZE = re.compile(r"\(([\d.]+\s*[kMG]B)\)")


def _pretty_name(raw: str) -> str:
    """'torch==2.9.0' or 'numpy>=2' -> 'torch 2.9.0' / 'numpy'."""
    raw = raw.strip()
    match = re.match(r"([A-Za-z0-9_.\-]+)(?:\[[^\]]*\])?\s*(==\s*([^\s;,]+))?", raw)
    if not match:
        return raw.split()[0] if raw else raw
    return f"{match.group(1)} {match.group(3)}" if match.group(3) else match.group(1)


def parse_pip_line(line: str) -> tuple[str, str] | None:
    """Classify one line of install.py / pip output.

    Returns (kind, detail) where kind is one of venv, pip, collect, download,
    install, done; or None for lines that say nothing new.
    """
    line = line.strip()
    if line.startswith("Creating virtual environment"):
        return "venv", "Creating the Python environment"
    if line.startswith("Updating existing environment"):
        return "venv", "Checking the existing Python environment"
    if line.startswith("$") and "--upgrade pip" in line:
        return "pip", "Updating the package installer"
    if line.startswith("Collecting "):
        return "collect", f"Looking up {_pretty_name(line[len('Collecting '):])}"
    if line.startswith("Requirement already satisfied: "):
        return "collect", f"Already installed: {_pretty_name(line.split(': ', 1)[1])}"
    if line.startswith("Downloading "):
        rest = line[len("Downloading "):]
        target = rest.split()[0].rsplit("/", 1)[-1]
        match = _WHEEL.match(target) or _SDIST.match(target)
        name = f"{match.group('name')} {match.group('version')}" if match else target
        size = _SIZE.search(rest)
        return "download", f"Downloading {name}" + (f" ({size.group(1)})" if size else "")
    if line.startswith("Installing collected packages:"):
        pkgs = [p.strip() for p in line.split(":", 1)[1].split(",") if p.strip()]
        shown = ", ".join(pkgs[:3]) + (f" +{len(pkgs) - 3} more" if len(pkgs) > 3 else "")
        return "install", f"Installing {len(pkgs)} packages: {shown}"
    if line.startswith("Successfully installed"):
        count = len(line.split()[2:])
        return "done", f"Installed {count} packages"
    return None


def pip_detail(line: str) -> str | None:
    """The details-line text for one line of pip output, if it has any."""
    parsed = parse_pip_line(line)
    return parsed[1] if parsed else None


def fun_message(elapsed: float) -> str:
    """The light message to show ``elapsed`` seconds into a quiet phase."""
    return FUN_MESSAGES[int(max(elapsed, 0) // FUN_INTERVAL) % len(FUN_MESSAGES)]


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
    """State the worker thread writes and the window polls.

    The bar covers the whole launch: each phase owns a ``span`` of it
    (for example 0.03-0.85 for the install) and reports its own 0-1
    ``fraction``, from pip's output where there is one, otherwise from the
    time elapsed against the phase's typical duration.
    """

    def __init__(self) -> None:
        self.status = "Preparing…"
        self.detail = ""
        self.detail_time = 0.0
        self.phase_start = time.time()
        self.typical = -1.0
        self.span = (0.0, 0.0)
        self.fraction = 0.0
        self.fun = False
        self.packages_seen = 0
        self._shown = 0.0
        self.error: str | None = None
        self.url: str | None = None  # set while the app runs and the window stays up
        self.done = threading.Event()
        self.cancelled = threading.Event()
        self.window_shown = False

    def phase(self, status: str, typical_secs: float = -1.0, detail: str = "", *,
              span: tuple[float, float] | None = None, fun: bool = False) -> None:
        """Start a phase. ``fun`` rotates friendly messages while it has no news."""
        self.status = status
        self.phase_start = time.time()
        self.typical = typical_secs
        self.span = span if span is not None else (self.span[1], self.span[1])
        self.fraction = 0.0
        self.fun = fun
        self.set_detail(detail)
        print(f"[bootstrap] {status}" + (f" ({detail})" if detail else ""), flush=True)

    def set_detail(self, detail: str, now: float | None = None) -> None:
        self.detail = detail
        self.detail_time = (time.time() if now is None else now) if detail else 0.0

    def finish_phase(self) -> None:
        self.fraction = 1.0

    def pip_line(self, line: str, now: float | None = None) -> None:
        """Update the details line and the install fraction from pip's output."""
        parsed = parse_pip_line(line)
        if not parsed:
            return
        kind, detail = parsed
        self.set_detail(detail, now)
        if kind == "venv":
            self.fraction = max(self.fraction, 0.02)
        elif kind in ("collect", "download"):
            if kind == "collect":
                self.packages_seen += 1
            # Downloading is most of the time; leave the rest for installing.
            self.fraction = max(self.fraction, 0.85 * min(self.packages_seen / EXPECTED_PACKAGES, 1.0))
        elif kind == "install":
            self.fraction = max(self.fraction, 0.88)
        elif kind == "done":
            self.fraction = 1.0

    def overall(self, now: float | None = None) -> float:
        """The bar's position, 0-1. Never moves backwards."""
        now = time.time() if now is None else now
        fraction = self.fraction
        if self.typical > 0:
            # Keep moving during long steps (one big download) but never claim done.
            fraction = max(fraction, min(0.95, (now - self.phase_start) / self.typical))
        start, end = self.span
        self._shown = max(self._shown, start + (end - start) * min(fraction, 1.0))
        return self._shown

    def detail_text(self, now: float | None = None) -> str:
        """What the details line says: real news, else a rotating fun message."""
        now = time.time() if now is None else now
        if self.detail and not (self.fun and now - self.detail_time >= FUN_INTERVAL):
            return self.detail
        if self.fun:
            return fun_message(now - self.phase_start)
        return self.detail

    def eta_text(self, now: float | None = None) -> str:
        now = time.time() if now is None else now
        percent = f"{int(self.overall(now) * 100)}%"
        if self.typical <= 0:
            return percent
        elapsed = now - self.phase_start
        expected = self.typical
        if self.fraction >= MEASURED_ETA_MIN:
            # pip says how far along we are, so go by the actual rate rather
            # than the typical duration (which showed "~8m left" at 85%).
            expected = elapsed / self.fraction
        return f"{percent} · {fmt_eta(elapsed, expected)}"

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
    width, height = 480, 250
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
    bar = ttk.Progressbar(root, orient="horizontal", mode="determinate", length=380, maximum=1000)
    bar.pack(pady=(0, 4))
    eta_var = tk.StringVar(value="")
    eta_label = tk.Label(root, textvariable=eta_var, font=(font, 9), fg=sub, bg=bg)
    eta_label.pack(pady=(0, 8))
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
        bar.pack_forget()
        eta_label.pack_forget()
        import webbrowser

        tk.Button(buttons, text="Open in browser", width=16,
                  command=lambda: webbrowser.open(progress.url or "")).pack(side="left", padx=6)
        tk.Button(buttons, text="Quit NoCode-DL", width=16, command=_quit).pack(side="left", padx=6)

    def _show_error() -> None:
        mode["value"] = "error"
        for child in buttons.winfo_children():
            child.destroy()
        bar.pack_forget()
        eta_label.pack_forget()
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
                if not progress.error:
                    bar["value"] = 1000
                    eta_var.set("100%")
                root.after(800, root.destroy)
            return
        if progress.url and mode["value"] == "working":
            _show_running()
        status_var.set(progress.status)
        if mode["value"] == "running":
            detail_var.set(f"Running at {progress.url}\nClosing this window stops NoCode-DL.")
        else:
            detail_var.set(progress.detail_text())
            bar["value"] = int(progress.overall() * 1000)
            eta_var.set(progress.eta_text())
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

"""
Windows bootstrap — mirrors the macOS runtime bootstrap.

NoCode-DL.ps1 runs this with the python.org Python in a hidden console. It
shows a small progress window (runtime/launcher_ui.py) while it syncs the
app, installs dependencies and starts the server, sends pip and server
output to log files under %LOCALAPPDATA%\\NoCode-DL\\logs, and opens the
browser once the app answers. The window then stays up as the app's
control: closing it stops the server. The exit code is the app's.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

try:
    from runtime import launcher_ui as ui  # imported as a package (tests)
except ImportError:  # run as a script: runtime/ is sys.path[0]
    import launcher_ui as ui


APP_NAME = "NoCode-DL"
USER_ROOT = Path(os.environ.get("LOCALAPPDATA", Path.home())) / APP_NAME
RUNTIME_APP_ROOT = USER_ROOT / "app"
VENV_DIR = USER_ROOT / ".venv"
LOG_DIR = USER_ROOT / "logs"
LAUNCHER_LOG = LOG_DIR / "launcher.log"
SERVER_LOG = LOG_DIR / "server.log"
LOG_MAX_BYTES = 5_000_000
VERSION_FILE = ".bundle_version"
SYNC_EXCLUDES = {
    "__pycache__",
    ".venv",
    "outputs",
    "logs",
    # The install folder is both the bundle and the parent of the runtime
    # copy, so without this an upgrade copies the old app/ into app/app/.
    "app",
    "build",
    "dist",
}
# User data inside the runtime app folder that must survive an upgrade
# (trained models, checkpoints, exports, run history, project state).
PRESERVE_ON_UPGRADE = {"outputs"}
SERVER_PORT = 7860
INSTALL_TYPICAL_SECS = 600.0
SERVER_TYPICAL_SECS = 60.0
# First start on Windows is slow: Defender scans every freshly installed DLL.
SERVER_TIMEOUT = 600.0
# Children write to pipes and log files, never to a console of their own.
_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)


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


def _child_env() -> dict[str, str]:
    # Unbuffered UTF-8 so pip's lines reach the window as they happen and
    # package names with non-ASCII characters don't break the log.
    return dict(os.environ, PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8")


def _open_log(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file() and path.stat().st_size > LOG_MAX_BYTES:
        path.replace(path.with_suffix(".old.log"))
    log = open(path, "a", encoding="utf-8", errors="replace")
    log.write(f"\n--- {time.strftime('%Y-%m-%d %H:%M:%S')} ---\n")
    log.flush()
    return log


class _Bootstrap:
    """The work behind the progress window, run on its worker thread."""

    def __init__(self, progress: ui.Progress, smoke: bool) -> None:
        self.progress = progress
        self.smoke = smoke
        self.exit_code = 0
        self.procs: list[subprocess.Popen] = []

    def quit(self) -> None:
        for proc in self.procs:
            ui.kill_tree(proc)

    def _stream(self, cmd: list[str], *, cwd: Path) -> tuple[int, list[str]]:
        """Run ``cmd`` with its output going to the launcher log, showing
        pip's progress in the window. Returns the exit code and last lines."""
        print("$", subprocess.list2cmdline(cmd), flush=True)
        proc = subprocess.Popen(
            cmd, cwd=cwd, env=_child_env(),
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding="utf-8", errors="replace",
            creationflags=_NO_WINDOW,
        )
        self.procs.append(proc)
        recent: list[str] = []
        assert proc.stdout is not None
        for line in proc.stdout:
            line = line.rstrip()
            print(line, flush=True)
            if line:
                recent = (recent + [line])[-20:]
            self.progress.pip_line(line)
        code = proc.wait()
        self.procs.remove(proc)
        return code, recent

    def _bootstrap_without_venv(self, app_root: Path, venv_dir: Path) -> None:
        """Install into the launching Python when it cannot create a venv.

        Only used when install.py reports VENV_UNAVAILABLE_EXIT. pip comes from the
        interpreter's own bundled ensurepip rather than a downloaded get-pip.py.
        """
        from runtime.runtime_setup import pip_install_commands, save_install_state

        python = sys.executable
        check = subprocess.run([python, "-m", "pip", "--version"], capture_output=True,
                               creationflags=_NO_WINDOW)
        if check.returncode == 0:
            print("[bootstrap] pip already available", flush=True)
        else:
            print("[bootstrap] installing pip with ensurepip ...", flush=True)
            code, recent = self._stream([python, "-m", "ensurepip", "--upgrade"], cwd=app_root)
            if code != 0:
                raise RuntimeError(_install_failure(code, recent))

        for cmd in pip_install_commands(python):
            code, recent = self._stream(cmd, cwd=app_root)
            if code != 0:
                raise RuntimeError(_install_failure(code, recent))

        venv_dir.mkdir(parents=True, exist_ok=True)
        save_install_state(venv_dir, python_path=Path(python))

    def run(self) -> None:
        progress = self.progress
        source_root = _bundled_root()
        progress.phase("Preparing NoCode-DL…", detail="Copying app files", span=(0.0, 0.03), fun=True)
        print(f"[bootstrap] syncing bundled app from {source_root}", flush=True)
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
            progress.phase("Installing dependencies…", INSTALL_TYPICAL_SECS,
                           "First launch — this downloads about 2 GB and takes a few minutes",
                           span=(0.03, 0.85))
            code, recent = self._stream(
                [sys.executable, str(install_script), "--venv-dir", str(VENV_DIR)], cwd=RUNTIME_APP_ROOT,
            )
            if progress.cancelled.is_set():
                return
            if code == VENV_UNAVAILABLE_EXIT:
                # This Python can't make virtual environments at all.
                print("[bootstrap] virtual environments unavailable; installing into this Python", flush=True)
                self._bootstrap_without_venv(RUNTIME_APP_ROOT, VENV_DIR)
            elif code != 0:
                # A pip/network failure: retrying the same ~2 GB install elsewhere
                # would only hide the real error.
                raise RuntimeError(_install_failure(code, recent))

        state = load_install_state(VENV_DIR) or {}
        python = Path(state.get("venv_python") or VENV_DIR / "Scripts" / "python.exe")
        port = find_free_port(SERVER_PORT)
        url = f"http://127.0.0.1:{port}"
        cmd = [str(python), str(run_script), "--venv-dir", str(VENV_DIR),
               "--server-port", str(port), "--no-browser"]
        if self.smoke:
            cmd.append("--smoke-test")

        progress.phase("Starting NoCode-DL…", SERVER_TYPICAL_SECS, "Starting the app's server",
                       span=(progress.span[1], 1.0), fun=True)
        print(f"[bootstrap] launching app with {python} on port {port}; output in {SERVER_LOG}", flush=True)
        # The server's output goes to a file, not a pipe: nobody reads a pipe
        # once the app runs, and a full pipe blocks the server.
        with _open_log(SERVER_LOG) as server_log:
            proc = subprocess.Popen(cmd, cwd=RUNTIME_APP_ROOT, env=_child_env(),
                                    stdout=server_log, stderr=subprocess.STDOUT,
                                    creationflags=_NO_WINDOW)
        self.procs.append(proc)
        progress.set_detail("Loading PyTorch and the app; waiting for it to answer")

        if self.smoke:
            # run_local --smoke-test starts the server, checks it answers
            # and shuts it down; its exit code is the result.
            self.exit_code = proc.wait()
            _copy_tail(SERVER_LOG)
            if self.exit_code != 0:
                raise RuntimeError(f"The app's smoke test failed (exit code {self.exit_code}).")
            return

        if not ui.wait_for_server(port, SERVER_TIMEOUT, proc):
            if progress.cancelled.is_set():
                return
            ui.kill_tree(proc)
            self.exit_code = proc.returncode or 1
            _copy_tail(SERVER_LOG)
            raise RuntimeError(
                f"NoCode-DL did not start (exit code {self.exit_code}). The server's messages are in {SERVER_LOG}."
            )

        progress.finish_phase()
        print(f"[bootstrap] app is up at {url}", flush=True)
        import webbrowser

        webbrowser.open(url)
        progress.phase("NoCode-DL is running")
        progress.url = url

        code = proc.wait()
        if progress.cancelled.is_set():
            print("[bootstrap] stopped from the NoCode-DL window", flush=True)
            return
        self.exit_code = code
        if code != 0:
            _copy_tail(SERVER_LOG)
            raise RuntimeError(f"NoCode-DL stopped unexpectedly (exit code {code}).")


def _install_failure(code: int, recent: list[str]) -> str:
    reason = next((l for l in reversed(recent) if "error" in l.lower()), recent[-1] if recent else "")
    return (
        f"Installing dependencies failed (exit code {code}). {reason[:200]}\n"
        "Check your internet connection or proxy, then open NoCode-DL again."
    )


def _copy_tail(path: Path, lines: int = 40) -> None:
    """Repeat the end of the server log in the launcher log (and CI output)."""
    try:
        tail = path.read_text(encoding="utf-8", errors="replace").splitlines()[-lines:]
    except OSError:
        return
    print(f"[bootstrap] last lines of {path}:", flush=True)
    for line in tail:
        print(f"    {line}", flush=True)


def main() -> int:
    USER_ROOT.mkdir(parents=True, exist_ok=True)
    smoke = ui.smoke_mode()
    progress = ui.Progress()
    boot = _Bootstrap(progress, smoke)
    ui.run_window(progress, boot.run, log_path=LOG_DIR, on_quit=boot.quit, interactive=not smoke)
    if progress.cancelled.is_set():
        return 0
    if progress.error:
        print(f"[bootstrap] ERROR: {progress.error}", flush=True)
        return boot.exit_code or 1
    return boot.exit_code


if __name__ == "__main__":
    # Everything this process and the installer print goes to launcher.log;
    # the console (if any) is hidden, and pythonw has none at all.
    _log = _open_log(LAUNCHER_LOG)
    sys.stdout = ui.Tee(_log, sys.stdout)
    sys.stderr = ui.Tee(_log, sys.stderr)
    try:
        _code = main()
    except Exception:
        import traceback

        traceback.print_exc()
        ui.show_error_box(APP_NAME, f"NoCode-DL failed to start.\n\nDetails: {LAUNCHER_LOG}")
        _code = 1
    if _code:
        print(f"[bootstrap] exiting with code {_code}", flush=True)
    raise SystemExit(_code)

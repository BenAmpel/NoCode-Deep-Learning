from __future__ import annotations

import collections
import re
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

try:
    from runtime import launcher_ui as ui  # imported as a package (tests)
except ImportError:  # run as a script: runtime/ is sys.path[0]
    import launcher_ui as ui


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
# User data that lives inside the runtime app folder and must survive an
# upgrade. The app writes trained models, checkpoints, exports, run history
# and project state under app/outputs.
PRESERVE_ON_UPGRADE = {"outputs"}
LOG_DIR = Path.home() / "Library" / "Logs" / APP_NAME
SERVER_LOG = LOG_DIR / "server.log"
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


def main() -> int:
    source_root = _bundled_root()
    USER_ROOT.mkdir(parents=True, exist_ok=True)
    smoke = ui.smoke_mode()
    progress = ui.Progress()
    result = {"code": 0}

    def bootstrap() -> None:
        progress.phase("Preparing NoCode-DL\u2026", detail="Copying app files", span=(0.0, 0.03), fun=True)
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
            progress.phase("Installing dependencies\u2026", typical_secs=300.0,
                           detail="First launch \u2014 this downloads about 2 GB and takes a few minutes",
                           span=(0.03, 0.85))
            proc = subprocess.Popen(
                [sys.executable, str(install_script), "--venv-dir", str(VENV_DIR)],
                cwd=RUNTIME_APP_ROOT,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
            recent: collections.deque[str] = collections.deque(maxlen=20)
            for line in proc.stdout:
                line = line.rstrip()
                # Our stdout is launcher.log (see the .app launcher), so pip's
                # full output, including any error, ends up there.
                print(line, flush=True)
                if line:
                    recent.append(line)
                progress.pip_line(line)
            proc.wait()
            if proc.returncode != 0:
                reason = next((l for l in reversed(recent) if "error" in l.lower()),
                              recent[-1] if recent else "")
                raise RuntimeError(
                    f"Installing dependencies failed (exit code {proc.returncode}). {reason[:200]}"
                )

        venv_python = VENV_DIR / "bin" / "python"
        if not venv_python.is_file():
            raise RuntimeError(f"Expected virtualenv Python at {venv_python}")

        from runtime.runtime_setup import find_free_port

        port = find_free_port()
        progress.phase("Starting NoCode-DL\u2026", typical_secs=SERVER_TYPICAL_SECS,
                       detail="Loading PyTorch and the app; waiting for it to answer",
                       span=(progress.span[1], 1.0), fun=True)
        cmd = [
            str(venv_python),
            str(run_script),
            "--venv-dir", str(VENV_DIR),
            "--server-port", str(port),
            "--no-browser",
        ]
        if smoke:
            cmd.append("--smoke-test")
        # The server outlives this launcher, so its output goes to a log
        # file rather than a pipe: once nobody reads a pipe it fills up
        # and the server blocks (or dies with BrokenPipe) on its next print.
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        log_start = SERVER_LOG.stat().st_size if SERVER_LOG.exists() else 0
        with open(SERVER_LOG, "a", encoding="utf-8") as server_log:
            server_log.write(f"\n--- launch {time.strftime('%Y-%m-%d %H:%M:%S')} on port {port} ---\n")
            server_log.flush()
            server = subprocess.Popen(
                cmd,
                cwd=RUNTIME_APP_ROOT,
                stdout=server_log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )

        if smoke:
            # run_local --smoke-test starts the server, checks it answers
            # and shuts it down; its exit code is the result.
            result["code"] = server.wait()
            print(SERVER_LOG.read_text(encoding="utf-8", errors="replace")[log_start:], flush=True)
            if result["code"] != 0:
                raise RuntimeError(f"The app's smoke test failed (exit code {result['code']}).")
            progress.phase("Done")
            return

        server_ready = threading.Event()

        def _follow_server_log() -> None:
            skip = re.compile(r"^\s*$|uvicorn|httpx|HTTP Request|^---")
            with open(SERVER_LOG, encoding="utf-8", errors="replace") as fh:
                fh.seek(log_start)
                while not server_ready.is_set():
                    line = fh.readline()
                    if not line:
                        time.sleep(0.2)
                        continue
                    line = line.rstrip()
                    if line and not skip.search(line):
                        clean = re.sub(r"^\S+\s+\S+\s+\S+\s+", "", line).strip()
                        if clean:
                            progress.set_detail(clean[:80])

        threading.Thread(target=_follow_server_log, daemon=True).start()

        ready = ui.wait_for_server(port, POLL_TIMEOUT, server)
        server_ready.set()

        if not ready:
            raise RuntimeError(f"The server did not start in time. Check {SERVER_LOG} for details.")
        progress.finish_phase()
        progress.phase("Opening browser\u2026")
        import webbrowser
        webbrowser.open(f"http://127.0.0.1:{port}")
        progress.phase("Done")

    ui.run_window(progress, bootstrap, log_path=LOG_DIR, interactive=not smoke)

    if progress.error:
        print(f"[bootstrap] ERROR: {progress.error}", flush=True)
        return result["code"] or 1
    return result["code"]


if __name__ == "__main__":
    raise SystemExit(main())

import importlib.util
import sys
from pathlib import Path

import pytest

from runtime import bootstrap_macos, bootstrap_windows, launcher_ui

BOOTSTRAPS = [bootstrap_macos, bootstrap_windows]


def _write(path: Path, text: str = "x") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def _make_bundle(root: Path, version: str) -> Path:
    _write(root / ".bundle_version", version)
    _write(root / "app.py", f"# app {version}")
    _write(root / "runtime" / "runtime_setup.py", "# setup")
    _write(root / "outputs" / "should_not_ship.txt")  # excluded from the sync
    return root


@pytest.mark.parametrize("bootstrap", BOOTSTRAPS, ids=["macos", "windows"])
def test_upgrade_keeps_user_outputs(tmp_path, bootstrap):
    target = tmp_path / "runtime_app"
    _write(target / ".bundle_version", "1.0.0")
    _write(target / "removed_in_new_version.py")
    model = _write(target / "outputs" / "my_run" / "model.pt", "weights")

    bootstrap._sync_bundle_to_runtime(_make_bundle(tmp_path / "bundle", "1.0.1"), target)

    assert model.read_text(encoding="utf-8") == "weights"
    assert (target / ".bundle_version").read_text(encoding="utf-8") == "1.0.1"
    assert (target / "app.py").read_text(encoding="utf-8") == "# app 1.0.1"
    assert not (target / "removed_in_new_version.py").exists()
    assert not (target / "outputs" / "should_not_ship.txt").exists()


@pytest.mark.parametrize("bootstrap", BOOTSTRAPS, ids=["macos", "windows"])
def test_first_install_copies_bundle(tmp_path, bootstrap):
    target = tmp_path / "runtime_app"
    bootstrap._sync_bundle_to_runtime(_make_bundle(tmp_path / "bundle", "1.0.0"), target)
    assert (target / "runtime" / "runtime_setup.py").is_file()
    assert not (target / "outputs").exists()


@pytest.mark.parametrize("bootstrap", BOOTSTRAPS, ids=["macos", "windows"])
def test_same_version_is_left_alone(tmp_path, bootstrap):
    target = tmp_path / "runtime_app"
    _write(target / ".bundle_version", "1.0.0")
    local_edit = _write(target / "app.py", "# locally modified")
    bootstrap._sync_bundle_to_runtime(_make_bundle(tmp_path / "bundle", "1.0.0"), target)
    assert local_edit.read_text(encoding="utf-8") == "# locally modified"


def test_launcher_window_imports_ttk_submodule():
    # `import tkinter as tk` does not load tkinter.ttk, so `tk.ttk.X` raises
    # AttributeError and the launcher died before showing its window.
    source = Path(launcher_ui.__file__).read_text(encoding="utf-8")
    assert "tk.ttk." not in source
    assert "from tkinter import ttk" in source


def test_windows_upgrade_does_not_copy_runtime_into_itself(tmp_path):
    # On Windows the install folder is the bundle and also holds app/.
    bundle = _make_bundle(tmp_path / "NoCode-DL", "1.0.1")
    target = bundle / "app"
    _write(target / ".bundle_version", "1.0.0")
    model = _write(target / "outputs" / "model.pt", "weights")
    bootstrap_windows._sync_bundle_to_runtime(bundle, target)
    assert not (target / "app").exists()
    assert model.read_text(encoding="utf-8") == "weights"


@pytest.mark.parametrize("line, expected", [
    ("Collecting torch==2.9.0", "Installing: torch"),
    ("Downloading numpy-2.4.0-cp312-cp312-win_amd64.whl (12.6 MB)", "Installing: numpy-2.4.0-cp312-cp312-win_amd64.whl"),
    ("Installing collected packages: a, b, c, d, e", "Finalizing: a, b, c +2 more"),
    ("Successfully installed a-1 b-2", "Installed: a-1, b-2"),
    ("Requirement already satisfied: pip", None),
])
def test_pip_detail(line, expected):
    assert launcher_ui.pip_detail(line) == expected


@pytest.mark.parametrize("value, expected", [("", False), ("0", False), ("1", True), ("yes", True)])
def test_smoke_mode_env(monkeypatch, value, expected):
    monkeypatch.setenv(launcher_ui.SMOKE_ENV, value)
    assert launcher_ui.smoke_mode() is expected


def test_headless_window_reports_worker_error(monkeypatch, tmp_path):
    # Without Tk the work still runs, and a failure is recorded, not raised.
    monkeypatch.setitem(sys.modules, "tkinter", None)
    progress = launcher_ui.Progress()

    def worker():
        raise RuntimeError("pip failed")

    launcher_ui.run_window(progress, worker, log_path=tmp_path, interactive=False)
    assert progress.done.is_set()
    assert progress.error == "pip failed"
    assert not progress.window_shown


def test_windows_launcher_script(tmp_path):
    # By path: the PyPI `packaging` library shadows the packaging/ folder.
    path = Path(__file__).resolve().parent.parent / "packaging" / "build_windows_installer.py"
    spec = importlib.util.spec_from_file_location("build_windows_installer", path)
    builder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(builder)

    script = builder.launcher_ps1()
    script.encode("ascii")  # PowerShell 5.1 reads BOM-less files as ANSI
    assert "\r\n" in script and "\n" not in script.replace("\r\n", "")
    for sha in builder.PYTHON_INSTALLER_SHA256.values():
        assert sha in script
    assert "Get-FileHash -Algorithm SHA256" in script
    assert "NOCODE_DL_SMOKE_TEST" in script
    assert "exit $LASTEXITCODE" in script
    # Hidden window: nothing may wait on console input.
    assert "Read-Host \"" not in script
    assert "-WindowStyle Hidden" in builder.SHORTCUT_PARAMETERS

    builder.stage_files(tmp_path / "staged", "1.2.3")
    bat = (tmp_path / "staged" / "NoCode-DL.bat").read_text(encoding="utf-8")
    assert "exit /b %ERRORLEVEL%" in bat

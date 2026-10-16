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
    ("Creating virtual environment at C:\\x\\.venv ...", "Creating the Python environment"),
    ("$ C:\\x\\python.exe -m pip install --upgrade pip --retries 5", "Updating the package installer"),
    ("Collecting torch==2.9.0 (from -r requirements.txt (line 1))", "Looking up torch 2.9.0"),
    ("Collecting gradio[oauth]==6.3.0", "Looking up gradio 6.3.0"),
    ("Requirement already satisfied: numpy==2.4.0 in c:\\x", "Already installed: numpy 2.4.0"),
    ("Downloading torch-2.9.0-cp312-cp312-win_amd64.whl (110.9 MB)", "Downloading torch 2.9.0 (110.9 MB)"),
    ("Downloading https://files.pythonhosted.org/p/numpy-2.4.0-cp312-cp312-macosx_14_0_arm64.whl (5.2 MB)",
     "Downloading numpy 2.4.0 (5.2 MB)"),
    ("Downloading antlr4-python3-runtime-4.9.3.tar.gz (117 kB)", "Downloading antlr4-python3-runtime 4.9.3 (117 kB)"),
    ("Installing collected packages: a, b, c, d, e", "Installing 5 packages: a, b, c +2 more"),
    ("Successfully installed a-1 b-2", "Installed 2 packages"),
    ("  Using cached torch-2.9.0.whl", None),
    ("", None),
])
def test_pip_detail(line, expected):
    assert launcher_ui.pip_detail(line) == expected


def test_install_progress_follows_pip():
    progress = launcher_ui.Progress()
    progress.phase("Installing", span=(0.1, 0.9))
    assert progress.overall() == pytest.approx(0.1)
    progress.pip_line("Creating virtual environment at x ...")
    for i in range(launcher_ui.EXPECTED_PACKAGES // 2):
        progress.pip_line(f"Collecting pkg{i}==1.0")
    halfway = progress.overall()
    assert 0.1 + 0.8 * 0.4 < halfway < 0.1 + 0.8 * 0.5  # half the packages, 85% weight
    progress.pip_line("Downloading pkg0-1.0-py3-none-any.whl (1.0 MB)")
    assert progress.detail_text() == "Downloading pkg0 1.0 (1.0 MB)"
    progress.pip_line("Installing collected packages: a, b")
    assert progress.overall() == pytest.approx(0.1 + 0.8 * 0.88)
    progress.pip_line("Successfully installed a-1 b-2")
    assert progress.overall() == pytest.approx(0.9)


def test_bar_uses_time_but_never_goes_back_or_claims_done():
    progress = launcher_ui.Progress()
    progress.phase("Starting", typical_secs=40, span=(0.5, 1.0))
    start = progress.phase_start
    assert progress.overall(start + 20) == pytest.approx(0.75)
    assert progress.overall(start + 400) == pytest.approx(0.5 + 0.5 * 0.95)
    assert progress.overall(start + 1) == pytest.approx(0.5 + 0.5 * 0.95)  # never backwards
    progress.finish_phase()
    assert progress.overall(start + 401) == pytest.approx(1.0)
    assert progress.eta_text(start + 401).startswith("100%")


def test_fun_messages_rotate_every_five_seconds():
    assert launcher_ui.FUN_INTERVAL == 5
    seen = [launcher_ui.fun_message(t) for t in (0, 4.9, 5, 9.9, 10)]
    assert seen[0] == seen[1] != seen[2] == seen[3] != seen[4]
    n = len(launcher_ui.FUN_MESSAGES)
    assert launcher_ui.fun_message(5 * n) == launcher_ui.fun_message(0)  # wraps around
    assert all(len(m) <= 40 for m in launcher_ui.FUN_MESSAGES)


def test_real_news_beats_fun_messages():
    progress = launcher_ui.Progress()
    progress.phase("Starting", detail="Starting the app's server", fun=True)
    t0 = progress.phase_start
    progress.detail_time = t0
    assert progress.detail_text(t0 + 1) == "Starting the app's server"
    assert progress.detail_text(t0 + 6) == launcher_ui.fun_message(6)
    progress.set_detail("Loading models", now=t0 + 12)
    assert progress.detail_text(t0 + 13) == "Loading models"
    assert progress.detail_text(t0 + 18) == launcher_ui.fun_message(18)


def test_installing_never_shows_fun_messages():
    progress = launcher_ui.Progress()
    progress.phase("Installing", detail="First launch", span=(0.03, 0.85))
    assert progress.detail_text(progress.phase_start + 60) == "First launch"


def test_error_replaces_status_and_details():
    progress = launcher_ui.Progress()
    progress.phase("Starting", detail="x", fun=True)
    progress.fail("Port in use")
    assert progress.error == "Port in use"
    assert progress.status == "NoCode-DL could not start"


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


def test_time_left_follows_measured_progress():
    # Windows CI showed "85% · ~8m 33s remaining": the estimate counted down
    # from the 10-minute typical install while pip was nearly done.
    progress = launcher_ui.Progress()
    progress.phase("Installing", typical_secs=600, span=(0.03, 0.85))
    start = progress.phase_start
    assert progress.eta_text(start + 20).endswith("~9m 40s remaining")  # nothing measured yet
    for i in range(launcher_ui.EXPECTED_PACKAGES):
        progress.pip_line(f"Collecting pkg{i}==1.0")
    progress.pip_line("Installing collected packages: a, b")  # 88% of the install
    assert progress.eta_text(start + 90).endswith("~12s remaining")  # 90 s * 0.12 / 0.88
    progress.pip_line("Successfully installed a-1 b-2")
    assert progress.eta_text(start + 95).endswith("almost ready\u2026")

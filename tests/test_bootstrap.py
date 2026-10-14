from pathlib import Path

import pytest

from runtime import bootstrap_macos, bootstrap_windows

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


def test_macos_launcher_imports_ttk_submodule():
    # `import tkinter as tk` does not load tkinter.ttk, so `tk.ttk.X` raises
    # AttributeError and the launcher died before showing its window.
    source = Path(bootstrap_macos.__file__).read_text(encoding="utf-8")
    assert "tk.ttk." not in source
    assert "from tkinter import ttk" in source

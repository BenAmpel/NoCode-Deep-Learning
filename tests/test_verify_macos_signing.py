import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(sys.platform != "darwin", reason="needs codesign and pkgutil")

PACKAGING = Path(__file__).resolve().parent.parent / "packaging"


def _load(name):
    # By path: the PyPI `packaging` library shadows this folder as a package.
    spec = importlib.util.spec_from_file_location(name, PACKAGING / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _adhoc_pkg(tmp_path) -> Path:
    app = tmp_path / "root" / "Applications" / "Dummy.app" / "Contents"
    (app / "MacOS").mkdir(parents=True)
    (app / "MacOS" / "Dummy").write_text("#!/bin/bash\necho hi\n")
    (app / "MacOS" / "Dummy").chmod(0o755)
    (app / "Info.plist").write_text(
        '<?xml version="1.0"?><plist version="1.0"><dict>'
        "<key>CFBundleExecutable</key><string>Dummy</string>"
        "<key>CFBundleIdentifier</key><string>com.test.dummy</string></dict></plist>")
    subprocess.run(["codesign", "--force", "--sign", "-", str(app.parent)], check=True, capture_output=True)
    pkg = tmp_path / "dummy.pkg"
    subprocess.run(["pkgbuild", "--root", str(tmp_path / "root"), "--identifier", "com.test.dummy",
                    "--version", "1", "--install-location", "/", str(pkg)], check=True, capture_output=True)
    return pkg


def test_adhoc_unsigned_installer_lists_every_notarization_blocker(tmp_path):
    problems = _load("verify_macos_signing").check_pkg(_adhoc_pkg(tmp_path), developer_id=True)
    text = "\n".join(problems)
    for expected in ("installer is not signed", "ad-hoc signature",
                     "hardened runtime is off", "no secure timestamp"):
        assert expected in text, text


def test_build_script_can_import_the_checker():
    # build_macos_installer.py imports it as a top-level module (its folder is
    # sys.path[0] when run as a script); `packaging.` would hit the PyPI library.
    source = (PACKAGING / "build_macos_installer.py").read_text()
    assert "from verify_macos_signing import check_pkg" in source

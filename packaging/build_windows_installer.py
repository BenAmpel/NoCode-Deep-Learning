"""
Build a Windows installer for NoCode-DL.

Generates an Inno Setup .iss script and a staging directory.  If the Inno
Setup compiler (iscc.exe) is on PATH, the script is compiled into a
NoCode-DL-Setup-{version}.exe installer automatically.

Usage
-----
    python packaging/build_windows_installer.py --version 1.0.0

Output
------
    dist/NoCode-DL-Setup-{version}.iss   — Inno Setup script
    dist/NoCode-DL-Setup-{version}.exe   — compiled installer (if iscc available)
"""
from __future__ import annotations

import argparse
import os
import shutil
import struct
import subprocess
import zlib
from datetime import datetime
from pathlib import Path


APP_NAME = "NoCode-DL"
ROOT = Path(__file__).resolve().parent.parent
DIST_ROOT = ROOT / "dist"
VENDOR_DIR = ROOT / "packaging" / "vendor"

PYTHON_VERSION = "3.12.10"
PYTHON_INSTALLER_URL = f"https://www.python.org/ftp/python/{PYTHON_VERSION}/python-{PYTHON_VERSION}-{{arch}}.exe"

RUNTIME_ITEMS = [
    "SETUP.md",
    "app.py",
    "config.py",
    "data_pipeline",
    "detection",
    "eval",
    "export",
    "install.py",
    "modalities",
    "models",
    "requirements.txt",
    "run_local.py",
    "runtime",
    "training",
    "ui",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a Windows installer for NoCode-DL.")
    parser.add_argument("--version", help="Installer version string. Defaults to timestamp.")
    parser.add_argument(
        "--skip-compile",
        action="store_true",
        help="Generate the .iss script only; do not attempt to run iscc.",
    )
    return parser.parse_args()


def version_string(raw: str | None) -> str:
    return raw if raw else datetime.now().strftime("%Y.%m.%d.%H%M")




# ── Icon generation ──────────────────────────────────────────────────────────

def _generate_icon_ico(path: Path) -> None:
    """Generate a simple 48x48 .ico file with the NoCode-DL brand colours."""
    size = 48
    pixels = bytearray()
    cx, cy = size // 2, size // 2
    r_outer = size // 2 - 2

    for y in range(size):
        for x in range(size):
            dx, dy = x - cx, y - cy
            dist = (dx * dx + dy * dy) ** 0.5
            if dist <= r_outer:
                t = dist / r_outer
                # Gradient: emerald centre → teal edge
                r = int(22 + (16 - 22) * t)
                g = int(163 + (133 - 163) * t)
                b = int(138 + (111 - 138) * t)
                a = 255
            else:
                r = g = b = a = 0
            pixels.extend([b, g, r, a])  # BGRA

    # ICO format: header + entry + raw BGRA + AND mask
    and_mask = b"\x00" * (size * ((size + 31) // 32 * 4))
    bmp_size = 40 + len(pixels) + len(and_mask)
    ico_header = struct.pack("<HHH", 0, 1, 1)  # reserved, type=ICO, count=1
    ico_entry = struct.pack(
        "<BBBBHHII",
        size, size, 0, 0, 1, 32, bmp_size, 22,  # 22 = 6 (header) + 16 (entry)
    )
    bmp_header = struct.pack(
        "<IiiHHIIiiII",
        40, size, size * 2, 1, 32, 0, len(pixels) + len(and_mask), 0, 0, 0, 0,
    )
    with open(path, "wb") as f:
        f.write(ico_header + ico_entry + bmp_header + pixels + and_mask)


# ── Staging ──────────────────────────────────────────────────────────────────

def stage_files(staging: Path, version: str) -> None:
    """Copy runtime items into the staging directory."""
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)

    # Copy source items
    for item in RUNTIME_ITEMS:
        src = ROOT / item
        dst = staging / item
        if src.is_dir():
            shutil.copytree(src, dst, ignore=shutil.ignore_patterns(
                "__pycache__", "*.pyc", ".DS_Store"))
        elif src.is_file():
            shutil.copy2(src, dst)
        else:
            print(f"  WARNING: {item} not found, skipping")

    # Write bundle version
    (staging / ".bundle_version").write_text(version, encoding="utf-8")

    # PowerShell launcher: finds/installs Python then runs runtime/bootstrap_windows.py.
    # Architecture-aware: downloads amd64 or arm64 installer from python.org.
    amd64_url = PYTHON_INSTALLER_URL.format(arch="amd64")
    arm64_url  = PYTHON_INSTALLER_URL.format(arch="arm64")
    ps1 = staging / f"{APP_NAME}.ps1"
    ps1.write_text(
        f'$ErrorActionPreference = "Stop"\r\n'
        f'try {{ $host.UI.RawUI.WindowTitle = "{APP_NAME}" }} catch {{}}\r\n'
        f'Set-Location $PSScriptRoot\r\n'
        f'\r\n'
        f'function Find-Python {{\r\n'
        f'    $candidates = @(\r\n'
        f'        "$env:LOCALAPPDATA\\Programs\\Python\\Python312\\python.exe",\r\n'
        f'        "$env:LOCALAPPDATA\\Programs\\Python\\Python311\\python.exe",\r\n'
        f'        "$env:LOCALAPPDATA\\Programs\\Python\\Python310\\python.exe",\r\n'
        f'        "C:\\Python312\\python.exe",\r\n'
        f'        "$env:ProgramFiles\\Python312\\python.exe"\r\n'
        f'    )\r\n'
        f'    foreach ($c in $candidates) {{\r\n'
        f'        if (Test-Path $c) {{\r\n'
        f'            $ok = & $c -c "import sys; exit(0 if sys.version_info>=(3,10) else 1)" 2>$null\r\n'
        f'            if ($LASTEXITCODE -eq 0) {{ return $c }}\r\n'
        f'        }}\r\n'
        f'    }}\r\n'
        f'    try {{\r\n'
        f'        $cmd = (Get-Command python -ErrorAction Stop).Source\r\n'
        f'        $ok = & $cmd -c "import sys; exit(0 if sys.version_info>=(3,10) else 1)" 2>$null\r\n'
        f'        if ($LASTEXITCODE -eq 0) {{ return $cmd }}\r\n'
        f'    }} catch {{}}\r\n'
        f'    return $null\r\n'
        f'}}\r\n'
        f'\r\n'
        f'$python = Find-Python\r\n'
        f'\r\n'
        f'if (-not $python) {{\r\n'
        f'    # Detect true architecture.\r\n'
        f'    # PROCESSOR_ARCHITECTURE can report ARM64 inside x64 VMs on Apple Silicon\r\n'
        f'    # (e.g. UTM/Parallels running x64 Windows). We verify by actually trying\r\n'
        f'    # to run an ARM64 binary — if it fails we fall back to amd64.\r\n'
        f'    $isArm64 = $false\r\n'
        f'    if ($env:PROCESSOR_ARCHITECTURE -eq "ARM64") {{\r\n'
        f'        # Quick probe: does an arm64 python installer even launch here?\r\n'
        f'        try {{\r\n'
        f'            $probe = Start-Process -FilePath "cmd.exe" -ArgumentList "/c exit 0" -PassThru -Wait -ErrorAction Stop\r\n'
        f'            # Check via WMI whether the OS is truly ARM64\r\n'
        f'            $cpu = (Get-WmiObject Win32_Processor -ErrorAction SilentlyContinue | Select-Object -First 1).Architecture\r\n'
        f'            # Architecture 12 = ARM64; anything else (9=x64, 0=x86) means emulated\r\n'
        f'            $isArm64 = ($cpu -eq 12)\r\n'
        f'        }} catch {{ $isArm64 = $false }}\r\n'
        f'    }}\r\n'
        f'    $arch = if ($isArm64) {{ "arm64" }} else {{ "amd64" }}\r\n'
        f'    $url  = if ($isArm64) {{ "{arm64_url}" }} else {{ "{amd64_url}" }}\r\n'
        f'    Write-Host "Python 3.10+ not found. Downloading Python {PYTHON_VERSION} ($arch)..."\r\n'
        f'    $installer = "$env:TEMP\\python-{PYTHON_VERSION}-installer.exe"\r\n'
        f'    (New-Object System.Net.WebClient).DownloadFile($url, $installer)\r\n'
        f'    Write-Host "Installing Python (this may take a minute)..."\r\n'
        f'    Start-Process $installer -ArgumentList "/quiet InstallAllUsers=0 PrependPath=1 Include_test=0 Include_launcher=0" -Wait\r\n'
        f'    Remove-Item $installer -Force -ErrorAction SilentlyContinue\r\n'
        f'    $python = "$env:LOCALAPPDATA\\Programs\\Python\\Python312\\python.exe"\r\n'
        f'}}\r\n'
        f'\r\n'
        f'if (-not (Test-Path $python)) {{\r\n'
        f'    Write-Host ""\r\n'
        f'    Write-Host "ERROR: Python could not be found or installed."\r\n'
        f'    Write-Host "Please install Python 3.10+ from https://www.python.org and re-launch."\r\n'
        f'    Read-Host "Press Enter to exit"\r\n'
        f'    exit 1\r\n'
        f'}}\r\n'
        f'\r\n'
        f'Write-Host "[{APP_NAME}] Using Python: $python"\r\n'
        f'& $python runtime\\bootstrap_windows.py\r\n'
        f'$code = $LASTEXITCODE\r\n'
        f'if ($code -ne 0) {{\r\n'
        f'    Write-Host ""\r\n'
        f'    Write-Host "{APP_NAME} exited with code $code"\r\n'
        f'    Read-Host "Press Enter to close"\r\n'
        f'}}\r\n'
        f'exit $code\r\n',
        encoding="utf-8",
    )

    # Thin bat wrapper — keeps the shortcut simple and avoids PowerShell title-bar flash
    bat = staging / f"{APP_NAME}.bat"
    bat.write_text(
        f"@echo off\r\n"
        f'powershell.exe -ExecutionPolicy Bypass -File "%~dp0{APP_NAME}.ps1"\r\n',
        encoding="utf-8",
    )

    print(f"  Staged {len(RUNTIME_ITEMS)} items (no bundled Python — downloaded on first run)")


# ── Inno Setup script generation ─────────────────────────────────────────────

def generate_iss(staging: Path, version: str, icon_path: Path) -> Path:
    """Generate an Inno Setup .iss script and return its path."""
    DIST_ROOT.mkdir(parents=True, exist_ok=True)
    iss_path = DIST_ROOT / f"{APP_NAME}-Setup-{version}.iss"

    iss_content = f"""; Inno Setup script for {APP_NAME}
; Generated by build_windows_installer.py

[Setup]
AppName={APP_NAME}
AppVersion={version}
AppPublisher=NoCode Deep Learning Project
DefaultDirName={{localappdata}}\\{APP_NAME}
DefaultGroupName={APP_NAME}
OutputDir=Output
OutputBaseFilename={APP_NAME}-Setup-{version}
Compression=lzma2/ultra64
SolidCompression=yes
UninstallDisplayIcon={{app}}\\icon.ico
PrivilegesRequired=lowest
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
WizardStyle=modern
DisableProgramGroupPage=yes

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Files]
Source: "NoCode-DL-staged\\*"; DestDir: "{{app}}"; Flags: recursesubdirs createallsubdirs
Source: "NoCode-DL-staged\\.bundle_version"; DestDir: "{{app}}"

[Icons]
Name: "{{group}}\\{APP_NAME}"; Filename: "{{app}}\\{APP_NAME}.bat"; IconFilename: "{{app}}\\icon.ico"; Comment: "Launch {APP_NAME}"
Name: "{{commondesktop}}\\{APP_NAME}"; Filename: "{{app}}\\{APP_NAME}.bat"; IconFilename: "{{app}}\\icon.ico"; Comment: "Launch {APP_NAME}"; Tasks: desktopicon

[Tasks]
Name: "desktopicon"; Description: "Create a desktop shortcut"; GroupDescription: "Additional shortcuts:"

[Run]
Filename: "{{app}}\\{APP_NAME}.bat"; Description: "Launch {APP_NAME} now"; Flags: postinstall nowait skipifsilent shellexec

[UninstallDelete]
Type: filesandordirs; Name: "{{app}}"
"""

    iss_path.write_text(iss_content, encoding="utf-8")
    print(f"  Generated {iss_path}")
    return iss_path


# ── Compile ──────────────────────────────────────────────────────────────────

def _find_iscc() -> tuple[list[str], str] | tuple[None, None]:
    """
    Return ([cmd...], description) for the best available Inno Setup compiler.

    Checks (in order):
      1. Native iscc/ISCC on PATH (Windows)
      2. Common Windows install paths (Windows)
      3. Wine + ISCC inside ~/.wine prefix (macOS/Linux)
    """
    # Native
    native = shutil.which("iscc") or shutil.which("ISCC")
    if native:
        return [native], native

    # Windows install dirs (when running natively on Windows)
    for candidate in [
        Path(r"C:\Program Files (x86)\Inno Setup 6\ISCC.exe"),
        Path(r"C:\Program Files\Inno Setup 6\ISCC.exe"),
    ]:
        if candidate.is_file():
            return [str(candidate)], str(candidate)

    # Wine on macOS/Linux
    wine = shutil.which("wine") or "/opt/homebrew/bin/wine"
    wine_iscc = Path.home() / ".wine" / "drive_c" / "InnoSetup6" / "ISCC.exe"
    if Path(wine).is_file() and wine_iscc.is_file():
        return [wine, str(wine_iscc)], f"wine {wine_iscc}"

    return None, None


def _iss_to_wine_path(iss_path: Path) -> str:
    """Convert an absolute Mac/Linux path to a Wine Z: drive path."""
    return "Z:" + str(iss_path).replace("/", "\\")


def compile_iss(iss_path: Path) -> Path | None:
    """Attempt to compile the .iss script with Inno Setup. Returns .exe path or None."""
    cmd_prefix, desc = _find_iscc()

    if cmd_prefix is None:
        print("\n  Inno Setup compiler (iscc) not found.")
        print("  Options:")
        print("    • On macOS: brew install --cask wine-stable, then install Inno Setup 6 inside Wine")
        print("    • On Windows: install Inno Setup 6 from https://jrsoftware.org/isdl.php")
        print(f"    • Or open {iss_path} in Inno Setup manually and click Build")
        return None

    # Wine needs a Z: path; native iscc takes a normal path
    is_wine = "wine" in str(cmd_prefix[0]).lower() or (len(cmd_prefix) > 1 and "wine" in str(cmd_prefix[1]).lower())
    iss_arg = _iss_to_wine_path(iss_path) if is_wine else str(iss_path)

    env = os.environ.copy()
    if is_wine:
        env.setdefault("WINEPREFIX", str(Path.home() / ".wine"))

    print(f"  Compiling with {desc} ...")
    subprocess.run(cmd_prefix + [iss_arg], check=True, env=env)

    exe_name = iss_path.stem + ".exe"
    # OutputDir=Output → exe lands in Output/ sibling of the .iss file
    exe_path = iss_path.parent / "Output" / exe_name
    if exe_path.is_file():
        print(f"  Built {exe_path} ({exe_path.stat().st_size / 1_000_000:.1f} MB)")
        return exe_path
    return None


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    version = version_string(args.version)

    print(f"\n{'=' * 60}")
    print(f"  {APP_NAME} Windows Installer Builder")
    print(f"  Version: {version}")
    print(f"{'=' * 60}\n")

    print("[1/4] Generating icon ...")
    icon_path = DIST_ROOT / "icon.ico"
    DIST_ROOT.mkdir(parents=True, exist_ok=True)
    _generate_icon_ico(icon_path)
    print(f"  Created {icon_path}")

    print("[2/4] Staging files ...")
    staging = DIST_ROOT / f"{APP_NAME}-staged"
    stage_files(staging, version)

    print("[3/4] Generating Inno Setup script ...")
    iss_path = generate_iss(staging, version, icon_path)

    exe_path = None
    if not args.skip_compile:
        print("[4/4] Compiling installer ...")
        exe_path = compile_iss(iss_path)
    else:
        print("[4/4] Skipping compilation (--skip-compile)")

    print(f"\n{'=' * 60}")
    print(f"  Done!")
    print(f"  ISS script: {iss_path}")
    if exe_path:
        print(f"  Installer:  {exe_path}")
    print(f"  Staging:    {staging}")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()

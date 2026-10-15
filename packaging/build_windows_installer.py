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
# SHA-256 of the official installers, checked against python.org's Sigstore
# signatures (identity thomas@python.org, the 3.12 release manager).
PYTHON_INSTALLER_SHA256 = {
    "amd64": "67b5635e80ea51072b87941312d00ec8927c4db9ba18938f7ad2d27b328b95fb",
    "arm64": "377ac8fd478987940088e879441e702a71b53164d2a1e6f1d51ff77a7e470258",
}

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
    "pipeline",
    "models",
    "requirements.txt",
    "run_local.py",
    "runtime",
    "styles",
    "training",
    "ui",
]


# Windows PowerShell 5.1 reads a .ps1 without a BOM as ANSI, so this script
# must stay ASCII-only.
LAUNCHER_PS1 = """$ErrorActionPreference = "Stop"
# Invoke-WebRequest is many times slower while it draws a progress bar.
$ProgressPreference = "SilentlyContinue"
try { $host.UI.RawUI.WindowTitle = "__APP__" } catch {}
Set-Location $PSScriptRoot

# CI sets NOCODE_DL_SMOKE_TEST: run the whole first launch, but never wait on
# a person (bootstrap_windows.py starts the app with --smoke-test).
$smoke = [bool]$env:NOCODE_DL_SMOKE_TEST -and ($env:NOCODE_DL_SMOKE_TEST -ne "0")

# The shortcuts run this script in a hidden window, so anything a person
# must read or answer is a dialog, never Write-Host or Read-Host.
function Show-Message([string]$text, [string]$buttons = "OK", [string]$icon = "Information") {
    Write-Host $text
    if ($smoke) { if ($buttons -eq "YesNo") { return "Yes" } else { return "OK" } }
    Add-Type -AssemblyName System.Windows.Forms
    return [System.Windows.Forms.MessageBox]::Show($text, "__APP__", $buttons, $icon).ToString()
}
function Fail([string]$text) {
    Show-Message $text "OK" "Error" | Out-Null
    exit 1
}

# 3.11-3.13 only: numpy 2.4 / scikit-learn 1.8 need >= 3.11.
$versionCheck = "import sys; exit(0 if (3,11) <= sys.version_info[:2] <= (3,13) else 1)"
function Find-Python {
    $candidates = @(
        "$env:LOCALAPPDATA\\Programs\\Python\\Python312\\python.exe",
        "$env:LOCALAPPDATA\\Programs\\Python\\Python313\\python.exe",
        "$env:LOCALAPPDATA\\Programs\\Python\\Python311\\python.exe",
        "C:\\Python312\\python.exe",
        "$env:ProgramFiles\\Python312\\python.exe"
    )
    foreach ($c in $candidates) {
        if (Test-Path $c) {
            $ok = & $c -c $versionCheck 2>$null
            if ($LASTEXITCODE -eq 0) { return $c }
        }
    }
    try {
        $cmd = (Get-Command python -ErrorAction Stop).Source
        $ok = & $cmd -c $versionCheck 2>$null
        if ($LASTEXITCODE -eq 0) { return $cmd }
    } catch {}
    return $null
}

$python = Find-Python

if (-not $python) {
    # Detect true architecture.
    # PROCESSOR_ARCHITECTURE can report ARM64 inside x64 VMs on Apple Silicon
    # (e.g. UTM/Parallels running x64 Windows), so ask WMI what the CPU is.
    $isArm64 = $false
    if ($env:PROCESSOR_ARCHITECTURE -eq "ARM64") {
        try {
            $cpu = (Get-WmiObject Win32_Processor -ErrorAction SilentlyContinue | Select-Object -First 1).Architecture
            # Architecture 12 = ARM64; anything else (9=x64, 0=x86) means emulated
            $isArm64 = ($cpu -eq 12)
        } catch { $isArm64 = $false }
    }
    $arch = if ($isArm64) { "arm64" } else { "amd64" }
    $url  = if ($isArm64) { "__ARM64_URL__" } else { "__AMD64_URL__" }
    $expected = if ($isArm64) { "__ARM64_SHA256__" } else { "__AMD64_SHA256__" }
    $answer = Show-Message ("__APP__ needs Python __PYVER__ from python.org (about 25 MB). Download and install it now?" +
        "`n`nA Python setup window will show its progress; __APP__ opens when it is done.") "YesNo" "Question"
    if ($answer -ne "Yes") { exit 1 }
    Write-Host "Downloading Python __PYVER__ ($arch) from $url"
    $installer = Join-Path $env:TEMP "python-__PYVER__-installer.exe"
    try {
        # Use the system proxy with the signed-in user's credentials (corporate networks).
        [System.Net.WebRequest]::DefaultWebProxy.Credentials = [System.Net.CredentialCache]::DefaultNetworkCredentials
        [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
        Invoke-WebRequest -Uri $url -OutFile $installer -UseBasicParsing -UseDefaultCredentials
    } catch {
        Remove-Item $installer -Force -ErrorAction SilentlyContinue
        Fail ("Could not download Python __PYVER__. Check your internet connection and open __APP__ again." +
            "`n`n$($_.Exception.Message)")
    }
    $actual = (Get-FileHash -Algorithm SHA256 $installer).Hash.ToLower()
    if ($actual -ne $expected) {
        Remove-Item $installer -Force -ErrorAction SilentlyContinue
        Fail ("The downloaded Python installer failed its checksum check, so it was not installed." +
            "`n`nExpected $expected`nGot      $actual")
    }
    # /passive shows the installer's progress without asking anything.
    $mode = if ($smoke) { "/quiet" } else { "/passive" }
    $proc = Start-Process $installer -ArgumentList "$mode InstallAllUsers=0 PrependPath=1 Include_test=0 Include_launcher=0" -Wait -PassThru
    Remove-Item $installer -Force -ErrorAction SilentlyContinue
    if ($proc.ExitCode -ne 0) {
        Fail "The Python installer exited with code $($proc.ExitCode). Open __APP__ again to retry."
    }
    $python = "$env:LOCALAPPDATA\\Programs\\Python\\Python312\\python.exe"
}

if (-not (Test-Path $python)) {
    Fail "Python could not be found or installed. Please install Python 3.12 from https://www.python.org and open __APP__ again."
}

# bootstrap_windows.py shows the progress window, writes logs to
# %LOCALAPPDATA%\\__APP__\\logs and reports its own errors; its exit code is the app's.
Write-Host "[__APP__] Using Python: $python"
& $python runtime\\bootstrap_windows.py
exit $LASTEXITCODE
"""


def launcher_ps1() -> str:
    script = LAUNCHER_PS1
    for key, value in {
        "__APP__": APP_NAME,
        "__PYVER__": PYTHON_VERSION,
        "__AMD64_URL__": PYTHON_INSTALLER_URL.format(arch="amd64"),
        "__ARM64_URL__": PYTHON_INSTALLER_URL.format(arch="arm64"),
        "__AMD64_SHA256__": PYTHON_INSTALLER_SHA256["amd64"],
        "__ARM64_SHA256__": PYTHON_INSTALLER_SHA256["arm64"],
    }.items():
        script = script.replace(key, value)
    return script.replace("\n", "\r\n")


# The Start menu / desktop shortcut: PowerShell in a hidden (and, until it
# hides, minimized) window, so only bootstrap_windows.py's progress window shows.
SHORTCUT_PARAMETERS = (
    f'-NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File ""{{app}}\\{APP_NAME}.ps1""'
)


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
    ps1 = staging / f"{APP_NAME}.ps1"
    ps1.write_text(launcher_ps1(), encoding="utf-8")

    # For running by hand or from CI: a visible console and the app's exit code.
    # The Start menu and desktop shortcuts run the .ps1 in a hidden window instead.
    bat = staging / f"{APP_NAME}.bat"
    bat.write_text(
        f"@echo off\r\n"
        f'powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0{APP_NAME}.ps1"\r\n'
        f"exit /b %ERRORLEVEL%\r\n",
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
Name: "{{group}}\\{APP_NAME}"; Filename: "{{sys}}\\WindowsPowerShell\\v1.0\\powershell.exe"; Parameters: "{SHORTCUT_PARAMETERS}"; WorkingDir: "{{app}}"; IconFilename: "{{app}}\\icon.ico"; Comment: "Launch {APP_NAME}"; Flags: runminimized
Name: "{{autodesktop}}\\{APP_NAME}"; Filename: "{{sys}}\\WindowsPowerShell\\v1.0\\powershell.exe"; Parameters: "{SHORTCUT_PARAMETERS}"; WorkingDir: "{{app}}"; IconFilename: "{{app}}\\icon.ico"; Comment: "Launch {APP_NAME}"; Tasks: desktopicon; Flags: runminimized

[Tasks]
Name: "desktopicon"; Description: "Create a desktop shortcut"; GroupDescription: "Additional shortcuts:"

[Run]
Filename: "{{sys}}\\WindowsPowerShell\\v1.0\\powershell.exe"; Parameters: "{SHORTCUT_PARAMETERS}"; WorkingDir: "{{app}}"; Description: "Launch {APP_NAME} now"; Flags: postinstall nowait skipifsilent runminimized

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
    shutil.copy2(icon_path, staging / "icon.ico")  # the shortcuts' IconFilename

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

"""Build, sign, notarize, and staple the macOS NoCode-DL installer.

All the work happens inside build_macos_installer.py, in this order:
assemble the .app -> codesign it -> pkgbuild -> productbuild -> productsign
-> notarize -> staple. Signing the app has to happen before pkgbuild, which
snapshots the bundle; this script used to sign a copy afterwards (and in a
build folder the build script no longer uses), so the shipped app was unsigned.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
BUILD_SCRIPT = ROOT / "packaging" / "build_macos_installer.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build, sign, notarize, and staple the macOS NoCode-DL installer."
    )
    parser.add_argument("--version", required=True, help="Version string to build and notarize.")
    parser.add_argument(
        "--app-sign-identity",
        required=True,
        help='Developer ID Application identity, e.g. "Developer ID Application: Your Name (TEAMID)"',
    )
    parser.add_argument(
        "--installer-sign-identity",
        required=True,
        help='Developer ID Installer identity, e.g. "Developer ID Installer: Your Name (TEAMID)"',
    )
    parser.add_argument(
        "--keychain-profile",
        required=True,
        help='Stored notarytool keychain profile name, e.g. "AC_PASSWORD"',
    )
    parser.add_argument(
        "--no-hardened-runtime",
        action="store_true",
        help="Skip the hardened runtime flag when signing the app bundle.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cmd = [
        sys.executable, str(BUILD_SCRIPT),
        "--version", args.version,
        "--app-sign-identity", args.app_sign_identity,
        "--installer-sign-identity", args.installer_sign_identity,
        "--notary-profile", args.keychain_profile,
    ]
    if args.no_hardened_runtime:
        cmd.append("--no-hardened-runtime")
    subprocess.run(cmd, cwd=ROOT, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

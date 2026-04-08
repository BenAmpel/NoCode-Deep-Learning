from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
BUILD_SCRIPT = ROOT / "packaging" / "build_macos_installer.py"
APP_NAME = "NoCode-DL"


def run(cmd: list[str], *, cwd: Path | None = None) -> None:
    subprocess.run(cmd, cwd=cwd or ROOT, check=True)


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


def sign_app_bundle(app_bundle: Path, identity: str, hardened_runtime: bool) -> None:
    cmd = [
        "codesign",
        "--force",
        "--deep",
        "--sign",
        identity,
        "--timestamp",
    ]
    if hardened_runtime:
        cmd.extend(["--options", "runtime"])
    cmd.append(str(app_bundle))
    run(cmd)
    run(["codesign", "--verify", "--deep", "--strict", "--verbose=2", str(app_bundle)])
    run(["spctl", "-a", "-t", "exec", "-vv", str(app_bundle)])


def build_signed_installer(version: str, installer_identity: str) -> Path:
    build_root = ROOT / "build" / "macos-installer"
    distribution = build_root / "distribution.xml"
    package_path = build_root / "packages"
    resources = build_root / "installer-resources"
    signed_pkg = ROOT / "dist" / f"{APP_NAME}-{version}-signed.pkg"

    if signed_pkg.exists():
        signed_pkg.unlink()

    run(
        [
            "productbuild",
            "--sign",
            installer_identity,
            "--distribution",
            str(distribution),
            "--package-path",
            str(package_path),
            "--resources",
            str(resources),
            str(signed_pkg),
        ]
    )
    run(["pkgutil", "--check-signature", str(signed_pkg)])
    return signed_pkg


def notarize(pkg_path: Path, keychain_profile: str) -> None:
    run(
        [
            "xcrun",
            "notarytool",
            "submit",
            str(pkg_path),
            "--keychain-profile",
            keychain_profile,
            "--wait",
        ]
    )
    run(["xcrun", "stapler", "staple", str(pkg_path)])
    run(["xcrun", "stapler", "validate", str(pkg_path)])


def main() -> int:
    args = parse_args()

    run([sys.executable, str(BUILD_SCRIPT), "--version", args.version])

    app_bundle = (
        ROOT
        / "build"
        / "macos-installer"
        / "component-root"
        / "Applications"
        / f"{APP_NAME}.app"
    )
    if not app_bundle.is_dir():
        raise RuntimeError(f"Expected app bundle at {app_bundle}")

    sign_app_bundle(
        app_bundle=app_bundle,
        identity=args.app_sign_identity,
        hardened_runtime=not args.no_hardened_runtime,
    )

    signed_pkg = build_signed_installer(
        version=args.version,
        installer_identity=args.installer_sign_identity,
    )
    notarize(signed_pkg, args.keychain_profile)
    print(f"Signed and notarized installer: {signed_pkg}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

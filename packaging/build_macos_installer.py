from __future__ import annotations

import argparse
import hashlib
import html
import os
import plistlib
import shutil
import stat
import subprocess
import tempfile
import urllib.request
import zlib
from datetime import datetime
from pathlib import Path


APP_NAME = "NoCode-DL"
ROOT = Path(__file__).resolve().parent.parent
BUILD_ROOT = Path(tempfile.gettempdir()) / "NoCode-DL-macos-installer"
DIST_ROOT = ROOT / "dist"
COMPONENT_ROOT = BUILD_ROOT / "component-root"
PACKAGES_DIR = BUILD_ROOT / "packages"
RESOURCES_DIR = BUILD_ROOT / "installer-resources"
SCRIPTS_DIR = BUILD_ROOT / "scripts"
APP_BUNDLE = COMPONENT_ROOT / "Applications" / f"{APP_NAME}.app"
CONTENTS = APP_BUNDLE / "Contents"
MACOS_DIR = CONTENTS / "MacOS"
APP_RESOURCES_DIR = CONTENTS / "Resources"
SOURCE_COPY_DIR = APP_RESOURCES_DIR / "app"
APP_PKG = PACKAGES_DIR / f"{APP_NAME}-app.pkg"

PYTHON_VERSION = "3.12.10"
PYTHON_PKG_NAME = f"python-{PYTHON_VERSION}-macos11.pkg"
PYTHON_PKG_URL = f"https://www.python.org/ftp/python/{PYTHON_VERSION}/{PYTHON_PKG_NAME}"
PYTHON_PKG_SHA256 = "8373e58da4ea146b3eb1c1f9834f19a319440b6b679b06050b1f9ee3237aa8e4"
VENDOR_DIR = ROOT / "packaging" / "vendor"
VENDOR_PYTHON_PKG = VENDOR_DIR / PYTHON_PKG_NAME
RUNTIME_ITEMS = [
    "SETUP.md",
    "app.py",
    "bootstrap_macos.py",
    "bootstrap_windows.py",
    "config.py",
    "data_pipeline",
    "detection",
    "eval",
    "export",
    "healthcheck.py",
    "install.py",
    "modalities",
    "models",
    "requirements.txt",
    "run_local.py",
    "runtime_setup.py",
    "training",
    "ui",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a branded macOS .pkg installer for NoCode-DL.")
    parser.add_argument("--version", help="Installer/app version. Defaults to current timestamp.")
    parser.add_argument(
        "--installer-sign-identity",
        default=os.environ.get("NOCODE_DL_MAC_INSTALLER_IDENTITY", ""),
        help="Optional Developer ID Installer identity. If omitted, the pkg is left unsigned.",
    )
    parser.add_argument(
        "--notary-profile",
        default=os.environ.get("NOCODE_DL_MAC_NOTARY_PROFILE", ""),
        help="Optional notarytool keychain profile. Used only when signing is enabled.",
    )
    parser.add_argument(
        "--keep-build-artifacts",
        action="store_true",
        help="Keep the intermediate build/macos-installer directory after the .pkg is created.",
    )
    return parser.parse_args()


def version_string(raw: str | None) -> str:
    if raw:
        return raw
    return datetime.now().strftime("%Y.%m.%d.%H%M")


def clean_build_root() -> None:
    if BUILD_ROOT.exists():
        shutil.rmtree(BUILD_ROOT)
    BUILD_ROOT.mkdir(parents=True, exist_ok=True)
    DIST_ROOT.mkdir(parents=True, exist_ok=True)
    PACKAGES_DIR.mkdir(parents=True, exist_ok=True)
    RESOURCES_DIR.mkdir(parents=True, exist_ok=True)
    SCRIPTS_DIR.mkdir(parents=True, exist_ok=True)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ensure_python_pkg() -> Path:
    VENDOR_DIR.mkdir(parents=True, exist_ok=True)
    if not VENDOR_PYTHON_PKG.exists():
        print(f"Downloading official Python installer from {PYTHON_PKG_URL}")
        urllib.request.urlretrieve(PYTHON_PKG_URL, VENDOR_PYTHON_PKG)
    sha256_value = file_sha256(VENDOR_PYTHON_PKG)
    if sha256_value != PYTHON_PKG_SHA256:
        raise RuntimeError(
            f"Embedded Python installer checksum mismatch for {VENDOR_PYTHON_PKG}: "
            f"expected {PYTHON_PKG_SHA256}, got {sha256_value}"
        )
    return VENDOR_PYTHON_PKG


def _clamp(value: float) -> int:
    return max(0, min(255, int(round(value))))


def _blend(a: tuple[int, int, int, int], b: tuple[int, int, int, int], t: float) -> tuple[int, int, int, int]:
    return tuple(_clamp(a[i] * (1 - t) + b[i] * t) for i in range(4))


def _put_pixel(buf: bytearray, width: int, x: int, y: int, rgba: tuple[int, int, int, int]) -> None:
    idx = (y * width + x) * 4
    buf[idx: idx + 4] = bytes(rgba)


def _draw_circle(buf: bytearray, width: int, height: int, cx: float, cy: float, radius: float, rgba: tuple[int, int, int, int]) -> None:
    x0 = max(0, int(cx - radius - 1))
    x1 = min(width - 1, int(cx + radius + 1))
    y0 = max(0, int(cy - radius - 1))
    y1 = min(height - 1, int(cy + radius + 1))
    r2 = radius * radius
    for y in range(y0, y1 + 1):
        for x in range(x0, x1 + 1):
            dx = x + 0.5 - cx
            dy = y + 0.5 - cy
            if dx * dx + dy * dy <= r2:
                _put_pixel(buf, width, x, y, rgba)


def _draw_line(buf: bytearray, width: int, height: int, ax: float, ay: float, bx: float, by: float, thickness: float, rgba: tuple[int, int, int, int]) -> None:
    x0 = max(0, int(min(ax, bx) - thickness - 1))
    x1 = min(width - 1, int(max(ax, bx) + thickness + 1))
    y0 = max(0, int(min(ay, by) - thickness - 1))
    y1 = min(height - 1, int(max(ay, by) + thickness + 1))
    vx = bx - ax
    vy = by - ay
    vv = vx * vx + vy * vy or 1.0
    threshold = thickness * thickness
    for y in range(y0, y1 + 1):
        for x in range(x0, x1 + 1):
            px = x + 0.5
            py = y + 0.5
            t = ((px - ax) * vx + (py - ay) * vy) / vv
            t = 0.0 if t < 0.0 else 1.0 if t > 1.0 else t
            dx = px - (ax + t * vx)
            dy = py - (ay + t * vy)
            if dx * dx + dy * dy <= threshold:
                _put_pixel(buf, width, x, y, rgba)


def write_png(path: Path, width: int, height: int, rgba_bytes: bytes) -> None:
    raw = bytearray()
    stride = width * 4
    for y in range(height):
        raw.append(0)
        start = y * stride
        raw.extend(rgba_bytes[start:start + stride])

    def chunk(tag: bytes, data: bytes) -> bytes:
        return (
            len(data).to_bytes(4, "big")
            + tag
            + data
            + zlib.crc32(tag + data).to_bytes(4, "big")
        )

    png = bytearray()
    png.extend(b"\x89PNG\r\n\x1a\n")
    png.extend(chunk(b"IHDR", width.to_bytes(4, "big") + height.to_bytes(4, "big") + b"\x08\x06\x00\x00\x00"))
    png.extend(chunk(b"IDAT", zlib.compress(bytes(raw), level=9)))
    png.extend(chunk(b"IEND", b""))
    path.write_bytes(bytes(png))


def generate_icon_png(path: Path, size: int = 1024) -> None:
    buf = bytearray(size * size * 4)
    c1 = (14, 38, 52, 255)
    c2 = (255, 133, 84, 255)
    radius = size * 0.22
    margin = size * 0.08
    inner = size - 2 * margin

    for y in range(size):
        for x in range(size):
            tx = x / max(1, size - 1)
            ty = y / max(1, size - 1)
            bg = _blend(c1, c2, (tx * 0.45 + ty * 0.55))
            rx = min(x - margin, size - margin - x - 1)
            ry = min(y - margin, size - margin - y - 1)
            inside = (
                margin <= x < size - margin
                and margin <= y < size - margin
                and (
                    (rx >= radius and ry >= 0)
                    or (ry >= radius and rx >= 0)
                    or (rx - radius) ** 2 + (ry - radius) ** 2 <= radius ** 2
                )
            )
            pixel = bg if inside else (0, 0, 0, 0)
            _put_pixel(buf, size, x, y, pixel)

    white = (247, 251, 255, 255)
    pale = (183, 232, 230, 255)
    coral = (255, 221, 207, 255)
    left = margin + inner * 0.28
    top = margin + inner * 0.24
    bottom = margin + inner * 0.74
    _draw_line(buf, size, size, left, top, left, bottom, size * 0.03, white)
    _draw_line(buf, size, size, left + inner * 0.12, top, left + inner * 0.12, bottom, size * 0.03, white)
    for yy in (top, (top + bottom) / 2, bottom):
        _draw_line(buf, size, size, left - inner * 0.04, yy, left + inner * 0.16, yy, size * 0.018, coral)

    n1 = (margin + inner * 0.60, margin + inner * 0.30)
    n2 = (margin + inner * 0.78, margin + inner * 0.45)
    n3 = (margin + inner * 0.64, margin + inner * 0.68)
    n4 = (margin + inner * 0.83, margin + inner * 0.76)

    for a, b in ((n1, n2), (n1, n3), (n2, n3), (n2, n4), (n3, n4)):
        _draw_line(buf, size, size, a[0], a[1], b[0], b[1], size * 0.014, pale)
    for node, color in ((n1, white), (n2, pale), (n3, white), (n4, coral)):
        _draw_circle(buf, size, size, node[0], node[1], size * 0.05, color)

    write_png(path, size, size, bytes(buf))


def generate_background_png(path: Path, width: int = 1280, height: int = 800) -> None:
    buf = bytearray(width * height * 4)
    top = (10, 20, 28, 255)
    bottom = (28, 52, 60, 255)
    accent = (255, 133, 84, 255)
    mint = (116, 214, 210, 255)

    for y in range(height):
        row = _blend(top, bottom, y / max(1, height - 1))
        for x in range(width):
            shade = _blend(row, accent, max(0.0, 1.0 - ((x - width * 0.82) ** 2 + (y - height * 0.18) ** 2) / (width * height * 0.08)))
            shade = _blend(shade, mint, max(0.0, 1.0 - ((x - width * 0.18) ** 2 + (y - height * 0.72) ** 2) / (width * height * 0.12)) * 0.25)
            _put_pixel(buf, width, x, y, shade)

    white = (246, 249, 252, 255)
    for points in (
        ((170, 590), (340, 460), (510, 540), (670, 360)),
        ((740, 210), (920, 280), (1080, 170)),
    ):
        for a, b in zip(points, points[1:]):
            _draw_line(buf, width, height, a[0], a[1], b[0], b[1], 6, (255, 255, 255, 80))
        for cx, cy in points:
            _draw_circle(buf, width, height, cx, cy, 14, white)

    write_png(path, width, height, bytes(buf))


def write_welcome_html(path: Path, version: str) -> None:
    body = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, sans-serif; color: #10262f; margin: 48px; }}
    h1 {{ font-size: 34px; margin-bottom: 10px; }}
    p {{ font-size: 16px; line-height: 1.5; max-width: 760px; }}
    .badge {{ display: inline-block; padding: 6px 10px; border-radius: 999px; background: #ffe1d1; color: #9a4d26; font-size: 12px; letter-spacing: 0.08em; text-transform: uppercase; }}
  </style>
</head>
<body>
  <div class="badge">Version {html.escape(version)}</div>
  <h1>NoCode-DL</h1>
  <p>Install a local deep learning studio for students. This installer adds <strong>NoCode-DL.app</strong> to <strong>/Applications</strong>. If Python 3.12 is not already installed, it will be downloaded automatically during installation. An internet connection is required the first time.</p>
  <p>The app runs entirely on the local machine and opens at <code>http://127.0.0.1:7860</code> when launched.</p>
</body>
</html>
"""
    path.write_text(body, encoding="utf-8")


def write_conclusion_html(path: Path) -> None:
    body = """<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <style>
    body { font-family: -apple-system, BlinkMacSystemFont, sans-serif; color: #10262f; margin: 48px; }
    h1 { font-size: 30px; margin-bottom: 10px; }
    p { font-size: 16px; line-height: 1.5; max-width: 760px; }
  </style>
</head>
<body>
  <h1>Ready to launch</h1>
  <p>Open <strong>NoCode-DL</strong> from Applications. On first launch, the app finalizes its local environment in <code>~/Library/Application Support/NoCode-DL</code> and then opens the student-facing interface in your browser.</p>
</body>
</html>
"""
    path.write_text(body, encoding="utf-8")


def create_iconset(icon_png: Path) -> Path:
    iconset_dir = BUILD_ROOT / "NoCode-DL.iconset"
    if iconset_dir.exists():
        shutil.rmtree(iconset_dir)
    iconset_dir.mkdir(parents=True)

    sizes = {
        "icon_16x16.png": 16,
        "icon_16x16@2x.png": 32,
        "icon_32x32.png": 32,
        "icon_32x32@2x.png": 64,
        "icon_128x128.png": 128,
        "icon_128x128@2x.png": 256,
        "icon_256x256.png": 256,
        "icon_256x256@2x.png": 512,
        "icon_512x512.png": 512,
        "icon_512x512@2x.png": 1024,
    }
    for name, size in sizes.items():
        subprocess.run(
            ["sips", "-z", str(size), str(size), str(icon_png), "--out", str(iconset_dir / name)],
            check=True,
            cwd=ROOT,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

    icns_path = RESOURCES_DIR / f"{APP_NAME}.icns"
    subprocess.run(["iconutil", "-c", "icns", str(iconset_dir), "-o", str(icns_path)], check=True, cwd=ROOT)
    return icns_path


def generate_branding_assets(version: str) -> Path:
    icon_png = RESOURCES_DIR / "NoCode-DL-icon-1024.png"
    background_png = RESOURCES_DIR / "background.png"
    generate_icon_png(icon_png)
    generate_background_png(background_png)
    write_welcome_html(RESOURCES_DIR / "welcome.html", version)
    write_conclusion_html(RESOURCES_DIR / "conclusion.html")
    return create_iconset(icon_png)


def copy_source_tree(version: str) -> None:
    if SOURCE_COPY_DIR.exists():
        shutil.rmtree(SOURCE_COPY_DIR)
    SOURCE_COPY_DIR.parent.mkdir(parents=True, exist_ok=True)
    SOURCE_COPY_DIR.mkdir(parents=True, exist_ok=True)

    for item_name in RUNTIME_ITEMS:
        source = ROOT / item_name
        destination = SOURCE_COPY_DIR / item_name
        print(f"Bundling {item_name} ...")
        if source.is_dir():
            shutil.copytree(
                source,
                destination,
                ignore=shutil.ignore_patterns(".DS_Store", "._*", "__pycache__"),
                copy_function=shutil.copyfile,
            )
        else:
            shutil.copyfile(source, destination)

    subprocess.run(["xattr", "-cr", str(SOURCE_COPY_DIR)], check=True, cwd=ROOT)
    (SOURCE_COPY_DIR / ".bundle_version").write_text(version + "\n", encoding="utf-8")


def write_launcher() -> None:
    launcher = """#!/bin/bash
set -euo pipefail

APP_DIR="$(cd "$(dirname "$0")/.." && pwd)"
RESOURCE_APP="$APP_DIR/Resources/app"
LOG_DIR="$HOME/Library/Logs/NoCode-DL"
LOG_FILE="$LOG_DIR/launcher.log"

PYTHON="/Library/Frameworks/Python.framework/Versions/3.12/bin/python3.12"
if [ ! -x "$PYTHON" ]; then
  PYTHON=""
  for cmd in python3.13 python3.12 python3.11 python3.10 python3; do
    if command -v "$cmd" >/dev/null 2>&1; then
      if "$cmd" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 10) else 1)' >/dev/null 2>&1; then
        PYTHON="$cmd"
        break
      fi
    fi
  done
fi

if [ -z "$PYTHON" ]; then
  osascript -e 'display dialog "Python 3.12 was not found. Re-run the NoCode-DL installer to install the bundled Python runtime." buttons {"OK"} default button "OK" with icon caution' >/dev/null 2>&1 || true
  exit 1
fi

mkdir -p "$LOG_DIR"

# On Apple Silicon, force native arm64 execution so pip installs arm64
# packages instead of x86_64 ones (which would happen if the app was
# launched via Rosetta). On Intel Macs, arch -arm64 would fail, so we
# detect the CPU and only use it when needed.
ARCH="$(uname -m)"
if [ "$ARCH" = "arm64" ]; then
  exec arch -arm64 "$PYTHON" "$RESOURCE_APP/bootstrap_macos.py" >>"$LOG_FILE" 2>&1
else
  exec "$PYTHON" "$RESOURCE_APP/bootstrap_macos.py" >>"$LOG_FILE" 2>&1
fi
"""
    launcher_path = MACOS_DIR / APP_NAME
    launcher_path.write_text(launcher, encoding="utf-8")
    current = launcher_path.stat().st_mode
    launcher_path.chmod(current | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


def write_info_plist(version: str) -> None:
    info = {
        "CFBundleDisplayName": APP_NAME,
        "CFBundleExecutable": APP_NAME,
        "CFBundleIconFile": APP_NAME,
        "CFBundleIdentifier": "com.nocode_dl.app",
        "CFBundleInfoDictionaryVersion": "6.0",
        "CFBundleName": APP_NAME,
        "CFBundlePackageType": "APPL",
        "CFBundleShortVersionString": version,
        "CFBundleVersion": version,
        "LSMinimumSystemVersion": "12.0",
        "NSHighResolutionCapable": True,
        "LSRequiresNativeExecution": True,
    }
    with open(CONTENTS / "Info.plist", "wb") as fh:
        plistlib.dump(info, fh)


def write_postinstall_script() -> None:
    script = f"""#!/bin/bash
set -euo pipefail

PYTHON_BIN="/Library/Frameworks/Python.framework/Versions/3.12/bin/python3.12"

if [ ! -x "$PYTHON_BIN" ]; then
  echo "Python 3.12 not found — downloading from python.org..."
  TMPDIR_PKG="$(mktemp -d)"
  PYTHON_PKG="$TMPDIR_PKG/{PYTHON_PKG_NAME}"
  curl -fsSL "{PYTHON_PKG_URL}" -o "$PYTHON_PKG"
  EXPECTED_SHA256="{PYTHON_PKG_SHA256}"
  ACTUAL_SHA256="$(shasum -a 256 "$PYTHON_PKG" | awk '{{print $1}}')"
  if [ "$ACTUAL_SHA256" != "$EXPECTED_SHA256" ]; then
    echo "Python installer checksum mismatch"
    echo "Expected: $EXPECTED_SHA256"
    echo "Actual:   $ACTUAL_SHA256"
    exit 1
  fi
  /usr/sbin/installer -pkg "$PYTHON_PKG" -target /
  rm -rf "$TMPDIR_PKG"
fi

exit 0
"""
    postinstall = SCRIPTS_DIR / "postinstall"
    postinstall.write_text(script, encoding="utf-8")
    current = postinstall.stat().st_mode
    postinstall.chmod(current | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


def prepare_scripts() -> None:
    write_postinstall_script()


def build_component_pkg(version: str) -> None:
    if APP_PKG.exists():
        APP_PKG.unlink()
    subprocess.run(
        [
            "pkgbuild",
            "--root",
            str(COMPONENT_ROOT),
            "--scripts",
            str(SCRIPTS_DIR),
            "--install-location",
            "/",
            "--identifier",
            "com.nocode_dl.app",
            "--version",
            version,
            str(APP_PKG),
        ],
        check=True,
        cwd=ROOT,
        env=dict(os.environ, COPYFILE_DISABLE="1"),
    )


def write_distribution(version: str) -> Path:
    distribution = BUILD_ROOT / "distribution.xml"
    xml = f"""<?xml version="1.0" encoding="utf-8"?>
<installer-gui-script minSpecVersion="2">
  <title>{APP_NAME}</title>
  <background file="background.png" alignment="center" scaling="proportional"/>
  <welcome file="welcome.html" mime-type="text/html"/>
  <conclusion file="conclusion.html" mime-type="text/html"/>
  <options customize="never" require-scripts="false"/>
  <choices-outline>
    <line choice="default">
      <line choice="app"/>
    </line>
  </choices-outline>
  <choice id="default"/>
  <choice id="app" title="{APP_NAME}" visible="false">
    <pkg-ref id="com.nocode_dl.app"/>
  </choice>
  <pkg-ref id="com.nocode_dl.app" version="{version}" onConclusion="none">{APP_PKG.name}</pkg-ref>
</installer-gui-script>
"""
    distribution.write_text(xml, encoding="utf-8")
    return distribution


def build_product_pkg(version: str) -> Path:
    pkg_path = DIST_ROOT / f"{APP_NAME}-{version}.pkg"
    if pkg_path.exists():
        pkg_path.unlink()
    distribution = write_distribution(version)
    subprocess.run(
        [
            "productbuild",
            "--distribution",
            str(distribution),
            "--package-path",
            str(PACKAGES_DIR),
            "--resources",
            str(RESOURCES_DIR),
            str(pkg_path),
        ],
        check=True,
        cwd=ROOT,
        env=dict(os.environ, COPYFILE_DISABLE="1"),
    )
    return pkg_path


def sign_pkg(pkg_path: Path, installer_identity: str) -> Path:
    signed_path = pkg_path.with_name(pkg_path.stem + "-signed.pkg")
    subprocess.run(
        ["productsign", "--sign", installer_identity, str(pkg_path), str(signed_path)],
        check=True,
    )
    return signed_path


def notarize_and_staple(pkg_path: Path, notary_profile: str) -> None:
    print(f"  Submitting {pkg_path.name} for notarization...")
    subprocess.run(
        ["xcrun", "notarytool", "submit", str(pkg_path),
         "--keychain-profile", notary_profile, "--wait"],
        check=True,
    )
    print("  Stapling notarization ticket...")
    subprocess.run(["xcrun", "stapler", "staple", str(pkg_path)], check=True)


def main() -> int:
    args = parse_args()
    version = version_string(args.version)
    clean_build_root()
    MACOS_DIR.mkdir(parents=True, exist_ok=True)
    APP_RESOURCES_DIR.mkdir(parents=True, exist_ok=True)
    copy_source_tree(version)
    icon_path = generate_branding_assets(version)
    shutil.copy2(icon_path, APP_RESOURCES_DIR / icon_path.name)
    write_launcher()
    write_info_plist(version)
    prepare_scripts()
    build_component_pkg(version)
    pkg_path = build_product_pkg(version)
    if not args.keep_build_artifacts:
        shutil.rmtree(BUILD_ROOT)

    final_path = pkg_path
    if args.installer_sign_identity:
        print("Signing...")
        final_path = sign_pkg(pkg_path, args.installer_sign_identity)
        if args.notary_profile:
            print("Notarizing and stapling...")
            notarize_and_staple(final_path, args.notary_profile)

    print(f"Built installer: {final_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

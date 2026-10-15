"""Check a built macOS installer for what Apple's notary service rejects.

Notarization needs a Developer ID and an Apple account, so CI can't run it.
This runs everything checkable locally, so a release build fails in seconds
here instead of minutes later at Apple:

- the .pkg carries an installer signature;
- the .app inside passes ``codesign --verify --strict``, is not ad-hoc signed,
  uses the hardened runtime, has a secure timestamp and no ``get-task-allow``
  entitlement;
- every Mach-O binary in the payload is signed the same way (today there are
  none: the app is Python source plus a bash launcher).

With ``--developer-id`` it also requires Developer ID certificates, which is
what build_macos_installer.py asks for before it submits for notarization.

Usage: python packaging/verify_macos_signing.py dist/NoCode-DL-<version>.pkg
"""
from __future__ import annotations

import argparse
import plistlib
import re
import subprocess
import tempfile
from pathlib import Path

_MACHO_MAGICS = {
    bytes.fromhex(m) for m in ("feedface", "feedfacf", "cefaedfe", "cffaedfe", "cafebabe", "bebafeca")
}


def _run(cmd: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True)


def _is_macho(path: Path) -> bool:
    try:
        with path.open("rb") as fh:
            return fh.read(4) in _MACHO_MAGICS
    except OSError:
        return False


def check_code(path: Path, *, developer_id: bool) -> list[str]:
    """Problems with the signature of an app bundle or Mach-O file."""
    verify = _run(["codesign", "--verify", "--strict", "--verbose=2", str(path)])
    if verify.returncode != 0:
        return [f"{path.name}: codesign --verify failed: {verify.stderr.strip()}"]

    info = _run(["codesign", "-dvvv", str(path)]).stderr  # codesign -d reports on stderr
    problems = []
    if "Signature=adhoc" in info:
        problems.append(f"{path.name}: ad-hoc signature (notarization needs a real identity)")
    flags = re.search(r"flags=0x[0-9a-f]+\(([^)]*)\)", info)
    if not flags or "runtime" not in flags.group(1).split(","):
        problems.append(f"{path.name}: hardened runtime is off (codesign --options runtime)")
    if not re.search(r"^Timestamp=", info, re.M):
        problems.append(f"{path.name}: no secure timestamp (codesign --timestamp)")
    authority = re.search(r"^Authority=(.*)$", info, re.M)
    if developer_id and not (authority and authority.group(1).startswith("Developer ID Application:")):
        problems.append(f"{path.name}: not signed with a Developer ID Application certificate")

    ents = _run(["codesign", "-d", "--entitlements", "-", "--xml", str(path)]).stdout
    if ents.strip():
        try:
            if plistlib.loads(ents.encode()).get("com.apple.security.get-task-allow"):
                problems.append(f"{path.name}: has the get-task-allow (debugging) entitlement")
        except Exception:
            pass
    return problems


def check_pkg(pkg: Path, *, developer_id: bool = False) -> list[str]:
    """Problems that would stop notarization of ``pkg``; empty when ready."""
    problems = []
    sig = _run(["pkgutil", "--check-signature", str(pkg)]).stdout
    if "Status: no signature" in sig or "Status: signed" not in sig:
        problems.append(f"{pkg.name}: installer is not signed (productsign)")
    elif developer_id and "Developer ID Installer:" not in sig:
        problems.append(f"{pkg.name}: not signed with a Developer ID Installer certificate")

    with tempfile.TemporaryDirectory() as tmp:
        expanded = Path(tmp) / "expanded"
        result = _run(["pkgutil", "--expand-full", str(pkg), str(expanded)])
        if result.returncode != 0:
            return problems + [f"{pkg.name}: pkgutil --expand-full failed: {result.stderr.strip()}"]
        apps = [p for p in expanded.rglob("*.app") if p.is_dir() and p.parent.suffix != ".app"]
        if not apps:
            problems.append(f"{pkg.name}: no .app found in the payload")
        for app in apps:
            problems += check_code(app, developer_id=developer_id)
        for path in expanded.rglob("*"):
            if path.is_file() and not path.is_symlink() and _is_macho(path):
                problems += check_code(path, developer_id=developer_id)
    return problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("pkg", type=Path)
    parser.add_argument("--developer-id", action="store_true",
                        help="Also require Developer ID certificates (release builds).")
    args = parser.parse_args(argv)
    problems = check_pkg(args.pkg, developer_id=args.developer_id)
    for problem in problems:
        print(f"FAIL {problem}")
    if not problems:
        print(f"OK {args.pkg.name}: signed, hardened runtime, secure timestamp, no debug entitlement")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())

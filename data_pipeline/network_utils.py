from __future__ import annotations

import hashlib
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import urlopen



def validate_https_url(url: str, *, allowed_hosts: set[str]) -> None:
    parsed = urlparse(url)
    if parsed.scheme != "https":
        raise ValueError(f"Only https downloads are allowed: {url}")
    if not parsed.hostname or parsed.hostname not in allowed_hosts:
        raise ValueError(f"Download host is not allowlisted: {url}")


def download_file(
    url: str,
    destination: Path,
    *,
    allowed_hosts: set[str],
    expected_sha256: str | None = None,
    timeout: int = 60,
) -> Path:
    validate_https_url(url, allowed_hosts=allowed_hosts)
    destination.parent.mkdir(parents=True, exist_ok=True)

    hasher = hashlib.sha256() if expected_sha256 else None
    with urlopen(url, timeout=timeout) as response:
        final_url = response.geturl()
        validate_https_url(final_url, allowed_hosts=allowed_hosts)
        with destination.open("wb") as fh:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                fh.write(chunk)
                if hasher is not None:
                    hasher.update(chunk)

    if expected_sha256 is not None:
        actual_sha256 = hasher.hexdigest()
        if actual_sha256 != expected_sha256:
            destination.unlink(missing_ok=True)
            raise RuntimeError(
                f"Downloaded file checksum mismatch for {destination}: "
                f"expected {expected_sha256}, got {actual_sha256}"
            )

    return destination

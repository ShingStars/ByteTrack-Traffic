#!/usr/bin/env python3
"""Download and verify the YOLOX-S COCO pretrained checkpoint.

Downloads the official YOLOX-S checkpoint, verifies its SHA-256 digest, and
atomically installs it at the requested destination. Existing files are left
untouched unless --force is given.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

WEIGHTS_URL = (
    "https://github.com/Megvii-BaseDetection/YOLOX/releases/download/"
    "0.1.1rc0/yolox_s.pth"
)
WEIGHTS_SHA256 = "f55ded7181e1b0c13285c56e7790b8f0e8f8db590fe4edb37f0b7f345c913a30"
WEIGHTS_SIZE = 72089125
DEFAULT_DEST = Path("weights/yolox_s.pth")
CHUNK = 1024 * 1024


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(CHUNK), b""):
            digest.update(block)
    return digest.hexdigest()


def download(url: str, dest: Path) -> None:
    part = dest.with_suffix(dest.suffix + ".part")
    if part.exists():
        part.unlink()
    try:
        with urllib.request.urlopen(url, timeout=60) as response:
            total = response.headers.get("Content-Length")
            total = int(total) if total is not None else None
            written = 0
            with part.open("wb") as out:
                while True:
                    block = response.read(CHUNK)
                    if not block:
                        break
                    out.write(block)
                    written += len(block)
                    if total:
                        pct = written * 100 // total
                        sys.stderr.write(f"\r  {pct:3d}%  {written}/{total} bytes")
                        sys.stderr.flush()
        sys.stderr.write("\n")
    except urllib.error.HTTPError as exc:
        raise SystemExit(
            f"Download failed with HTTP {exc.code} for {url}. "
            "Check network access and the release URL."
        ) from exc
    except urllib.error.URLError as exc:
        raise SystemExit(
            f"Download failed for {url}: {exc.reason}. "
            "Check TLS/proxy settings; this script does not disable certificate "
            "verification."
        ) from exc
    except OSError as exc:
        raise SystemExit(f"Download failed while writing {part}: {exc}") from exc

    actual = sha256_of(part)
    if actual != WEIGHTS_SHA256:
        part.unlink(missing_ok=True)
        raise SystemExit(
            "Checksum mismatch for downloaded checkpoint.\n"
            f"  expected {WEIGHTS_SHA256}\n"
            f"  actual   {actual}\n"
            "The file was removed; re-run to retry."
        )
    os.replace(part, dest)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dest",
        type=Path,
        default=DEFAULT_DEST,
        help=f"destination path (default: {DEFAULT_DEST})",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="re-download even if a valid file already exists",
    )
    args = parser.parse_args(argv)

    dest: Path = args.dest
    dest.parent.mkdir(parents=True, exist_ok=True)

    if dest.exists() and not args.force:
        actual = sha256_of(dest)
        if actual == WEIGHTS_SHA256:
            print(f"Already present and verified: {dest}")
            return 0
        raise SystemExit(
            f"{dest} exists but does not match the expected SHA-256.\n"
            f"  expected {WEIGHTS_SHA256}\n"
            f"  actual   {actual}\n"
            "Re-run with --force to replace it."
        )

    print(f"Downloading {WEIGHTS_URL}")
    download(WEIGHTS_URL, dest)
    print(f"Verified SHA-256 and installed: {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

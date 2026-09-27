"""Automated downloader and integrity verifier for on-device quantized SLM (Qwen 2.5 0.5B GGUF)."""

import argparse
import logging
import os
import sys
import urllib.request
from pathlib import Path
from typing import Union

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("download_slm")

DEFAULT_GGUF_URL = (
    "https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/"
    "qwen2.5-0.5b-instruct-q4_k_m.gguf"
)
DEFAULT_GGUF_PATH = Path("models/qwen2.5-0.5b-instruct-q4_k_m.gguf")
GGUF_MAGIC = b"GGUF"


def verify_gguf_header(file_path: Union[str, Path]) -> bool:
    """Check that the file exists, has at least 8 bytes, and begins with GGUF magic bytes."""
    path = Path(file_path)
    if not path.is_file():
        return False
    try:
        with open(path, "rb") as f:
            header = f.read(8)
            if len(header) < 4:
                return False
            # Check 'GGUF' magic
            if header[:4] != GGUF_MAGIC:
                return False
            # Version uint32 should be 2 or 3
            if len(header) >= 8:
                version = int.from_bytes(header[4:8], byteorder="little")
                if version not in (2, 3):
                    return False
            return True
    except Exception as e:
        logger.debug("Error checking GGUF header: %s", e)
        return False


def download_slm_model(
    url: str = DEFAULT_GGUF_URL,
    dest_path: Union[str, Path] = DEFAULT_GGUF_PATH,
    force: bool = False,
) -> Path:
    """Download the GGUF model with streaming progress bar and header verification."""
    dest = Path(dest_path)
    dest.parent.mkdir(parents=True, exist_ok=True)

    if not force and verify_gguf_header(dest):
        size_mb = dest.stat().st_size / (1024 * 1024)
        logger.info("Model already downloaded and verified at %s (%.1f MB).", dest, size_mb)
        return dest

    logger.info("Downloading quantized SLM from %s -> %s...", url, dest)

    def report_progress(block_num: int, block_size: int, total_size: int):
        downloaded = block_num * block_size
        if total_size > 0:
            percent = downloaded * 100.0 / total_size
            mb_down = downloaded / (1024 * 1024)
            mb_tot = total_size / (1024 * 1024)
            sys.stdout.write(f"\r  Progress: {percent:.1f}% ({mb_down:.1f} MB / {mb_tot:.1f} MB)")
            sys.stdout.flush()

    try:
        urllib.request.urlretrieve(url, str(dest), reporthook=report_progress)
        sys.stdout.write("\n")
    except Exception as e:
        if dest.exists():
            dest.unlink()
        raise RuntimeError(f"Failed to download SLM model: {e}") from e

    if not verify_gguf_header(dest):
        if dest.exists():
            dest.unlink()
        raise ValueError(f"Downloaded file at {dest} does not have valid GGUF header!")

    size_mb = dest.stat().st_size / (1024 * 1024)
    logger.info("Successfully downloaded and verified %s (%.1f MB).", dest, size_mb)
    return dest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download Qwen 2.5 0.5B GGUF model")
    parser.add_argument("--url", default=DEFAULT_GGUF_URL, help="Download URL")
    parser.add_argument("--output", default=str(DEFAULT_GGUF_PATH), help="Output destination path")
    parser.add_argument("--force", action="store_true", help="Force re-download")
    args = parser.parse_args()

    download_slm_model(url=args.url, dest_path=args.output, force=args.force)

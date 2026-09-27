"""Verify downloaded GGUF file integrity and GGUF header magic."""

from pathlib import Path
import pytest
from scripts.download_slm import verify_gguf_header, DEFAULT_GGUF_PATH


def test_verify_gguf_nonexistent_file(tmp_path):
    fake_path = tmp_path / "nonexistent.gguf"
    assert not verify_gguf_header(fake_path)


def test_verify_gguf_corrupted_header(tmp_path):
    corrupt_file = tmp_path / "corrupt.gguf"
    corrupt_file.write_bytes(b"NOT_A_GGUF_HEADER_12345678")
    assert not verify_gguf_header(corrupt_file)


def test_verify_gguf_valid_magic(tmp_path):
    valid_file = tmp_path / "valid.gguf"
    # GGUF magic is b'GGUF' followed by version 3 uint32 (3, 0, 0, 0)
    valid_file.write_bytes(b"GGUF\x03\x00\x00\x00")
    assert verify_gguf_header(valid_file)

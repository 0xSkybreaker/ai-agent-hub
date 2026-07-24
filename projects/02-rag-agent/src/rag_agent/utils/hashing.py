"""Content hashing utilities for document change detection."""

from __future__ import annotations

import hashlib
from pathlib import Path


def compute_file_hash(file_path: str | Path, algorithm: str = "sha256") -> str:
    """Compute the hash of a file's contents.

    Args:
        file_path: Path to the file.
        algorithm: Hash algorithm to use (sha256, md5, etc.).

    Returns:
        Hex-encoded hash string.
    """
    file_path = Path(file_path)
    hasher = hashlib.new(algorithm)
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def compute_text_hash(text: str, algorithm: str = "sha256") -> str:
    """Compute the hash of a text string.

    Args:
        text: The text to hash.
        algorithm: Hash algorithm to use.

    Returns:
        Hex-encoded hash string.
    """
    return hashlib.new(algorithm, text.encode("utf-8")).hexdigest()

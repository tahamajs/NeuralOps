"""
File hash computation and verification.

Uses SHA‑256 by default; the algorithm can be changed if needed.
"""

import hashlib
from pathlib import Path
from typing import Optional

def compute_file_hash(file_path: str, algorithm: str = "sha256") -> str:
    """
    Compute the hex digest of a file.

    Args:
        file_path: path to the file.
        algorithm: hash algorithm (``sha256``, ``sha512``, ``blake2b``).

    Returns:
        hex digest string.
    """
    h = hashlib.new(algorithm)
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_file_hash(file_path: str, expected_hash: str,
                     algorithm: str = "sha256") -> bool:
    """
    Verify that the file's hash matches the expected value.

    Returns:
        ``True`` if they match, ``False`` otherwise.
    """
    actual = compute_file_hash(file_path, algorithm)
    return actual == expected_hash
#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Version Synchronization Utility

Propagates the canonical version from neuralops/__init__.py to all metadata files
including CITATION.cff and pyproject.toml according to AGENTS.md §14.
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent.parent


def get_canonical_version() -> str:
    init_path = ROOT_DIR / "neuralops" / "__init__.py"
    content = init_path.read_text(encoding="utf-8")
    match = re.search(r'^__version__\s*=\s*["\']([^"\']+)["\']', content, re.M)
    if not match:
        raise ValueError(f"Could not find __version__ in {init_path}")
    return match.group(1)


def sync_citation_cff(version: str) -> bool:
    citation_path = ROOT_DIR / "CITATION.cff"
    if not citation_path.exists():
        return False
    content = citation_path.read_text(encoding="utf-8")
    new_content = re.sub(r"^version:\s*.*$", f"version: {version}", content, flags=re.M)
    if new_content != content:
        citation_path.write_text(new_content, encoding="utf-8")
        return True
    return False


def main() -> None:
    version = get_canonical_version()
    print(f"Canonical version: {version}")
    if sync_citation_cff(version):
        print(f"Updated CITATION.cff -> {version}")
    else:
        print("CITATION.cff is already up to date.")


if __name__ == "__main__":
    main()

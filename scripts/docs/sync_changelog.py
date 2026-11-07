#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Changelog Synchronization Utility (F-73).

Synchronizes the root CHANGELOG.md into docs/changelog.md for MkDocs documentation.
Supports --check mode for CI quality gates.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
ROOT_CHANGELOG = ROOT_DIR / "CHANGELOG.md"
DOCS_CHANGELOG = ROOT_DIR / "docs" / "changelog.md"


def sync_changelog(check: bool = False) -> int:
    """Synchronize or verify synchronization of CHANGELOG.md and docs/changelog.md."""
    if not ROOT_CHANGELOG.exists():
        print(f"Error: Root changelog not found at {ROOT_CHANGELOG}", file=sys.stderr)
        return 1

    root_content = ROOT_CHANGELOG.read_text(encoding="utf-8")

    if check:
        if not DOCS_CHANGELOG.exists():
            print(f"Error: {DOCS_CHANGELOG} does not exist. Run sync_changelog.py to generate it.", file=sys.stderr)
            return 1
        docs_content = DOCS_CHANGELOG.read_text(encoding="utf-8")
        if root_content != docs_content:
            print("Error: docs/changelog.md is out of sync with root CHANGELOG.md.", file=sys.stderr)
            print("Run `python scripts/docs/sync_changelog.py` to synchronize them.", file=sys.stderr)
            return 1
        print("docs/changelog.md is in sync with root CHANGELOG.md.")
        return 0

    DOCS_CHANGELOG.parent.mkdir(parents=True, exist_ok=True)
    DOCS_CHANGELOG.write_text(root_content, encoding="utf-8")
    print(f"Synchronized {ROOT_CHANGELOG} -> {DOCS_CHANGELOG}")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Synchronize root CHANGELOG.md to docs/changelog.md")
    parser.add_argument(
        "--check",
        action="store_true",
        help="Check whether docs/changelog.md is in sync without modifying it",
    )
    args = parser.parse_args()
    sys.exit(sync_changelog(check=args.check))


if __name__ == "__main__":
    main()

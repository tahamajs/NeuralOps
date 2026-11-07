#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
CI Test Layout Guard.

Enforces that every top-level neuralops subsystem package has a corresponding
test directory under tests/<subsystem>/ containing at least one test file.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

# Explicit justified exceptions where a package does not require tests/<pkg>/
ALLOWED_EXCEPTIONS: set[str] = set()


def check_test_layout(repo_root: Path) -> tuple[bool, list[str]]:
    """Verify that every neuralops subpackage has a non-empty tests/<name>/ directory."""
    pkg_dir = repo_root / "neuralops"
    tests_dir = repo_root / "tests"

    if not pkg_dir.is_dir() or not tests_dir.is_dir():
        return False, [f"Invalid directory structure: neuralops/ or tests/ missing in {repo_root}"]

    errors: list[str] = []

    subpackages = sorted(
        [
            d.name
            for d in pkg_dir.iterdir()
            if d.is_dir() and not d.name.startswith((".", "__")) and (d / "__init__.py").exists()
        ]
    )

    for subpkg in subpackages:
        if subpkg in ALLOWED_EXCEPTIONS:
            continue

        test_subdir = tests_dir / subpkg
        if not test_subdir.is_dir():
            errors.append(f"Missing test directory: tests/{subpkg}/ for package neuralops/{subpkg}/")
            continue

        test_files = list(test_subdir.glob("test_*.py"))
        if not test_files:
            errors.append(f"Test directory tests/{subpkg}/ contains no test_*.py files")

    return len(errors) == 0, errors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Check NeuralOps test layout compliance.")
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[2],
        help="Repository root directory",
    )
    args = parser.parse_args(argv)

    ok, errors = check_test_layout(args.root)
    if ok:
        print("[SUCCESS] All NeuralOps subsystems have matching non-empty test directories.")
        return 0

    print("[FAILURE] Test layout policy violations detected:", file=sys.stderr)
    for err in errors:
        print(f"  - {err}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())

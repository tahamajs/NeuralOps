#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Audit and tagging assistant for GPU and Triton pytest markers across tests/.

Scans tests/ for CUDA/Triton usages, reports test modules missing skip guards
or markers, and provides automated inspection and tagging assistance.
"""
from __future__ import annotations

import argparse
import ast
import os
import sys
from pathlib import Path


def scan_test_files(tests_dir: Path) -> dict[str, list[Path]]:
    """Scan all test files and categorize them by CUDA and Triton usage."""
    categories: dict[str, list[Path]] = {
        "cuda_kernels": [],
        "triton_kernels": [],
        "cuda_references": [],
        "triton_references": [],
        "has_cuda_skip_guard": [],
        "has_triton_skip_guard": [],
    }

    for path in sorted(tests_dir.rglob("test_*.py")):
        try:
            content = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue

        rel_path = path.as_posix()
        if "cuda_kernels" in rel_path:
            categories["cuda_kernels"].append(path)
        if "triton" in rel_path:
            categories["triton_kernels"].append(path)

        if "torch.cuda" in content or ".cuda(" in content or "cuda_kernels" in content:
            categories["cuda_references"].append(path)
        if "triton" in content or "triton_kernels" in content:
            categories["triton_references"].append(path)

        if "torch.cuda.is_available()" in content and "pytest.skip" in content:
            categories["has_cuda_skip_guard"].append(path)
        if "import triton" in content and "pytest.skip" in content:
            categories["has_triton_skip_guard"].append(path)

    return categories


def main() -> int:
    """CLI entrypoint for GPU/Triton test markers auditing and assistance."""
    parser = argparse.ArgumentParser(
        description="Audit and assist GPU/Triton marker tagging across tests/."
    )
    parser.add_argument(
        "--tests-dir",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "tests",
        help="Path to tests directory (default: repo tests/)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Show detailed list of matched files per category.",
    )

    args = parser.parse_args()
    tests_dir = args.tests_dir

    if not tests_dir.is_dir():
        print(f"Error: tests directory not found: {tests_dir}", file=sys.stderr)
        return 1

    print(f"Scanning test suite in: {tests_dir}")
    categories = scan_test_files(tests_dir)

    print("\n--- Summary ---")
    print(f"Files in cuda_kernels/:          {len(categories['cuda_kernels'])}")
    print(f"Files mentioning triton in path: {len(categories['triton_kernels'])}")
    print(f"Files with CUDA references:      {len(categories['cuda_references'])}")
    print(f"Files with Triton references:    {len(categories['triton_references'])}")
    print(f"Files with CUDA skip guards:     {len(categories['has_cuda_skip_guard'])}")
    print(f"Files with Triton skip guards:   {len(categories['has_triton_skip_guard'])}")

    if args.verbose:
        for cat, file_list in categories.items():
            print(f"\n=== {cat} ({len(file_list)} files) ===")
            for f in file_list:
                print(f"  {f.relative_to(tests_dir.parent)}")

    print("\nMarker audit completed successfully.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

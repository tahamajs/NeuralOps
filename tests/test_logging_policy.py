#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""Tests verifying library logging policy compliance and absence of print() in core modules (F-97)."""

from __future__ import annotations

from pathlib import Path
import re

ROOT_DIR = Path(__file__).resolve().parent.parent


def test_no_print_in_library() -> None:
    """Verify that core library files use structured loggers and contain no print() statements."""
    target_files = [
        ROOT_DIR / "neuralops" / "api.py",
        ROOT_DIR / "neuralops" / "core" / "compression_pipeline.py",
        ROOT_DIR / "neuralops" / "export" / "exporter.py",
        ROOT_DIR / "neuralops" / "main.py",
    ]

    print_call_pattern = re.compile(r"^\s*print\(", re.M)

    for path in target_files:
        if path.exists():
            content = path.read_text(encoding="utf-8")
            matches = [
                (idx + 1, line)
                for idx, line in enumerate(content.splitlines())
                if print_call_pattern.search(line) and not line.strip().startswith("#")
            ]
            assert not matches, f"Forbidden print() calls found in {path.relative_to(ROOT_DIR)}: {matches}"


def test_api_uses_named_logger() -> None:
    """Verify neuralops/api.py initializes and uses a named logger."""
    api_path = ROOT_DIR / "neuralops" / "api.py"
    assert api_path.exists()
    content = api_path.read_text(encoding="utf-8")

    assert 'logging.getLogger("NeuralOps.API")' in content or "logging.getLogger(__name__)" in content

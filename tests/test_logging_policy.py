#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""Tests verifying codebase logging policy and zero print statements in library code (F-97)."""

from __future__ import annotations

from pathlib import Path
import re
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent


def test_no_print_in_library() -> None:
    """Verify that core library files do not contain print() calls (only logging is permitted)."""
    library_files: list[Path] = [
        REPO_ROOT / "neuralops" / "api.py",
        REPO_ROOT / "neuralops" / "core" / "compression_pipeline.py",
        REPO_ROOT / "neuralops" / "export" / "exporter.py",
    ]

    violations: list[str] = []
    print_pattern = re.compile(r"^\s*print\(", re.M)

    for fpath in library_files:
        assert fpath.exists(), f"File missing: {fpath}"
        content = fpath.read_text(encoding="utf-8")
        matches = print_pattern.findall(content)
        if matches:
            violations.append(f"{fpath.name}: {len(matches)} print() call(s) found")

    assert not violations, "Forbidden print() statements found in library code:\n" + "\n".join(violations)


def test_api_uses_named_logger() -> None:
    """Verify that neuralops/api.py initializes and uses a dedicated named logger."""
    api_path = REPO_ROOT / "neuralops" / "api.py"
    assert api_path.exists(), "neuralops/api.py missing"
    content = api_path.read_text(encoding="utf-8")

    assert re.search(r'logging\.getLogger\(["\']NeuralOps\.API["\']\)', content) or "logging.getLogger(__name__)" in content, (
        "neuralops/api.py must initialize a named logger ('NeuralOps.API' or __name__)"
    )
    # Verify that logger is actively invoked in api.py
    logger_calls = re.findall(r"logger\.(info|debug|warning|error)\(", content)
    assert len(logger_calls) >= 10, f"Expected at least 10 logger invocations in api.py, found {len(logger_calls)}"

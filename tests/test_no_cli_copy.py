#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""Guard test ensuring no CLI copy exists in the test tree (F-76)."""

from __future__ import annotations

from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent


def test_no_cli_copy_in_tests() -> None:
    """Verify that tests/benchmarks/cli.py does not exist."""
    cli_copy = ROOT_DIR / "tests" / "benchmarks" / "cli.py"
    assert not cli_copy.exists(), f"Found forbidden CLI copy in tests: {cli_copy}"

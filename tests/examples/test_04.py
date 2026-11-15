#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Tests for examples/04_knowledge_distillation.py.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

EXAMPLE_PATH = Path(__file__).resolve().parent.parent.parent / "examples" / "04_knowledge_distillation.py"


def test_04_runs_clean():
    """Verify example 04 executes cleanly and exits with code 0."""
    result = subprocess.run(
        [sys.executable, str(EXAMPLE_PATH)],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, f"Example 04 failed with stderr:\n{result.stderr}"
    assert "[PASS] Overall Example 04 Status" in result.stdout


def test_04_no_inline_pillar_reimplementations():
    """Verify example 04 contains no inline reimplemented def pillar... functions."""
    content = EXAMPLE_PATH.read_text(encoding="utf-8")
    for line in content.splitlines():
        if line.strip().startswith("def pillar") or line.strip().startswith("def loss_pillar"):
            assert False, f"Forbidden inline pillar function detected: {line}"


def test_04_uses_library_multi_level_kd():
    """Verify example 04 imports MultiLevelKDTrainer from neuralops."""
    content = EXAMPLE_PATH.read_text(encoding="utf-8")
    assert "MultiLevelKDTrainer" in content, "Example 04 must use MultiLevelKDTrainer from library"

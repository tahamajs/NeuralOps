#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Tests for examples/26_streaming_shadow_twin_distillation.py.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

EXAMPLE_PATH = Path(__file__).resolve().parent.parent.parent / "examples" / "26_streaming_shadow_twin_distillation.py"


def test_26_runs_clean():
    """Verify example 26 executes cleanly with exit code 0."""
    result = subprocess.run(
        [sys.executable, str(EXAMPLE_PATH), "--device", "cpu"],
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert result.returncode == 0, f"Example 26 failed with stderr:\n{result.stderr}\nstdout:\n{result.stdout}"
    assert "[PASS] Overall Example 26 Status" in result.stdout
    assert "[PASS] Composite Distillation Loss Finite Check" in result.stdout
    assert "[PASS] Student Trainable Gradients Preserved" in result.stdout
    assert "[PASS] Teacher Frozen Invariant Preserved" in result.stdout


def test_26_distillation_invariants():
    """Verify distillation setup uses MicroChunkedShadowTwinDistiller and streaming CKA."""
    content = EXAMPLE_PATH.read_text(encoding="utf-8")
    assert "MicroChunkedShadowTwinDistiller" in content
    assert "distiller.step" in content

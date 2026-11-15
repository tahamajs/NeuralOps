#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Tests for examples/23_universal_surgery_director_ten_laws.py.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

EXAMPLE_PATH = Path(__file__).resolve().parent.parent.parent / "examples" / "23_universal_surgery_director_ten_laws.py"


def test_23_runs_clean_cpu():
    """Verify Example 23 runs with exit code 0 on CPU."""
    proc = subprocess.run(
        [sys.executable, str(EXAMPLE_PATH), "--device", "cpu"],
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert proc.returncode == 0, f"Example 23 failed with stderr:\n{proc.stderr}"
    assert "[PASS] Overall Example 23 Status" in proc.stdout
    assert "Ten Laws Certification Score" in proc.stdout
    assert "Law 10 Master Integration" in proc.stdout


def test_23_certifies_all_ten_laws():
    """Verify all 10 laws are dynamically evaluated and present in output."""
    proc = subprocess.run(
        [sys.executable, str(EXAMPLE_PATH), "--device", "cpu"],
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert proc.returncode == 0
    law_markers = [
        "Law 1 QK Norm",
        "Law 2 Softcapping",
        "Law 3 Zero Bleed Padding",
        "Law 4 Static Cross Attention",
        "Law 5 RoPE Pre Projection",
        "Law 6 Decoupled Latent Router",
        "Law 7 Causal Monotonicity State Decay",
        "Law 8 Tied Embeddings",
        "Law 9 FP32 Accumulation",
        "Law 10 Master Integration",
    ]
    for marker in law_markers:
        assert marker in proc.stdout, f"Missing {marker} in certification output"


def test_23_no_hardcoded_pass():
    """Verify example 23 computes its pass status dynamically without hardcoded bypasses."""
    content = EXAMPLE_PATH.read_text(encoding="utf-8")
    assert 'pass_fail("Overall Example 23 Status", True' not in content
    assert "all_laws_certified" in content

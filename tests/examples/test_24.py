#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Tests for examples/24_deepseek_mla_surgery_and_healing.py.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

EXAMPLE_PATH = Path(__file__).resolve().parent.parent.parent / "examples" / "24_deepseek_mla_surgery_and_healing.py"


def test_24_runs_clean_cpu():
    """Verify Example 24 runs cleanly on CPU with returncode 0."""
    proc = subprocess.run(
        [sys.executable, str(EXAMPLE_PATH), "--device", "cpu"],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc.returncode == 0, f"Example 24 failed with stderr:\n{proc.stderr}"
    assert "[PASS] Overall Example 24 Status" in proc.stdout
    assert "[PASS] Chunk Prefill Finite Check" in proc.stdout
    assert "[PASS] Decode Step Finite Check" in proc.stdout
    assert "[PASS] Trainable Parameters Gradient Check" in proc.stdout


def test_24_compression_telemetry_and_shapes():
    """Verify MLA compression ratios and KV shapes are dynamically checked."""
    proc = subprocess.run(
        [sys.executable, str(EXAMPLE_PATH), "--device", "cpu"],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc.returncode == 0
    assert "[PASS] MLA Latent Compression Ratio (512 vs 4096 dims)" in proc.stdout
    assert "87.50%" in proc.stdout
    assert "[PASS] Compressed KV Cache Tensor Shapes" in proc.stdout


def test_24_no_hardcoded_pass():
    """Verify example 24 computes status dynamically from real tensors and ratios."""
    content = EXAMPLE_PATH.read_text(encoding="utf-8")
    assert 'pass_fail("Overall Example 24 Status", True' not in content
    assert "DeepSeekMLASurgeryBridge" in content

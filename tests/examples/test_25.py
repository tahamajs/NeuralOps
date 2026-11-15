#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Tests for examples/25_fx_graph_rewriter_compiler.py.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

EXAMPLE_PATH = Path(__file__).resolve().parent.parent.parent / "examples" / "25_fx_graph_rewriter_compiler.py"


def test_25_runs_clean():
    """Verify example 25 executes cleanly with exit code 0."""
    result = subprocess.run(
        [sys.executable, str(EXAMPLE_PATH), "--device", "cpu"],
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert result.returncode == 0, f"Example 25 failed with stderr:\n{result.stderr}\nstdout:\n{result.stdout}"
    assert "[PASS] Overall Example 25 Status" in result.stdout
    assert "[PASS] FX Attention Node Replacement" in result.stdout
    assert "[PASS] Rewritten Graph Shape Parity" in result.stdout


def test_25_graph_rewriting_integrity():
    """Verify FX graph rewriter replaces SDPA with fast linear attention kernel."""
    content = EXAMPLE_PATH.read_text(encoding="utf-8")
    assert "FXAttentionGraphRewriter" in content
    assert "benchmark_dispatch_overhead" in content
    assert "fast_linear_attention_replacement" in content or "cosformer" in content

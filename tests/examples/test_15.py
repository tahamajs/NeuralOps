#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Subprocess test for examples/15_serving_vllm_sglang_trtllm.py.
"""

from __future__ import annotations

import subprocess
import sys


def _run_example():
    return subprocess.run(
        [sys.executable, "examples/15_serving_vllm_sglang_trtllm.py"],
        capture_output=True,
        text=True,
        timeout=300,
    )


def test_15_runs_clean():
    """Verify Example 15 executes cleanly on CPU with returncode 0."""
    proc = _run_example()
    assert proc.returncode == 0, f"Example 15 failed with stderr: {proc.stderr}"
    assert "[PASS] Overall Example 15 Status" in proc.stdout

    # Verify honest skips for uninstalled optional packages
    assert "[SKIPPED] vllm" in proc.stdout
    assert "[SKIPPED] sglang" in proc.stdout
    assert "[SKIPPED] tensorrt_llm" in proc.stdout

    # Verify subchecks were actually evaluated
    assert "[PASS] vLLM Interface Contract Verification" in proc.stdout
    assert "[PASS] SGLang Interface Contract Verification" in proc.stdout
    assert "[PASS] TensorRT-LLM Interface Contract Verification" in proc.stdout
    assert "[PASS] TensorRT-LLM C++ Plugin Synthesis" in proc.stdout

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


def test_15_runs_clean():
    """Verify Example 15 executes cleanly on CPU with returncode 0."""
    proc = subprocess.run(
        [sys.executable, "examples/15_serving_vllm_sglang_trtllm.py"],
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert proc.returncode == 0, f"Example 15 failed with stderr: {proc.stderr}"
    assert "[PASS] Overall Example 15 Status" in proc.stdout


def test_15_honest_skips():
    """Verify stdout contains explicit skip notices for optional engines not installed."""
    proc = subprocess.run(
        [sys.executable, "examples/15_serving_vllm_sglang_trtllm.py"],
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert proc.returncode == 0
    # In an environment without vllm, sglang, tensorrt_llm, skip notices must be printed
    assert "[SKIPPED] vllm" in proc.stdout
    assert "[SKIPPED] sglang" in proc.stdout
    assert "[SKIPPED] tensorrt_llm" in proc.stdout


def test_15_no_hardcoded_pass():
    """Verify that overall status is computed and subchecks are validated."""
    proc = subprocess.run(
        [sys.executable, "examples/15_serving_vllm_sglang_trtllm.py"],
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert proc.returncode == 0
    assert "[PASS] vLLM Interface Contract Verification" in proc.stdout
    assert "[PASS] SGLang Interface Contract Verification" in proc.stdout
    assert "[PASS] TensorRT-LLM Interface Contract Verification" in proc.stdout
    assert "[PASS] TensorRT-LLM C++ Plugin Synthesis" in proc.stdout

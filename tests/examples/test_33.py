#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Tests for examples/33_triton_26_hardware_kernels.py.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import torch

import neuralops.ops.triton as notr

EXAMPLE_PATH = Path(__file__).resolve().parent.parent.parent / "examples" / "33_triton_26_hardware_kernels.py"


def test_33_runs_clean_cpu():
    """Verify Example 33 runs cleanly via subprocess on CPU."""
    proc = subprocess.run(
        [sys.executable, str(EXAMPLE_PATH), "--device", "cpu"],
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert proc.returncode == 0, f"Example 33 failed with stderr:\n{proc.stderr}\nstdout:\n{proc.stdout}"
    assert "[PASS] Overall Example 33 Status" in proc.stdout
    assert "Linear Attention" in proc.stdout
    assert "Selective State Update" in proc.stdout
    assert "Differential Attention" in proc.stdout


def test_33_kernel_catalog_populated():
    """Verify Triton kernel catalog contains all dynamically registered operators."""
    catalog = getattr(notr, "TRITON_KERNEL_CATALOG", [])
    assert len(catalog) >= 26, f"Expected at least 26 kernels in catalog, found {len(catalog)}"


def test_33_linear_and_differential_attention_fallback():
    """Verify mathematical output shapes and numerical finiteness of reference fallbacks."""
    B, H, N, D = 2, 4, 8, 16
    q = torch.randn(B, H, N, D)
    k = torch.randn(B, H, N, D)
    v = torch.randn(B, H, N, D)

    # Linear Attention reference fallback
    o_lin = notr.linear_attn_pytorch_fallback(q, k, v)
    assert o_lin.shape == (B, H, N, D)
    assert torch.isfinite(o_lin).all()

    # Differential Attention reference fallback
    o_diff = notr.triton_differential_attention_pytorch_ref(q, k, q, k, v, lambda_val=0.8)
    assert o_diff.shape == (B, H, N, D)
    assert torch.isfinite(o_diff).all()

#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Subprocess test for examples/18_three_level_kernel_dispatch.py.
"""

from __future__ import annotations

import subprocess
import sys
import torch

from neuralops import force_kernel_backend
from neuralops.ops.cuda import cosformer_linear_attention


def test_18_runs_clean():
    """Verify that example 18 executes cleanly via subprocess and exits 0."""
    proc = subprocess.run(
        [sys.executable, "examples/18_three_level_kernel_dispatch.py"],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc.returncode == 0, f"Example 18 failed with stderr: {proc.stderr}"
    assert "[PASS] Overall Example 18 Status" in proc.stdout


def test_18_dispatch_parity():
    """Verify Level 2 (Triton) and Level 3 (PyTorch Reference) output parity."""
    torch.manual_seed(42)
    b, h, seq_len, d_head = 2, 4, 16, 32
    q = torch.randint(1, 10, (b, h, seq_len, d_head), dtype=torch.float32) / 10.0
    k = torch.randint(1, 10, (b, h, seq_len, d_head), dtype=torch.float32) / 10.0
    v = torch.randint(1, 10, (b, h, seq_len, d_head), dtype=torch.float32) / 10.0

    with force_kernel_backend("pytorch"):
        out_pt = cosformer_linear_attention(q, k, v)

    with force_kernel_backend("triton"):
        out_tr = cosformer_linear_attention(q, k, v)

    assert torch.isfinite(out_pt).all(), "PyTorch reference produced non-finite values"
    assert torch.isfinite(out_tr).all(), "Triton kernel produced non-finite values"
    assert torch.allclose(out_pt, out_tr, atol=1e-4), "Parity mismatch between Triton and PyTorch backends"

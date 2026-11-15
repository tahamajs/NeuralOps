#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Subprocess and mathematical tests for examples/07_custom_attention.py.
"""

from __future__ import annotations

import importlib
import math
import subprocess
import sys
import torch
import torch.nn as nn

from examples._shared import tiny_inputs

ex07 = importlib.import_module("examples.07_custom_attention")
MyCausalLinearAttention = ex07.MyCausalLinearAttention
MySVDVariantCompressor = ex07.MySVDVariantCompressor


def test_07_runs_clean():
    """Verify that example 07 executes cleanly via subprocess and exits 0."""
    proc = subprocess.run(
        [sys.executable, "examples/07_custom_attention.py"],
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert proc.returncode == 0, f"Example 07 failed with stderr:\n{proc.stderr}\nstdout:\n{proc.stdout}"
    assert "[PASS] Overall Example 07 Status" in proc.stdout


def test_07_attention_mask_enforced():
    """Verify causal attention mask mathematically blocks future tokens."""
    embed_dim = 32
    num_heads = 2
    seq_len = 12
    batch_size = 2

    module = MyCausalLinearAttention(embed_dim=embed_dim, num_heads=num_heads).eval()
    inputs = tiny_inputs(batch_size=batch_size, seq_len=seq_len)
    with torch.no_grad():
        x1 = nn.Embedding(128, embed_dim)(inputs["input_ids"])
    x2 = x1.clone()

    target_t = 3
    future_t = target_t + 5
    x2[:, future_t, :] += 10.0

    with torch.no_grad():
        out1, _ = module(x1, is_causal=True)
        out2, _ = module(x2, is_causal=True)

    diff_past = (out1[:, target_t, :] - out2[:, target_t, :]).abs().max().item()
    diff_future = (out1[:, future_t, :] - out2[:, future_t, :]).abs().max().item()

    assert diff_past == 0.0, f"Future token perturbation leaked to past token: diff={diff_past}"
    assert diff_future > 0.0, "Future token output was not affected by perturbation"


def test_07_custom_attention_forward_backward():
    """Verify forward and backward pass for registered custom attention module."""
    embed_dim = 32
    num_heads = 2
    seq_len = 8
    batch_size = 2

    module = MyCausalLinearAttention(embed_dim=embed_dim, num_heads=num_heads)
    x = torch.ones(batch_size, seq_len, embed_dim, requires_grad=True)

    out, final_state = module(x, is_causal=True)
    assert out.shape == (batch_size, seq_len, embed_dim)
    assert final_state is not None
    assert torch.isfinite(out).all()

    loss = out.sum()
    loss.backward()

    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
    for name, param in module.named_parameters():
        assert param.grad is not None, f"Gradient missing for {name}"
        assert torch.isfinite(param.grad).all(), f"Gradient non-finite for {name}"

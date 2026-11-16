#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Unit and regression tests for adaptive_span CUDA operator and fallbacks.
Tests compliance with torch.amp, CPU fallbacks, gradient flow, and parameter validation.
"""

from __future__ import annotations

import pytest
import torch

from neuralops.ops.cuda.adaptive_span import adaptive_span, fused_adaptive_span_penalty_pytorch


def test_adaptive_span_cpu_fallback_forward() -> None:
    """Test adaptive_span CPU fallback produces exact matching outputs with PyTorch reference."""
    H = 4
    B = 2
    N_q = 16
    N_k = 16
    ramp_size = 8.0

    spans = torch.tensor([4.0, 8.0, 12.0, 16.0], dtype=torch.float32)
    q_pos = torch.arange(N_q, dtype=torch.float32).unsqueeze(0).expand(B, -1).contiguous()
    k_pos = torch.arange(N_k, dtype=torch.float32).unsqueeze(0).expand(B, -1).contiguous()

    # Causal
    out_causal = adaptive_span(spans, q_pos, k_pos, ramp_size=ramp_size, is_causal=True)
    ref_causal = fused_adaptive_span_penalty_pytorch(spans, q_pos, k_pos, ramp_size=ramp_size, is_causal=True)

    assert out_causal.shape == (B, H, N_q, N_k)
    assert torch.isfinite(out_causal).all()
    assert torch.allclose(out_causal, ref_causal, atol=1e-6, rtol=1e-5)

    # Non-causal / bidirectional
    out_noncausal = adaptive_span(spans, q_pos, k_pos, ramp_size=ramp_size, is_causal=False)
    ref_noncausal = fused_adaptive_span_penalty_pytorch(spans, q_pos, k_pos, ramp_size=ramp_size, is_causal=False)

    assert out_noncausal.shape == (B, H, N_q, N_k)
    assert torch.isfinite(out_noncausal).all()
    assert torch.allclose(out_noncausal, ref_noncausal, atol=1e-6, rtol=1e-5)


def test_adaptive_span_gradient_flow() -> None:
    """Test analytical/autograd gradient flow through spans, q_pos, and k_pos."""
    H = 2
    B = 1
    N_q = 8
    N_k = 8
    ramp_size = 4.0

    spans = torch.tensor([4.0, 6.0], dtype=torch.float32, requires_grad=True)
    q_pos = torch.arange(N_q, dtype=torch.float32).unsqueeze(0).requires_grad_(True)
    k_pos = torch.arange(N_k, dtype=torch.float32).unsqueeze(0).requires_grad_(True)

    out = adaptive_span(spans, q_pos, k_pos, ramp_size=ramp_size, is_causal=True)
    loss = out.sum()
    loss.backward()

    assert spans.grad is not None
    assert torch.isfinite(spans.grad).all()
    assert (spans.grad != 0.0).any(), "Gradient to spans should be non-zero in ramp region"


def test_adaptive_span_amp_compliance() -> None:
    """Verify adaptive_span behaves properly under torch.amp autocast contexts."""
    H = 2
    B = 1
    N_q = 8
    N_k = 8
    ramp_size = 4.0

    spans = torch.tensor([4.0, 6.0], dtype=torch.float32)
    q_pos = torch.arange(N_q, dtype=torch.float32).unsqueeze(0)
    k_pos = torch.arange(N_k, dtype=torch.float32).unsqueeze(0)

    # Autocast on CPU with bfloat16
    with torch.amp.autocast("cpu", dtype=torch.bfloat16):
        out = adaptive_span(spans, q_pos, k_pos, ramp_size=ramp_size, is_causal=True)
        assert out.shape == (B, H, N_q, N_k)
        assert torch.isfinite(out).all()


def test_adaptive_span_shape_and_dtype_variations() -> None:
    """Test adaptive_span with different sequence lengths and float dtypes."""
    for dtype in [torch.float32, torch.bfloat16]:
        spans = torch.tensor([2.0, 4.0], dtype=dtype)
        q_pos = torch.arange(10, dtype=dtype).unsqueeze(0)
        k_pos = torch.arange(12, dtype=dtype).unsqueeze(0)

        out = adaptive_span(spans, q_pos, k_pos, ramp_size=3.0, is_causal=True)
        assert out.shape == (1, 2, 10, 12)
        assert torch.isfinite(out).all()

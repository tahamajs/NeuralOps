#!/usr/bin/env python3
"""
Comprehensive Unit Test Suite for Shared Triton Kernels & Unified Dispatcher.

Validates:
  1. linear_attn (causal, non-causal, phi functions, backward gradients)
  2. sparse_attn (block-sparse mask, forward/backward)
  3. low_rank (analytical backward, LoRA-style projection)
  4. state_update (SSM recurrence forward/backward)
  5. gqa (grouped query repeat/expansion)
  6. triton_dispatch (correct execution or graceful PyTorch fallback)
"""
from __future__ import annotations

import math

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from neuralops.core.triton_kernels import (HAS_TRITON, gqa,
                                           gqa_pytorch_fallback, linear_attn,
                                           linear_attn_pytorch_fallback,
                                           low_rank, low_rank_pytorch_fallback,
                                           sparse_attn,
                                           sparse_attn_pytorch_fallback,
                                           state_update,
                                           state_update_pytorch_fallback,
                                           triton_dispatch)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if torch.cuda.is_available() else torch.float32


# ---------------------------------------------------------------------------
# 1. GQA Tests
# ---------------------------------------------------------------------------
def test_gqa_fwd_and_fallback():
    B, H_kv, S, D = 2, 2, 16, 32
    H_q = 8
    k = torch.randn(B, H_kv, S, D, device=DEVICE, dtype=DTYPE)
    v = torch.randn(B, H_kv, S, D, device=DEVICE, dtype=DTYPE)

    k_exp, v_exp = gqa.fwd(k, v, num_kv_heads=H_kv, num_heads=H_q)
    assert k_exp.shape == (B, H_q, S, D)
    assert v_exp.shape == (B, H_q, S, D)

    # PyTorch fallback equivalence
    k_ref, v_ref = gqa_pytorch_fallback(k, v, num_kv_heads=H_kv, num_heads=H_q)
    assert torch.allclose(k_exp, k_ref, atol=1e-3, rtol=1e-3)
    assert torch.allclose(v_exp, v_ref, atol=1e-3, rtol=1e-3)


# ---------------------------------------------------------------------------
# 2. Low-Rank Projection Tests (Forward & Exact Analytical Backward)
# ---------------------------------------------------------------------------
def test_low_rank_fwd_and_analytical_bwd():
    B, S, D_in, R, D_out = 2, 16, 64, 16, 64
    x = torch.randn(B, S, D_in, device=DEVICE, dtype=DTYPE, requires_grad=True)
    U = torch.randn(D_out, R, device=DEVICE, dtype=DTYPE, requires_grad=True)
    V = torch.randn(R, D_in, device=DEVICE, dtype=DTYPE, requires_grad=True)
    bias = torch.randn(D_out, device=DEVICE, dtype=DTYPE, requires_grad=True)

    # Forward
    y = low_rank.fwd(x, U, V, bias=bias)
    assert y.shape == (B, S, D_out)

    # Reference PyTorch
    y_ref = low_rank_pytorch_fallback(x, U, V, bias=bias)
    assert torch.allclose(y, y_ref, atol=1e-2, rtol=1e-2)

    # Backward
    loss = y.sum()
    loss.backward()

    assert x.grad is not None
    assert U.grad is not None
    assert V.grad is not None
    assert bias.grad is not None
    assert torch.isfinite(x.grad).all()
    assert torch.isfinite(U.grad).all()


def test_low_rank_frozen_input_autograd_safety():
    """Ensure no autograd crash when x has requires_grad=False."""
    x_frozen = torch.randn(2, 8, 32, device=DEVICE, dtype=DTYPE, requires_grad=False)
    U = torch.randn(32, 8, device=DEVICE, dtype=DTYPE, requires_grad=True)
    V = torch.randn(8, 32, device=DEVICE, dtype=DTYPE, requires_grad=True)

    y = low_rank.fwd(x_frozen, U, V)
    loss = y.sum()
    loss.backward()
    assert U.grad is not None
    assert V.grad is not None


# ---------------------------------------------------------------------------
# 3. Linear Attention Tests
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("is_causal", [True, False])
@pytest.mark.parametrize("phi", ["elu+1", "relu", "identity"])
def test_linear_attn_fwd_bwd(is_causal, phi):
    B, H, S, D = 2, 4, 32, 32
    q = torch.randn(B, H, S, D, device=DEVICE, dtype=DTYPE, requires_grad=True)
    k = torch.randn(B, H, S, D, device=DEVICE, dtype=DTYPE, requires_grad=True)
    v = torch.randn(B, H, S, D, device=DEVICE, dtype=DTYPE, requires_grad=True)

    out = linear_attn.fwd(q, k, v, is_causal=is_causal, phi=phi)
    assert out.shape == (B, H, S, D)

    # Test backward
    loss = out.sum()
    loss.backward()
    assert q.grad is not None
    assert k.grad is not None
    assert v.grad is not None
    assert torch.isfinite(q.grad).all()


# ---------------------------------------------------------------------------
# 4. Sparse Attention Tests
# ---------------------------------------------------------------------------
def test_sparse_attn_block_mask():
    B, H, S, D = 2, 2, 64, 32
    block_size = 16
    n_blocks = S // block_size
    block_mask = torch.ones(n_blocks, n_blocks, device=DEVICE, dtype=torch.bool)
    # Causal lower-triangular block mask
    block_mask = torch.tril(block_mask)

    q = torch.randn(B, H, S, D, device=DEVICE, dtype=DTYPE, requires_grad=True)
    k = torch.randn(B, H, S, D, device=DEVICE, dtype=DTYPE, requires_grad=True)
    v = torch.randn(B, H, S, D, device=DEVICE, dtype=DTYPE, requires_grad=True)

    out = sparse_attn.fwd(q, k, v, block_mask=block_mask, block_size=block_size, is_causal=True)
    assert out.shape == (B, H, S, D)

    loss = out.sum()
    loss.backward()
    assert q.grad is not None
    assert torch.isfinite(q.grad).all()


# ---------------------------------------------------------------------------
# 5. State Update (SSM / RWKV) Tests
# ---------------------------------------------------------------------------
def test_state_update_recurrence():
    B, S, D, N = 2, 16, 32, 16
    u = torch.randn(B, S, D, device=DEVICE, dtype=DTYPE, requires_grad=True)
    delta = F.softplus(torch.randn(B, S, D, device=DEVICE, dtype=DTYPE))
    A = -torch.exp(torch.randn(D, N, device=DEVICE, dtype=DTYPE))
    B_proj = torch.randn(B, S, N, device=DEVICE, dtype=DTYPE)
    C_proj = torch.randn(B, S, N, device=DEVICE, dtype=DTYPE)
    D_res = torch.randn(D, device=DEVICE, dtype=DTYPE)

    y, new_state = state_update.fwd(u, delta, A, B_proj, C_proj, D_res)
    assert y.shape == (B, S, D)
    assert new_state.shape == (B, D, N)

    # Test backward
    loss = y.sum()
    loss.backward()
    assert u.grad is not None
    assert torch.isfinite(u.grad).all()


# ---------------------------------------------------------------------------
# 6. Unified triton_dispatch Dispatcher Tests
# ---------------------------------------------------------------------------
def test_triton_dispatch_linear_attn():
    B, H, S, D = 2, 2, 16, 32
    q = torch.randn(B, H, S, D, device=DEVICE, dtype=DTYPE)
    k = torch.randn(B, H, S, D, device=DEVICE, dtype=DTYPE)
    v = torch.randn(B, H, S, D, device=DEVICE, dtype=DTYPE)

    out = triton_dispatch("linear_attn", q, k, v, is_causal=True, phi="elu+1")
    assert out.shape == (B, H, S, D)
    assert torch.isfinite(out).all()


def test_triton_dispatch_low_rank():
    x = torch.randn(2, 8, 32, device=DEVICE, dtype=DTYPE)
    U = torch.randn(32, 8, device=DEVICE, dtype=DTYPE)
    V = torch.randn(8, 32, device=DEVICE, dtype=DTYPE)

    out = triton_dispatch("low_rank", x, U, V)
    assert out.shape == (2, 8, 32)
    assert torch.isfinite(out).all()


def test_triton_dispatch_unsupported_graceful_error():
    x = torch.randn(2, 4)
    with pytest.raises(ValueError, match="Unknown.*kernel"):
        triton_dispatch("non_existent_kernel", x)

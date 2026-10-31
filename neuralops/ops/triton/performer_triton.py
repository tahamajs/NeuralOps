#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Performer Fast Attention via Positive Orthogonal Random Features (FAVOR+) Triton Kernel

Computes causal linear attention over positive random feature projections:
O_i = (phi(Q)_i @ sum_{j<=i} phi(K)_j V_j^T) / (phi(Q)_i @ sum_{j<=i} phi(K)_j + eps)
"""

from __future__ import annotations

import logging
from typing import Optional

import torch

from neuralops.ops.triton._types import (
    FORCE_PYTORCH_FALLBACK,
    HAS_TRITON,
    check_triton_suitability,
)

logger = logging.getLogger(__name__)

if HAS_TRITON:
    import triton
    import triton.language as tl

    @triton.autotune(
        configs=[
            triton.Config({"BLOCK_M": 32, "BLOCK_D": 32}, num_warps=4, num_stages=2),
            triton.Config({"BLOCK_M": 64, "BLOCK_D": 32}, num_warps=4, num_stages=2),
            triton.Config({"BLOCK_M": 16, "BLOCK_D": 64}, num_warps=2, num_stages=2),
        ],
        key=["L", "M_DIM", "D"],
    )
    @triton.jit
    def _performer_causal_fwd_kernel(
        Q_prime_ptr, K_prime_ptr, V_ptr, Out_ptr,
        stride_qb, stride_qh, stride_ql, stride_qm,
        stride_kb, stride_kh, stride_kl, stride_km,
        stride_vb, stride_vh, stride_vl, stride_vd,
        stride_ob, stride_oh, stride_ol, stride_od,
        L, M_DIM: tl.constexpr, D: tl.constexpr, eps,
        BLOCK_M: tl.constexpr, BLOCK_D: tl.constexpr,
    ):
        off_h = tl.program_id(0)
        off_b = tl.program_id(1)

        offs_m = tl.arange(0, BLOCK_M)
        offs_d = tl.arange(0, BLOCK_D)

        mask_m = offs_m < M_DIM
        mask_d = offs_d < D

        # State accumulators: S (M_DIM x D) and z (M_DIM)
        s = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)
        z = tl.zeros([BLOCK_M], dtype=tl.float32)

        for l in range(L):
            k_ptrs = K_prime_ptr + off_b * stride_kb + off_h * stride_kh + l * stride_kl + offs_m * stride_km
            v_ptrs = V_ptr + off_b * stride_vb + off_h * stride_vh + l * stride_vl + offs_d * stride_vd

            k_val = tl.load(k_ptrs, mask=mask_m, other=0.0).to(tl.float32)
            v_val = tl.load(v_ptrs, mask=mask_d, other=0.0).to(tl.float32)

            s += k_val[:, None] * v_val[None, :]
            z += k_val

            q_ptrs = Q_prime_ptr + off_b * stride_qb + off_h * stride_qh + l * stride_ql + offs_m * stride_qm
            q_val = tl.load(q_ptrs, mask=mask_m, other=0.0).to(tl.float32)

            num = tl.sum(q_val[:, None] * s, axis=0)
            den = tl.sum(q_val * z) + eps

            out_val = num / den
            out_ptrs = Out_ptr + off_b * stride_ob + off_h * stride_oh + l * stride_ol + offs_d * stride_od
            tl.store(out_ptrs, out_val, mask=mask_d)


def performer_triton_forward(
    q_prime: torch.Tensor,
    k_prime: torch.Tensor,
    v: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Execute Performer FAVOR+ positive kernel causal linear attention."""
    is_suitable, reason = check_triton_suitability(q_prime, "performer")
    if not is_suitable:
        logger.warning(
            "Level 2 Triton fallback triggered for performer: %s. Falling back to Level 3 PyTorch reference.",
            reason,
        )
        return performer_pytorch_reference(q_prime, k_prime, v, eps=eps)

    B, H, L, M_DIM = q_prime.shape
    _, _, _, D = v.shape

    out = torch.empty((B, H, L, D), device=q_prime.device, dtype=q_prime.dtype)
    grid = (H, B)

    _performer_causal_fwd_kernel[grid](
        q_prime, k_prime, v, out,
        q_prime.stride(0), q_prime.stride(1), q_prime.stride(2), q_prime.stride(3),
        k_prime.stride(0), k_prime.stride(1), k_prime.stride(2), k_prime.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        L, M_DIM, D, eps,
    )
    return out


def performer_pytorch_reference(
    q_prime: torch.Tensor,
    k_prime: torch.Tensor,
    v: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Exact PyTorch reference implementation for causal Performer FAVOR+ attention."""
    # Prefix cumulative sum over time
    kv = torch.einsum("bhlm,bhld->bhlmd", k_prime, v)
    s = torch.cumsum(kv, dim=2)
    z = torch.cumsum(k_prime, dim=2)

    num = torch.einsum("bhlm,bhlmd->bhld", q_prime, s)
    den = torch.sum(q_prime * z, dim=-1, keepdim=True) + eps

    return num / den

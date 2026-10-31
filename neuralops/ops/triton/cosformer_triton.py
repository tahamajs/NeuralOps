#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Cosformer Linear Attention with Cosine Reweighting Triton Kernel

Computes causal linear attention with decomposed cosine positional reweighting:
S_1 = sum_{j<=i} K'_1 V^T, S_2 = sum_{j<=i} K'_2 V^T
"""

from __future__ import annotations

import logging
import math
from typing import Optional

import torch
import torch.nn.functional as F

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
            triton.Config({"BLOCK_D": 32}, num_warps=4, num_stages=2),
            triton.Config({"BLOCK_D": 64}, num_warps=4, num_stages=2),
            triton.Config({"BLOCK_D": 16}, num_warps=2, num_stages=2),
        ],
        key=["L", "D"],
    )
    @triton.jit
    def _cosformer_causal_fwd_kernel(
        Q1_ptr, Q2_ptr, K1_ptr, K2_ptr, V_ptr, Out_ptr,
        stride_qb, stride_qh, stride_ql, stride_qd,
        stride_kb, stride_kh, stride_kl, stride_kd,
        stride_vb, stride_vh, stride_vl, stride_vd,
        stride_ob, stride_oh, stride_ol, stride_od,
        L, D: tl.constexpr, eps,
        BLOCK_D: tl.constexpr,
    ):
        off_h = tl.program_id(0)
        off_b = tl.program_id(1)

        offs_d = tl.arange(0, BLOCK_D)
        mask_d = offs_d < D

        # Running KV state accumulators
        s1 = tl.zeros([BLOCK_D, BLOCK_D], dtype=tl.float32)
        s2 = tl.zeros([BLOCK_D, BLOCK_D], dtype=tl.float32)
        z1 = tl.zeros([BLOCK_D], dtype=tl.float32)
        z2 = tl.zeros([BLOCK_D], dtype=tl.float32)

        for l in range(L):
            k1_ptrs = K1_ptr + off_b * stride_kb + off_h * stride_kh + l * stride_kl + offs_d * stride_kd
            k2_ptrs = K2_ptr + off_b * stride_kb + off_h * stride_kh + l * stride_kl + offs_d * stride_kd
            v_ptrs = V_ptr + off_b * stride_vb + off_h * stride_vh + l * stride_vl + offs_d * stride_vd

            k1 = tl.load(k1_ptrs, mask=mask_d, other=0.0).to(tl.float32)
            k2 = tl.load(k2_ptrs, mask=mask_d, other=0.0).to(tl.float32)
            v = tl.load(v_ptrs, mask=mask_d, other=0.0).to(tl.float32)

            s1 += k1[:, None] * v[None, :]
            s2 += k2[:, None] * v[None, :]
            z1 += k1
            z2 += k2

            q1_ptrs = Q1_ptr + off_b * stride_qb + off_h * stride_qh + l * stride_ql + offs_d * stride_qd
            q2_ptrs = Q2_ptr + off_b * stride_qb + off_h * stride_qh + l * stride_ql + offs_d * stride_qd
            q1 = tl.load(q1_ptrs, mask=mask_d, other=0.0).to(tl.float32)
            q2 = tl.load(q2_ptrs, mask=mask_d, other=0.0).to(tl.float32)

            num = tl.sum(q1[:, None] * s1, axis=0) + tl.sum(q2[:, None] * s2, axis=0)
            den = tl.sum(q1 * z1) + tl.sum(q2 * z2) + eps

            out = num / den
            out_ptrs = Out_ptr + off_b * stride_ob + off_h * stride_oh + l * stride_ol + offs_d * stride_od
            tl.store(out_ptrs, out, mask=mask_d)


def cosformer_triton_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Execute Cosformer causal linear attention using Triton or PyTorch fallback."""
    is_suitable, reason = check_triton_suitability(q, "cosformer")
    if not is_suitable:
        logger.warning(
            "Level 2 Triton fallback triggered for cosformer: %s. Falling back to Level 3 PyTorch reference.",
            reason,
        )
        return cosformer_pytorch_reference(q, k, v, eps=eps)

    B, H, L, D = q.shape

    # Apply non-negative activation and cosine reweighting
    q_act = F.relu(q)
    k_act = F.relu(k)

    positions = torch.arange(L, device=q.device, dtype=q.dtype)
    angles = positions * (math.pi / (2.0 * L))
    cos_w = torch.cos(angles).view(1, 1, L, 1)
    sin_w = torch.sin(angles).view(1, 1, L, 1)

    q1 = q_act * cos_w
    q2 = q_act * sin_w
    k1 = k_act * cos_w
    k2 = k_act * sin_w

    out = torch.empty_like(v)
    grid = (H, B)

    _cosformer_causal_fwd_kernel[grid](
        q1, q2, k1, k2, v, out,
        q1.stride(0), q1.stride(1), q1.stride(2), q1.stride(3),
        k1.stride(0), k1.stride(1), k1.stride(2), k1.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        L, D, eps,
    )
    return out


def cosformer_pytorch_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Exact PyTorch reference implementation for causal Cosformer."""
    B, H, L, D = q.shape
    q_act = F.relu(q)
    k_act = F.relu(k)

    positions = torch.arange(L, device=q.device, dtype=q.dtype)
    angles = positions * (math.pi / (2.0 * L))
    cos_w = torch.cos(angles).view(1, 1, L, 1)
    sin_w = torch.sin(angles).view(1, 1, L, 1)

    q1 = q_act * cos_w
    q2 = q_act * sin_w
    k1 = k_act * cos_w
    k2 = k_act * sin_w

    # Causal prefix cumsum over time
    kv1 = torch.einsum("bhli,bhlj->bhlij", k1, v)
    kv2 = torch.einsum("bhli,bhlj->bhlij", k2, v)
    s1 = torch.cumsum(kv1, dim=2)
    s2 = torch.cumsum(kv2, dim=2)

    z1 = torch.cumsum(k1, dim=2)
    z2 = torch.cumsum(k2, dim=2)

    num1 = torch.einsum("bhli,bhlij->bhlj", q1, s1)
    num2 = torch.einsum("bhli,bhlij->bhlj", q2, s2)
    num = num1 + num2

    den = (torch.sum(q1 * z1, dim=-1, keepdim=True) +
           torch.sum(q2 * z2, dim=-1, keepdim=True) + eps)

    return num / den

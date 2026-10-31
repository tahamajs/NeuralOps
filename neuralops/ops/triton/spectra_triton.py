#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Spectra V2 Spectral Decay Attention Triton Kernel

Fuses scaled dot-product attention with continuous spectral exponential decay:
Scores_{i, j} = (Q_i @ K_j^T / sqrt(D)) * exp(-decay * (i - j))
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
            triton.Config({"BLOCK_M": 32, "BLOCK_N": 32}, num_warps=4, num_stages=2),
            triton.Config({"BLOCK_M": 64, "BLOCK_N": 32}, num_warps=4, num_stages=2),
            triton.Config({"BLOCK_M": 16, "BLOCK_N": 16}, num_warps=2, num_stages=2),
        ],
        key=["L", "D"],
    )
    @triton.jit
    def _spectra_causal_fwd_kernel(
        Q_ptr, K_ptr, V_ptr, Out_ptr,
        scale, decay_rate,
        stride_qb, stride_qh, stride_qm, stride_qd,
        stride_kb, stride_kh, stride_kn, stride_kd,
        stride_vb, stride_vh, stride_vn, stride_vd,
        stride_ob, stride_oh, stride_om, stride_od,
        L, D: tl.constexpr,
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    ):
        start_m = tl.program_id(0)
        off_h = tl.program_id(1)
        off_b = tl.program_id(2)

        offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_d = tl.arange(0, D)

        q_ptrs = Q_ptr + off_b * stride_qb + off_h * stride_qh + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd
        q = tl.load(q_ptrs, mask=offs_m[:, None] < L, other=0.0)

        m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
        l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
        acc = tl.zeros([BLOCK_M, D], dtype=tl.float32)

        for start_n in range(0, (start_m + 1) * BLOCK_M, BLOCK_N):
            offs_n = start_n + tl.arange(0, BLOCK_N)
            k_ptrs = K_ptr + off_b * stride_kb + off_h * stride_kh + offs_n[None, :] * stride_kn + offs_d[:, None] * stride_kd
            v_ptrs = V_ptr + off_b * stride_vb + off_h * stride_vh + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vd

            k = tl.load(k_ptrs, mask=offs_n[None, :] < L, other=0.0)
            v = tl.load(v_ptrs, mask=offs_n[:, None] < L, other=0.0)

            s = tl.dot(q, k) * scale
            # Spectral distance decay
            dist = (offs_m[:, None] - offs_n[None, :]).to(tl.float32)
            decay = tl.exp(-decay_rate * dist)
            s = s * decay

            # Causal mask
            causal_mask = offs_m[:, None] >= offs_n[None, :]
            s = tl.where(causal_mask & (offs_n[None, :] < L), s, float("-inf"))

            m_curr = tl.maximum(m_i, tl.max(s, axis=1))
            p = tl.exp(s - m_curr[:, None])
            l_curr = tl.exp(m_i - m_curr) * l_i + tl.sum(p, axis=1)

            acc_scale = tl.exp(m_i - m_curr)
            acc = acc * acc_scale[:, None] + tl.dot(p.to(v.dtype), v)

            l_i = l_curr
            m_i = m_curr

        out = acc / l_i[:, None]
        out_ptrs = Out_ptr + off_b * stride_ob + off_h * stride_oh + offs_m[:, None] * stride_om + offs_d[None, :] * stride_od
        tl.store(out_ptrs, out, mask=offs_m[:, None] < L)


def spectra_triton_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    decay_rate: float = 0.05,
    scale: Optional[float] = None,
) -> torch.Tensor:
    """Execute Spectra V2 causal spectral decay attention."""
    is_suitable, reason = check_triton_suitability(q, "spectra")
    if not is_suitable:
        logger.warning(
            "Level 2 Triton fallback triggered for spectra: %s. Falling back to Level 3 PyTorch reference.",
            reason,
        )
        return spectra_pytorch_reference(q, k, v, decay_rate=decay_rate, scale=scale)

    B, H, L, D = q.shape
    if scale is None:
        scale = 1.0 / math.sqrt(D)

    out = torch.empty_like(q)
    grid = lambda meta: (triton.cdiv(L, meta["BLOCK_M"]), H, B)

    _spectra_causal_fwd_kernel[grid](
        q, k, v, out,
        scale, decay_rate,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        L, D,
    )
    return out


def spectra_pytorch_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    decay_rate: float = 0.05,
    scale: Optional[float] = None,
) -> torch.Tensor:
    """Exact PyTorch reference implementation for Spectra V2 decay attention."""
    B, H, L, D = q.shape
    if scale is None:
        scale = 1.0 / math.sqrt(D)

    scores = torch.matmul(q, k.transpose(-1, -2)) * scale

    # Spectral exponential decay over token distances
    i_indices = torch.arange(L, device=q.device).view(-1, 1)
    j_indices = torch.arange(L, device=q.device).view(1, -1)
    dist = torch.clamp(i_indices - j_indices, min=0).float()
    decay = torch.exp(-decay_rate * dist).view(1, 1, L, L)

    scores = scores * decay
    causal_mask = torch.triu(torch.full((L, L), float("-inf"), device=q.device), diagonal=1)
    scores = scores + causal_mask

    attn = F.softmax(scores, dim=-1)
    return torch.matmul(attn, v)

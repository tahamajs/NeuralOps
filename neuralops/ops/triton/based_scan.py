#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Based Attention Taylor Scan Kernel

Fused 2nd-order Taylor expansion and Horner scheme scan for Based attention
with hardware-accelerated GPU SRAM execution and PyTorch CPU fallback.
"""

from __future__ import annotations

import logging
from typing import Optional

import torch

from neuralops.ops.triton._types import HAS_TRITON, check_triton_suitability, tl, triton

logger = logging.getLogger(__name__)

__all__ = [
    "based_taylor_scan",
    "based_taylor_scan_pytorch",
]


if HAS_TRITON and tl is not None:

    @triton.jit
    def _fused_based_taylor_scan_fwd_kernel(
        Q_ptr,
        K_ptr,
        V_ptr,
        Out_ptr,
        scale,
        B,
        H,
        T,
        D,
        stride_qb,
        stride_qh,
        stride_qt,
        stride_qd,
        stride_kb,
        stride_kh,
        stride_kt,
        stride_kd,
        stride_vb,
        stride_vh,
        stride_vt,
        stride_vd,
        stride_ob,
        stride_oh,
        stride_ot,
        stride_od,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        """Fused 2nd-order Taylor polynomial scan forward kernel in SRAM."""
        pid_b = tl.program_id(0)
        pid_h = tl.program_id(1)
        pid_m = tl.program_id(2)

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_d = tl.arange(0, BLOCK_D)

        q_ptrs = (
            Q_ptr + pid_b * stride_qb + pid_h * stride_qh + offs_m[:, None] * stride_qt + offs_d[None, :] * stride_qd
        )
        q = tl.load(q_ptrs, mask=(offs_m[:, None] < T) & (offs_d[None, :] < D), other=0.0)

        acc = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)
        denom_acc = tl.zeros((BLOCK_M,), dtype=tl.float32)

        num_n_blocks = tl.cdiv(T, BLOCK_N)
        for n_blk in range(num_n_blocks):
            offs_n = n_blk * BLOCK_N + tl.arange(0, BLOCK_N)
            k_ptrs = (
                K_ptr
                + pid_b * stride_kb
                + pid_h * stride_kh
                + offs_n[:, None] * stride_kt
                + offs_d[None, :] * stride_kd
            )
            v_ptrs = (
                V_ptr
                + pid_b * stride_vb
                + pid_h * stride_vh
                + offs_n[:, None] * stride_vt
                + offs_d[None, :] * stride_vd
            )

            k = tl.load(k_ptrs, mask=(offs_n[:, None] < T) & (offs_d[None, :] < D), other=0.0)
            v = tl.load(v_ptrs, mask=(offs_n[:, None] < T) & (offs_d[None, :] < D), other=0.0)

            # Dot product (offs_m, offs_n)
            sim = tl.dot(q, tl.trans(k)) * scale
            # Taylor 2nd-order: 1 + sim + 0.5 * sim^2
            scores = 1.0 + sim + 0.5 * sim * sim

            causal_mask = offs_m[:, None] >= offs_n[None, :]
            scores = tl.where(causal_mask & (offs_m[:, None] < T) & (offs_n[None, :] < T), scores, 0.0)

            denom_acc += tl.sum(scores, axis=1)
            acc += tl.dot(scores.to(v.dtype), v)

        denom = tl.maximum(denom_acc[:, None], 1e-6)
        out = acc / denom

        o_ptrs = (
            Out_ptr + pid_b * stride_ob + pid_h * stride_oh + offs_m[:, None] * stride_ot + offs_d[None, :] * stride_od
        )
        tl.store(o_ptrs, out.to(Out_ptr.dtype.element_ty), mask=(offs_m[:, None] < T) & (offs_d[None, :] < D))

else:

    def _fused_based_taylor_scan_fwd_kernel(*args, **kwargs):
        pass


def based_taylor_scan_pytorch(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: Optional[float] = None,
) -> torch.Tensor:
    """Exact reference PyTorch implementation for Based 2nd-order Taylor scan.

    Evaluates causal polynomial expansion 1 + (q k^T) + 0.5 (q k^T)^2 on CPU/MPS.
    """
    if scale is None:
        scale = q.shape[-1] ** -0.5
    q_scaled = q * scale
    k_scaled = k
    sim = torch.matmul(q_scaled.float(), k_scaled.float().transpose(-2, -1))
    scores = 1.0 + sim + 0.5 * torch.square(sim)
    t_q, t_k = q.shape[-2], k.shape[-2]
    causal_mask = torch.tril(torch.ones(t_q, t_k, device=q.device, dtype=torch.bool))
    scores = torch.where(causal_mask, scores, torch.zeros_like(scores))
    denom = scores.sum(dim=-1, keepdim=True).clamp(min=1e-6)
    attn = scores / denom
    return torch.matmul(attn.to(v.dtype), v)


def based_taylor_scan(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: Optional[float] = None,
) -> torch.Tensor:
    """Dispatches Based 2nd-order Taylor scan to Triton JIT or PyTorch fallback."""
    suitable, reason = check_triton_suitability(q, "linear_attention")
    if not (HAS_TRITON and suitable and q.is_cuda and k.is_cuda and v.is_cuda):
        return based_taylor_scan_pytorch(q, k, v, scale=scale)

    try:
        B, H, T, D = q.shape
        sc = float(scale) if scale is not None else (1.0 / (D**0.5))
        BLOCK_M = 32
        BLOCK_N = 32
        BLOCK_D = triton.next_power_of_2(D)
        out = torch.empty_like(q)
        grid = (B, H, triton.cdiv(T, BLOCK_M))

        _fused_based_taylor_scan_fwd_kernel[grid](
            q.contiguous(),
            k.contiguous(),
            v.contiguous(),
            out,
            sc,
            B,
            H,
            T,
            D,
            q.stride(0),
            q.stride(1),
            q.stride(2),
            q.stride(3),
            k.stride(0),
            k.stride(1),
            k.stride(2),
            k.stride(3),
            v.stride(0),
            v.stride(1),
            v.stride(2),
            v.stride(3),
            out.stride(0),
            out.stride(1),
            out.stride(2),
            out.stride(3),
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            BLOCK_D=BLOCK_D,
            num_warps=4,
        )
        return out
    except Exception as e:
        logger.debug("based_taylor_scan Triton launch failed, falling back to PyTorch: %s", e)
        return based_taylor_scan_pytorch(q, k, v, scale=scale)

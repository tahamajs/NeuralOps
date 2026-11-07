#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Lightning Attention Block-Tiled Scan Kernel

Block-tiled linear attention with systolic intra-chunk and inter-chunk state propagation
with hardware-accelerated GPU SRAM execution and PyTorch CPU fallback.
"""

from __future__ import annotations

import logging
from typing import Optional, Union

import torch

from neuralops.ops.triton._types import HAS_TRITON, check_triton_suitability, tl, triton

logger = logging.getLogger(__name__)

__all__ = [
    "lightning_scan",
    "lightning_scan_pytorch",
]


if HAS_TRITON and tl is not None:

    @triton.jit
    def _fused_lightning_scan_fwd_kernel(
        Q_ptr,
        K_ptr,
        V_ptr,
        Out_ptr,
        decay_rate,
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
        """Fused block-tiled Lightning Attention forward scan in SRAM."""
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

        # Intra-chunk and inter-chunk systolic tile accumulation
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

            # Causal distance-based decay
            dist = offs_m[:, None] - offs_n[None, :]
            decay = tl.math.exp2(1.44269504 * (-decay_rate * tl.maximum(dist.to(tl.float32), 0.0)))
            causal_mask = offs_m[:, None] >= offs_n[None, :]

            # Linear kernel weights
            weights = tl.dot(q, tl.trans(k)) * decay
            weights = tl.where(causal_mask & (offs_m[:, None] < T) & (offs_n[None, :] < T), weights, 0.0)

            acc += tl.dot(weights.to(v.dtype), v)

        o_ptrs = (
            Out_ptr + pid_b * stride_ob + pid_h * stride_oh + offs_m[:, None] * stride_ot + offs_d[None, :] * stride_od
        )
        tl.store(o_ptrs, acc.to(Out_ptr.dtype.element_ty), mask=(offs_m[:, None] < T) & (offs_d[None, :] < D))

else:

    def _fused_lightning_scan_fwd_kernel(*args, **kwargs):
        pass


def lightning_scan_pytorch(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    s: Union[torch.Tensor, float] = 1.0,
    slope: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Exact reference PyTorch implementation for Lightning Attention scan.

    Evaluates block-tiled linear attention with causal decay on CPU/MPS.
    """
    B, H, T, D = q.shape
    scale = D**-0.5
    q_scaled = q.float() * scale
    k_float = k.float()
    v_float = v.float()

    if slope is None:
        decay_val = 0.1
    else:
        decay_val = slope.float()

    # Linear attention with causal exponential decay
    positions = torch.arange(T, device=q.device)
    dist = (positions[:, None] - positions[None, :]).clamp(min=0).float()
    causal_mask = positions[:, None] >= positions[None, :]

    decay = torch.exp(-decay_val * dist) * causal_mask

    sim = torch.matmul(q_scaled, k_float.transpose(-2, -1)) * decay
    out = torch.matmul(sim, v_float)
    return out.to(q.dtype)


def lightning_scan(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    s: Union[torch.Tensor, float] = 1.0,
    slope: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Dispatches Lightning block-tiled linear attention to Triton JIT or PyTorch fallback."""
    suitable, reason = check_triton_suitability(q, "linear_attention")
    if not (HAS_TRITON and suitable and q.is_cuda and k.is_cuda and v.is_cuda):
        return lightning_scan_pytorch(q, k, v, s=s, slope=slope)

    try:
        B, H, T, D = q.shape
        BLOCK_M = 32
        BLOCK_N = 32
        BLOCK_D = triton.next_power_of_2(D)
        out = torch.empty_like(q)
        grid = (B, H, triton.cdiv(T, BLOCK_M))

        decay_rate = float(s) if not isinstance(s, torch.Tensor) else float(s.mean().item())

        _fused_lightning_scan_fwd_kernel[grid](
            q.contiguous(),
            k.contiguous(),
            v.contiguous(),
            out,
            decay_rate,
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
        logger.debug("lightning_scan Triton launch failed, falling back to PyTorch: %s", e)
        return lightning_scan_pytorch(q, k, v, s=s, slope=slope)

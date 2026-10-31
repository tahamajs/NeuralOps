#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Galerkin Attention Operator Triton Kernel

Computes the non-local continuous Galerkin projection operator:
G = (K^T @ V) / L, Out = Q @ G
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
            triton.Config({"BLOCK_M": 64, "BLOCK_D": 32}, num_warps=4, num_stages=2),
            triton.Config({"BLOCK_M": 32, "BLOCK_D": 32}, num_warps=2, num_stages=2),
            triton.Config({"BLOCK_M": 128, "BLOCK_D": 32}, num_warps=8, num_stages=3),
        ],
        key=["L", "D"],
    )
    @triton.jit
    def _galerkin_contract_fwd_kernel(
        Q_ptr, G_ptr, Out_ptr,
        scale,
        stride_qb, stride_qh, stride_ql, stride_qd,
        stride_gb, stride_gh, stride_gdk, stride_gdv,
        stride_ob, stride_oh, stride_ol, stride_od,
        L, D: tl.constexpr,
        BLOCK_M: tl.constexpr, BLOCK_D: tl.constexpr,
    ):
        start_m = tl.program_id(0)
        off_h = tl.program_id(1)
        off_b = tl.program_id(2)

        offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_d = tl.arange(0, BLOCK_D)

        q_ptrs = Q_ptr + off_b * stride_qb + off_h * stride_qh + offs_m[:, None] * stride_ql + offs_d[None, :] * stride_qd
        q = tl.load(q_ptrs, mask=(offs_m[:, None] < L) & (offs_d[None, :] < D), other=0.0).to(tl.float32)

        g_ptrs = G_ptr + off_b * stride_gb + off_h * stride_gh + offs_d[:, None] * stride_gdk + offs_d[None, :] * stride_gdv
        g = tl.load(g_ptrs, mask=(offs_d[:, None] < D) & (offs_d[None, :] < D), other=0.0).to(tl.float32)

        out = tl.dot(q, g) * scale

        out_ptrs = Out_ptr + off_b * stride_ob + off_h * stride_oh + offs_m[:, None] * stride_ol + offs_d[None, :] * stride_od
        tl.store(out_ptrs, out, mask=(offs_m[:, None] < L) & (offs_d[None, :] < D))


def galerkin_triton_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: Optional[float] = None,
) -> torch.Tensor:
    """Execute Galerkin attention contraction using Triton or PyTorch fallback."""
    is_suitable, reason = check_triton_suitability(q, "galerkin")
    if not is_suitable:
        logger.warning(
            "Level 2 Triton fallback triggered for galerkin: %s. Falling back to Level 3 PyTorch reference.",
            reason,
        )
        return galerkin_pytorch_reference(q, k, v, scale=scale)

    B, H, L, D = q.shape
    if scale is None:
        scale = 1.0 / float(L)

    # Compute global operator matrix G = K^T @ V in float32
    g = torch.matmul(k.transpose(-1, -2).float(), v.float())

    out = torch.empty_like(q)
    grid = lambda meta: (triton.cdiv(L, meta["BLOCK_M"]), H, B)

    _galerkin_contract_fwd_kernel[grid](
        q, g, out,
        scale,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        g.stride(0), g.stride(1), g.stride(2), g.stride(3),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        L, D,
    )
    return out


def galerkin_pytorch_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: Optional[float] = None,
) -> torch.Tensor:
    """Exact PyTorch reference implementation for Galerkin attention."""
    L = q.shape[-2]
    if scale is None:
        scale = 1.0 / float(L)
    g = torch.matmul(k.transpose(-1, -2).float(), v.float())
    out = torch.matmul(q.float(), g) * scale
    return out.to(q.dtype)

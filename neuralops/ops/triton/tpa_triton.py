#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Tensor-Product Attention (TPA) Triton Kernel

Fuses low-rank factors Kronecker outer product expansion with output projection:
Q_{i, j*rb + k} = q1_{i, j} * q2_{i, k}
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
            triton.Config({"BLOCK_L": 64}, num_warps=4, num_stages=2),
            triton.Config({"BLOCK_L": 32}, num_warps=2, num_stages=2),
            triton.Config({"BLOCK_L": 128}, num_warps=8, num_stages=3),
        ],
        key=["L", "RA", "RB"],
    )
    @triton.jit
    def _tpa_outer_fwd_kernel(
        Q1_ptr, Q2_ptr, Out_ptr,
        stride_q1b, stride_q1h, stride_q1l, stride_q1ra,
        stride_q2b, stride_q2h, stride_q2l, stride_q2rb,
        stride_ob, stride_oh, stride_ol, stride_od,
        L, RA: tl.constexpr, RB: tl.constexpr,
        BLOCK_L: tl.constexpr,
    ):
        start_l = tl.program_id(0)
        off_h = tl.program_id(1)
        off_b = tl.program_id(2)

        offs_l = start_l * BLOCK_L + tl.arange(0, BLOCK_L)
        offs_ra = tl.arange(0, RA)
        offs_rb = tl.arange(0, RB)

        q1_ptrs = Q1_ptr + off_b * stride_q1b + off_h * stride_q1h + offs_l[:, None] * stride_q1l + offs_ra[None, :] * stride_q1ra
        q2_ptrs = Q2_ptr + off_b * stride_q2b + off_h * stride_q2h + offs_l[:, None] * stride_q2l + offs_rb[None, :] * stride_q2rb

        q1 = tl.load(q1_ptrs, mask=offs_l[:, None] < L, other=0.0).to(tl.float32)
        q2 = tl.load(q2_ptrs, mask=offs_l[:, None] < L, other=0.0).to(tl.float32)

        # Outer product per token l
        for j in range(RA):
            v1 = tl.load(Q1_ptr + off_b * stride_q1b + off_h * stride_q1h + offs_l * stride_q1l + j * stride_q1ra, mask=offs_l < L, other=0.0).to(tl.float32)
            for k in range(RB):
                v2 = tl.load(Q2_ptr + off_b * stride_q2b + off_h * stride_q2h + offs_l * stride_q2l + k * stride_q2rb, mask=offs_l < L, other=0.0).to(tl.float32)
                prod = v1 * v2
                d_idx = j * RB + k
                out_ptrs = Out_ptr + off_b * stride_ob + off_h * stride_oh + offs_l * stride_ol + d_idx * stride_od
                tl.store(out_ptrs, prod, mask=offs_l < L)


def tpa_triton_forward(
    q1: torch.Tensor,
    q2: torch.Tensor,
) -> torch.Tensor:
    """Execute Tensor-Product Attention Kronecker expansion using Triton or PyTorch fallback."""
    is_suitable, reason = check_triton_suitability(q1, "tpa")
    if not is_suitable:
        logger.warning(
            "Level 2 Triton fallback triggered for tpa: %s. Falling back to Level 3 PyTorch reference.",
            reason,
        )
        return tpa_pytorch_reference(q1, q2)

    B, H, L, RA = q1.shape
    _, _, _, RB = q2.shape
    D = RA * RB

    out = torch.empty((B, H, L, D), device=q1.device, dtype=q1.dtype)
    grid = lambda meta: (triton.cdiv(L, meta["BLOCK_L"]), H, B)

    _tpa_outer_fwd_kernel[grid](
        q1, q2, out,
        q1.stride(0), q1.stride(1), q1.stride(2), q1.stride(3),
        q2.stride(0), q2.stride(1), q2.stride(2), q2.stride(3),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        L, RA, RB,
    )
    return out


def tpa_pytorch_reference(
    q1: torch.Tensor,
    q2: torch.Tensor,
) -> torch.Tensor:
    """Exact PyTorch reference implementation for TPA Kronecker outer product expansion."""
    return (q1.unsqueeze(-1) * q2.unsqueeze(-2)).flatten(-2, -1)

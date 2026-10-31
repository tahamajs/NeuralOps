#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Non-local Associative State Compression (NASC) Triton Kernel

Fuses multi-branch gating and token contraction across local, sparse, and global streams:
O = g0 * O_local + g1 * O_sparse + g2 * O_global
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
    def _nasc_gate_fusion_fwd_kernel(
        O1_ptr, O2_ptr, O3_ptr, G_ptr, Out_ptr,
        stride_ob, stride_oh, stride_ol, stride_od,
        stride_gb, stride_gh, stride_gl, stride_gg,
        L, D: tl.constexpr,
        BLOCK_M: tl.constexpr, BLOCK_D: tl.constexpr,
    ):
        start_m = tl.program_id(0)
        off_h = tl.program_id(1)
        off_b = tl.program_id(2)

        offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_d = tl.arange(0, BLOCK_D)

        mask_m = offs_m < L
        mask_d = offs_d < D
        mask_md = (offs_m[:, None] < L) & (offs_d[None, :] < D)

        o1_ptrs = O1_ptr + off_b * stride_ob + off_h * stride_oh + offs_m[:, None] * stride_ol + offs_d[None, :] * stride_od
        o2_ptrs = O2_ptr + off_b * stride_ob + off_h * stride_oh + offs_m[:, None] * stride_ol + offs_d[None, :] * stride_od
        o3_ptrs = O3_ptr + off_b * stride_ob + off_h * stride_oh + offs_m[:, None] * stride_ol + offs_d[None, :] * stride_od

        o1 = tl.load(o1_ptrs, mask=mask_md, other=0.0).to(tl.float32)
        o2 = tl.load(o2_ptrs, mask=mask_md, other=0.0).to(tl.float32)
        o3 = tl.load(o3_ptrs, mask=mask_md, other=0.0).to(tl.float32)

        g0_ptrs = G_ptr + off_b * stride_gb + off_h * stride_gh + offs_m * stride_gl + 0 * stride_gg
        g1_ptrs = G_ptr + off_b * stride_gb + off_h * stride_gh + offs_m * stride_gl + 1 * stride_gg
        g2_ptrs = G_ptr + off_b * stride_gb + off_h * stride_gh + offs_m * stride_gl + 2 * stride_gg

        g0 = tl.load(g0_ptrs, mask=mask_m, other=0.0).to(tl.float32)
        g1 = tl.load(g1_ptrs, mask=mask_m, other=0.0).to(tl.float32)
        g2 = tl.load(g2_ptrs, mask=mask_m, other=0.0).to(tl.float32)

        out = g0[:, None] * o1 + g1[:, None] * o2 + g2[:, None] * o3

        out_ptrs = Out_ptr + off_b * stride_ob + off_h * stride_oh + offs_m[:, None] * stride_ol + offs_d[None, :] * stride_od
        tl.store(out_ptrs, out, mask=mask_md)


def nasc_triton_forward(
    o_loc: torch.Tensor,
    o_sp: torch.Tensor,
    o_glb: torch.Tensor,
    gates: torch.Tensor,
) -> torch.Tensor:
    """Execute NASC multi-branch gated stream combination using Triton or PyTorch fallback."""
    is_suitable, reason = check_triton_suitability(o_loc, "nasc")
    if not is_suitable:
        logger.warning(
            "Level 2 Triton fallback triggered for nasc: %s. Falling back to Level 3 PyTorch reference.",
            reason,
        )
        return nasc_pytorch_reference(o_loc, o_sp, o_glb, gates)

    B, H, L, D = o_loc.shape
    out = torch.empty_like(o_loc)
    grid = lambda meta: (triton.cdiv(L, meta["BLOCK_M"]), H, B)

    _nasc_gate_fusion_fwd_kernel[grid](
        o_loc, o_sp, o_glb, gates, out,
        o_loc.stride(0), o_loc.stride(1), o_loc.stride(2), o_loc.stride(3),
        gates.stride(0), gates.stride(1), gates.stride(2), gates.stride(3),
        L, D,
    )
    return out


def nasc_pytorch_reference(
    o_loc: torch.Tensor,
    o_sp: torch.Tensor,
    o_glb: torch.Tensor,
    gates: torch.Tensor,
) -> torch.Tensor:
    """Exact PyTorch reference implementation for NASC gated multi-stream fusion."""
    g0 = gates[..., 0:1]
    g1 = gates[..., 1:2]
    g2 = gates[..., 2:3]
    return g0 * o_loc + g1 * o_sp + g2 * o_glb

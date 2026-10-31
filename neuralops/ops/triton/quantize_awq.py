#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""AWQ (Activation-aware Weight Quantization) Triton kernel with PyTorch fallback."""

from __future__ import annotations

import logging
import os
from typing import Optional

import torch

try:
    import triton
    import triton.language as tl
except ImportError:
    triton = None
    tl = None

from neuralops.ops.triton._types import HAS_TRITON, check_triton_suitability

logger = logging.getLogger(__name__)

_INTERPRET_MODE = os.environ.get("TRITON_INTERPRET") == "1"

if HAS_TRITON and triton is not None and tl is not None:
    @triton.jit
    def _awq_quantize_kernel(
        W_ptr,
        Out_ptr,
        Scale_ptr,
        N: tl.constexpr,
        q_min: tl.constexpr,
        q_max: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        pid = tl.program_id(0)
        offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offs < N
        w = tl.load(W_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        s = tl.load(Scale_ptr + offs, mask=mask, other=1.0).to(tl.float32)

        # AWQ: clamp(round(w / s), q_min, q_max) * s
        q = tl.clamp(tl.math.floor(w / s + 0.5), q_min, q_max)
        deq = q * s
        tl.store(Out_ptr + offs, deq, mask=mask)

    @triton.autotune(
        configs=[
            triton.Config({"BLOCK_SIZE": 256}, num_warps=4),
            triton.Config({"BLOCK_SIZE": 512}, num_warps=4),
            triton.Config({"BLOCK_SIZE": 1024}, num_warps=8),
        ],
        key=["N"],
    )
    @triton.jit
    def _awq_quantize_kernel_autotuned(
        W_ptr,
        Out_ptr,
        Scale_ptr,
        N: tl.constexpr,
        q_min: tl.constexpr,
        q_max: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        pid = tl.program_id(0)
        offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offs < N
        w = tl.load(W_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        s = tl.load(Scale_ptr + offs, mask=mask, other=1.0).to(tl.float32)

        q = tl.clamp(tl.math.floor(w / s + 0.5), q_min, q_max)
        deq = q * s
        tl.store(Out_ptr + offs, deq, mask=mask)


def _compute_awq_scales(
    W: torch.Tensor,
    group_size: int = 128,
    bits: int = 4,
    act_scales: Optional[torch.Tensor] = None,
    alpha: float = 0.5,
) -> torch.Tensor:
    """Compute effective per-element scales matching AWQQuantizer."""
    q_max = float(2 ** (bits - 1) - 1)
    orig_w = W.to(torch.float32)

    if orig_w.dim() == 1:
        out_features, in_features = 1, orig_w.shape[0]
        orig_w = orig_w.unsqueeze(0)
    else:
        out_features, in_features = orig_w.shape

    if act_scales is None:
        act_s = torch.zeros(in_features, device=W.device, dtype=torch.float32)
    else:
        act_s = act_scales.to(device=W.device, dtype=torch.float32)

    act_s = act_s.clamp(min=1e-5).pow(alpha)
    W_scaled = orig_w * act_s.unsqueeze(0)

    if group_size > 0 and in_features % group_size == 0:
        num_groups = in_features // group_size
        W_grouped = W_scaled.reshape(out_features, num_groups, group_size)
        max_val = torch.amax(torch.abs(W_grouped), dim=-1, keepdim=True).clamp(min=1e-8)
        group_scales = max_val / q_max
        weight_scale = group_scales.expand(out_features, num_groups, group_size).reshape(
            out_features, in_features
        )
    else:
        max_val = torch.amax(torch.abs(W_scaled), dim=-1, keepdim=True).clamp(min=1e-8)
        scale = max_val / q_max
        weight_scale = scale.expand(out_features, in_features)

    effective_scale = weight_scale / act_s.unsqueeze(0)
    return effective_scale.reshape(W.shape)


def _awq_pytorch(
    W: torch.Tensor,
    group_size: int = 128,
    bits: int = 4,
    act_scales: Optional[torch.Tensor] = None,
    alpha: float = 0.5,
) -> torch.Tensor:
    """Pure PyTorch reference matching AWQQuantizer."""
    q_max = float(2 ** (bits - 1) - 1)
    q_min = float(-(2 ** (bits - 1)))
    effective_scale = _compute_awq_scales(
        W, group_size=group_size, bits=bits, act_scales=act_scales, alpha=alpha
    )
    W_fp32 = W.to(torch.float32)
    q = torch.clamp(torch.round(W_fp32 / effective_scale), q_min, q_max)
    deq = q * effective_scale
    return deq.to(dtype=W.dtype)


def awq_compress(
    W: torch.Tensor,
    group_size: int = 128,
    bits: int = 4,
    act_scales: Optional[torch.Tensor] = None,
    alpha: float = 0.5,
) -> torch.Tensor:
    """AWQ compression: per-group activation-aware scaling, integer quantization."""
    if not _INTERPRET_MODE:
        is_suitable, reason = check_triton_suitability(W, "awq", allow_fp32=True)
        if not is_suitable:
            logger.warning("awq Triton fallback: %s. Using PyTorch reference.", reason)
            return _awq_pytorch(
                W, group_size=group_size, bits=bits, act_scales=act_scales, alpha=alpha
            )

    effective_scale = _compute_awq_scales(
        W, group_size=group_size, bits=bits, act_scales=act_scales, alpha=alpha
    )

    q_max = float(2 ** (bits - 1) - 1)
    q_min = float(-(2 ** (bits - 1)))

    W_flat = W.contiguous().reshape(-1).to(torch.float32)
    scales_flat = effective_scale.contiguous().reshape(-1).to(torch.float32)
    N = W_flat.numel()

    out = torch.empty_like(W_flat)

    if _INTERPRET_MODE:
        block_size = 256
        grid = (triton.cdiv(N, block_size),)
        _awq_quantize_kernel[grid](
            W_flat,
            out,
            scales_flat,
            N,
            q_min,
            q_max,
            BLOCK_SIZE=block_size,
        )
    else:
        grid = lambda meta: (triton.cdiv(N, meta["BLOCK_SIZE"]),)
        _awq_quantize_kernel_autotuned[grid](
            W_flat,
            out,
            scales_flat,
            N,
            q_min,
            q_max,
        )

    return out.reshape(W.shape).to(dtype=W.dtype)


awq_compress_pytorch = _awq_pytorch

__all__ = ["awq_compress", "awq_compress_pytorch"]

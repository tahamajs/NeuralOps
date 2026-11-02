#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""ADMM 2:4 structured pruning Triton kernel with PyTorch fallback."""

from __future__ import annotations

import logging
import os

import torch

try:
    import triton
    import triton.language as tl
except ImportError:
    triton = None
    tl = None

from neuralops.core.pruning_utils import apply_24_structured_mask
from neuralops.ops.triton._types import HAS_TRITON, check_triton_suitability

logger = logging.getLogger(__name__)

_INTERPRET_MODE = os.environ.get("TRITON_INTERPRET") == "1"

if HAS_TRITON and triton is not None and tl is not None:
    @triton.jit
    def _admm_24_kernel(
        W_ptr,
        Out_ptr,
        total_groups: tl.constexpr,
        BLOCK_GROUPS: tl.constexpr,
    ):
        pid = tl.program_id(0)
        group_idx = pid * BLOCK_GROUPS + tl.arange(0, BLOCK_GROUPS)
        mask = group_idx < total_groups

        offs0 = group_idx * 4 + 0
        offs1 = group_idx * 4 + 1
        offs2 = group_idx * 4 + 2
        offs3 = group_idx * 4 + 3

        w0 = tl.load(W_ptr + offs0, mask=mask, other=0.0).to(tl.float32)
        w1 = tl.load(W_ptr + offs1, mask=mask, other=0.0).to(tl.float32)
        w2 = tl.load(W_ptr + offs2, mask=mask, other=0.0).to(tl.float32)
        w3 = tl.load(W_ptr + offs3, mask=mask, other=0.0).to(tl.float32)

        s0 = tl.abs(w0)
        s1 = tl.abs(w1)
        s2 = tl.abs(w2)
        s3 = tl.abs(w3)

        r0 = (s1 > s0).to(tl.int32) + (s2 > s0).to(tl.int32) + (s3 > s0).to(tl.int32)
        r1 = (s0 >= s1).to(tl.int32) + (s2 > s1).to(tl.int32) + (s3 > s1).to(tl.int32)
        r2 = (s0 >= s2).to(tl.int32) + (s1 >= s2).to(tl.int32) + (s3 > s2).to(tl.int32)
        r3 = (s0 >= s3).to(tl.int32) + (s1 >= s3).to(tl.int32) + (s2 >= s3).to(tl.int32)

        k0 = r0 < 2
        k1 = r1 < 2
        k2 = r2 < 2
        k3 = r3 < 2

        out0 = tl.where(k0, w0, 0.0)
        out1 = tl.where(k1, w1, 0.0)
        out2 = tl.where(k2, w2, 0.0)
        out3 = tl.where(k3, w3, 0.0)

        tl.store(Out_ptr + offs0, out0, mask=mask)
        tl.store(Out_ptr + offs1, out1, mask=mask)
        tl.store(Out_ptr + offs2, out2, mask=mask)
        tl.store(Out_ptr + offs3, out3, mask=mask)

    @triton.autotune(
        configs=[
            triton.Config({"BLOCK_GROUPS": 64}, num_warps=4),
            triton.Config({"BLOCK_GROUPS": 128}, num_warps=4),
            triton.Config({"BLOCK_GROUPS": 256}, num_warps=8),
        ],
        key=["total_groups"],
    )
    @triton.jit
    def _admm_24_kernel_autotuned(
        W_ptr,
        Out_ptr,
        total_groups: tl.constexpr,
        BLOCK_GROUPS: tl.constexpr,
    ):
        pid = tl.program_id(0)
        group_idx = pid * BLOCK_GROUPS + tl.arange(0, BLOCK_GROUPS)
        mask = group_idx < total_groups

        offs0 = group_idx * 4 + 0
        offs1 = group_idx * 4 + 1
        offs2 = group_idx * 4 + 2
        offs3 = group_idx * 4 + 3

        w0 = tl.load(W_ptr + offs0, mask=mask, other=0.0).to(tl.float32)
        w1 = tl.load(W_ptr + offs1, mask=mask, other=0.0).to(tl.float32)
        w2 = tl.load(W_ptr + offs2, mask=mask, other=0.0).to(tl.float32)
        w3 = tl.load(W_ptr + offs3, mask=mask, other=0.0).to(tl.float32)

        s0 = tl.abs(w0)
        s1 = tl.abs(w1)
        s2 = tl.abs(w2)
        s3 = tl.abs(w3)

        r0 = (s1 > s0).to(tl.int32) + (s2 > s0).to(tl.int32) + (s3 > s0).to(tl.int32)
        r1 = (s0 >= s1).to(tl.int32) + (s2 > s1).to(tl.int32) + (s3 > s1).to(tl.int32)
        r2 = (s0 >= s2).to(tl.int32) + (s1 >= s2).to(tl.int32) + (s3 > s2).to(tl.int32)
        r3 = (s0 >= s3).to(tl.int32) + (s1 >= s3).to(tl.int32) + (s2 >= s3).to(tl.int32)

        k0 = r0 < 2
        k1 = r1 < 2
        k2 = r2 < 2
        k3 = r3 < 2

        out0 = tl.where(k0, w0, 0.0)
        out1 = tl.where(k1, w1, 0.0)
        out2 = tl.where(k2, w2, 0.0)
        out3 = tl.where(k3, w3, 0.0)

        tl.store(Out_ptr + offs0, out0, mask=mask)
        tl.store(Out_ptr + offs1, out1, mask=mask)
        tl.store(Out_ptr + offs2, out2, mask=mask)
        tl.store(Out_ptr + offs3, out3, mask=mask)


def _admm_pytorch(
    W: torch.Tensor,
    dim: int = 1,
) -> torch.Tensor:
    """Pure PyTorch reference implementation of 2:4 structured ADMM pruning."""
    pruned, _ = apply_24_structured_mask(W, dim=dim)
    return pruned.to(dtype=W.dtype)


def admm_pruner_compress(
    W: torch.Tensor,
    dim: int = 1,
) -> torch.Tensor:
    """ADMM 2:4 structured pruning compressor using Triton kernel with PyTorch fallback.

    Args:
        W: 2D weight matrix of shape (M, N).
        dim: Dimension along which 2:4 pattern is applied (default 1 for columns).

    Returns:
        Pruned weight matrix with 2:4 pattern along dim.
    """
    if W.dim() != 2 or dim != 1 or (W.shape[1] % 4 != 0):
        return _admm_pytorch(W, dim=dim)

    if not _INTERPRET_MODE:
        is_suitable, reason = check_triton_suitability(W, "admm_pruner", allow_fp32=True)
        if not is_suitable:
            logger.warning("admm_pruner Triton fallback: %s. Using PyTorch reference.", reason)
            return _admm_pytorch(W, dim=dim)

    total_elements = W.numel()
    total_groups = total_elements // 4

    W_float = W.contiguous().to(torch.float32)
    out = torch.empty_like(W_float)

    if _INTERPRET_MODE:
        block_groups = 64
        grid = (triton.cdiv(total_groups, block_groups),)
        _admm_24_kernel[grid](
            W_float,
            out,
            total_groups=total_groups,
            BLOCK_GROUPS=block_groups,
        )
    else:
        try:
            grid = lambda meta: (triton.cdiv(total_groups, meta["BLOCK_GROUPS"]),)
            _admm_24_kernel_autotuned[grid](
                W_float,
                out,
                total_groups=total_groups,
            )
        except Exception as exc:
            logger.warning("admm_pruner Triton launch failed: %s. Using PyTorch reference.", exc)
            return _admm_pytorch(W, dim=dim)

    return out.to(dtype=W.dtype)


admm_pruner_compress_pytorch = _admm_pytorch

__all__ = ["admm_pruner_compress", "admm_pruner_compress_pytorch", "_admm_pytorch"]

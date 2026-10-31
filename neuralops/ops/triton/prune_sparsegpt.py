#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""SparseGPT second-order pruning Triton kernel with PyTorch fallback."""

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

from neuralops.ops.triton._types import HAS_TRITON, check_triton_suitability

logger = logging.getLogger(__name__)

_INTERPRET_MODE = os.environ.get("TRITON_INTERPRET") == "1"

if HAS_TRITON and triton is not None and tl is not None:
    @triton.jit
    def _sparsegpt_mask_kernel(
        W_ptr,
        Out_ptr,
        Inv_2h_ptr,
        Threshold_ptr,
        stride_wm,
        stride_wn,
        stride_om,
        stride_on,
        M: tl.constexpr,
        N: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)

        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        mask = offs_n < N

        thresh = tl.load(Threshold_ptr + pid_m).to(tl.float32)
        inv_2h = tl.load(Inv_2h_ptr + offs_n, mask=mask, other=1.0).to(tl.float32)

        w_ptrs = W_ptr + pid_m * stride_wm + offs_n * stride_wn
        w = tl.load(w_ptrs, mask=mask, other=0.0).to(tl.float32)

        saliency = (w * w) * inv_2h
        keep = saliency > thresh

        out = tl.where(keep, w, 0.0)
        out_ptrs = Out_ptr + pid_m * stride_om + offs_n * stride_on
        tl.store(out_ptrs, out, mask=mask)

    @triton.autotune(
        configs=[
            triton.Config({"BLOCK_N": 128}, num_warps=4),
            triton.Config({"BLOCK_N": 256}, num_warps=4),
            triton.Config({"BLOCK_N": 512}, num_warps=8),
        ],
        key=["N"],
    )
    @triton.jit
    def _sparsegpt_mask_kernel_autotuned(
        W_ptr,
        Out_ptr,
        Inv_2h_ptr,
        Threshold_ptr,
        stride_wm,
        stride_wn,
        stride_om,
        stride_on,
        M: tl.constexpr,
        N: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)

        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        mask = offs_n < N

        thresh = tl.load(Threshold_ptr + pid_m).to(tl.float32)
        inv_2h = tl.load(Inv_2h_ptr + offs_n, mask=mask, other=1.0).to(tl.float32)

        w_ptrs = W_ptr + pid_m * stride_wm + offs_n * stride_wn
        w = tl.load(w_ptrs, mask=mask, other=0.0).to(tl.float32)

        saliency = (w * w) * inv_2h
        keep = saliency > thresh

        out = tl.where(keep, w, 0.0)
        out_ptrs = Out_ptr + pid_m * stride_om + offs_n * stride_on
        tl.store(out_ptrs, out, mask=mask)


def _sparsegpt_pytorch(
    W: torch.Tensor,
    sparsity: float = 0.5,
    diag_h_inv: torch.Tensor | None = None,
) -> torch.Tensor:
    """Pure PyTorch reference implementation of SparseGPT threshold pruning."""
    if sparsity <= 0.0:
        return W.clone()

    if diag_h_inv is None:
        diag_h_inv = torch.full((W.shape[1],), 0.5, device=W.device, dtype=W.dtype)

    saliency = (W.float() ** 2) / (2.0 * diag_h_inv.float().clamp(min=1e-8).unsqueeze(0))
    k = max(int(saliency.shape[1] * sparsity), 1)
    threshold = torch.kthvalue(saliency, k, dim=1, keepdim=True).values
    mask = saliency > threshold
    return (W.float() * mask.float()).to(dtype=W.dtype)


def sparsegpt_compress(
    W: torch.Tensor,
    sparsity: float = 0.5,
    diag_h_inv: torch.Tensor | None = None,
) -> torch.Tensor:
    """SparseGPT second-order pruning compressor using Triton kernel with PyTorch fallback.

    Args:
        W: 2D weight matrix of shape (M, N).
        sparsity: Ratio of weights to prune [0.0, 1.0).
        diag_h_inv: Optional diagonal of inverse Hessian of shape (N,).

    Returns:
        Pruned weight matrix of same shape and dtype as W.
    """
    if sparsity <= 0.0:
        return W.clone()

    if W.dim() != 2:
        return _sparsegpt_pytorch(W, sparsity=sparsity, diag_h_inv=diag_h_inv)

    if not _INTERPRET_MODE:
        is_suitable, reason = check_triton_suitability(W, "sparsegpt", allow_fp32=True)
        if not is_suitable:
            logger.warning("sparsegpt Triton fallback: %s. Using PyTorch reference.", reason)
            return _sparsegpt_pytorch(W, sparsity=sparsity, diag_h_inv=diag_h_inv)

    M, N = W.shape
    if diag_h_inv is None:
        diag_h_inv = torch.full((N,), 0.5, device=W.device, dtype=torch.float32)
    else:
        diag_h_inv = diag_h_inv.contiguous().to(device=W.device, dtype=torch.float32)

    inv_2h = (1.0 / (2.0 * diag_h_inv.clamp(min=1e-8))).contiguous().to(torch.float32)

    # Compute row-wise kthvalue threshold
    saliency = (W.float() ** 2) * inv_2h.unsqueeze(0)
    k = max(int(N * sparsity), 1)
    threshold = torch.kthvalue(saliency, k, dim=1).values.contiguous().to(torch.float32)

    W_float = W.contiguous().to(torch.float32)
    out = torch.empty_like(W_float)

    if _INTERPRET_MODE:
        block_n = 128
        grid = (M, triton.cdiv(N, block_n))
        _sparsegpt_mask_kernel[grid](
            W_float,
            out,
            inv_2h,
            threshold,
            W_float.stride(0),
            W_float.stride(1),
            out.stride(0),
            out.stride(1),
            M=M,
            N=N,
            BLOCK_N=block_n,
        )
    else:
        grid = lambda meta: (M, triton.cdiv(N, meta["BLOCK_N"]))
        _sparsegpt_mask_kernel_autotuned[grid](
            W_float,
            out,
            inv_2h,
            threshold,
            W_float.stride(0),
            W_float.stride(1),
            out.stride(0),
            out.stride(1),
            M=M,
            N=N,
        )

    return out.to(dtype=W.dtype)


sparsegpt_compress_pytorch = _sparsegpt_pytorch

__all__ = ["sparsegpt_compress", "sparsegpt_compress_pytorch", "_sparsegpt_pytorch"]

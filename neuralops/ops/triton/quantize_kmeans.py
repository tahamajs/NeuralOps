#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""K-Means codebook quantization Triton kernel with PyTorch fallback."""

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

from neuralops.compression.quantization.kmeans_quantize import fast_kmeans_1d
from neuralops.ops.triton._types import HAS_TRITON, check_triton_suitability

logger = logging.getLogger(__name__)

_INTERPRET_MODE = os.environ.get("TRITON_INTERPRET") == "1"

if HAS_TRITON and triton is not None and tl is not None:

    @triton.jit
    def _kmeans_assign_kernel(
        W_ptr,
        Out_ptr,
        Centroid_ptr,
        N: tl.constexpr,
        K: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        pid = tl.program_id(0)
        offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offs < N
        w = tl.load(W_ptr + offs, mask=mask, other=0.0).to(tl.float32)

        best_dist = tl.full((BLOCK_SIZE,), float("inf"), dtype=tl.float32)
        best_c = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)

        for k in range(K):
            c = tl.load(Centroid_ptr + k).to(tl.float32)
            dist = tl.abs(w - c)
            better = dist < best_dist
            best_dist = tl.where(better, dist, best_dist)
            best_c = tl.where(better, c, best_c)

        tl.store(Out_ptr + offs, best_c, mask=mask)

    @triton.autotune(
        configs=[
            triton.Config({"BLOCK_SIZE": 256}, num_warps=4),
            triton.Config({"BLOCK_SIZE": 512}, num_warps=4),
            triton.Config({"BLOCK_SIZE": 1024}, num_warps=8),
        ],
        key=["N"],
    )
    @triton.jit
    def _kmeans_assign_kernel_autotuned(
        W_ptr,
        Out_ptr,
        Centroid_ptr,
        N: tl.constexpr,
        K: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        pid = tl.program_id(0)
        offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offs < N
        w = tl.load(W_ptr + offs, mask=mask, other=0.0).to(tl.float32)

        best_dist = tl.full((BLOCK_SIZE,), float("inf"), dtype=tl.float32)
        best_c = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)

        for k in range(K):
            c = tl.load(Centroid_ptr + k).to(tl.float32)
            dist = tl.abs(w - c)
            better = dist < best_dist
            best_dist = tl.where(better, dist, best_dist)
            best_c = tl.where(better, c, best_c)

        tl.store(Out_ptr + offs, best_c, mask=mask)


def _kmeans_pytorch(W: torch.Tensor, num_clusters: int = 256) -> torch.Tensor:
    """Pure PyTorch fallback matching fast_kmeans_1d."""
    centroids, _ = fast_kmeans_1d(W, n_clusters=num_clusters)
    W_flat = W.reshape(-1, 1).float()
    c_flat = centroids.reshape(1, -1).float()
    dists = torch.abs(W_flat - c_flat)
    best_idx = torch.argmin(dists, dim=1)
    deq = centroids[best_idx].reshape(W.shape)
    return deq.to(dtype=W.dtype)


def kmeans_quantize_compress(
    W: torch.Tensor,
    num_clusters: int = 256,
) -> torch.Tensor:
    """K-Means codebook quantization."""
    if not _INTERPRET_MODE:
        is_suitable, reason = check_triton_suitability(W, "kmeans_quantize", allow_fp32=True)
        if not is_suitable:
            logger.warning("kmeans Triton fallback: %s. Using PyTorch reference.", reason)
            return _kmeans_pytorch(W, num_clusters=num_clusters)

    centroids, _ = fast_kmeans_1d(W, n_clusters=num_clusters)
    centroids = centroids.contiguous().to(device=W.device, dtype=torch.float32)
    K = centroids.numel()

    W_flat = W.contiguous().reshape(-1).to(torch.float32)
    N = W_flat.numel()

    out = torch.empty_like(W_flat)

    if _INTERPRET_MODE:
        block_size = 256
        grid = (triton.cdiv(N, block_size),)
        _kmeans_assign_kernel[grid](
            W_flat,
            out,
            centroids,
            N,
            K,
            BLOCK_SIZE=block_size,
        )
    else:
        grid = lambda meta: (triton.cdiv(N, meta["BLOCK_SIZE"]),)
        _kmeans_assign_kernel_autotuned[grid](
            W_flat,
            out,
            centroids,
            N,
            K,
        )

    return out.reshape(W.shape).to(dtype=W.dtype)


kmeans_quantize_compress_pytorch = _kmeans_pytorch

__all__ = ["kmeans_quantize_compress", "kmeans_quantize_compress_pytorch"]

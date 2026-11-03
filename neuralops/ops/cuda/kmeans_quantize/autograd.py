#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Autograd Function for KMeans Quantize CUDA Kernel.
"""

from __future__ import annotations

from pathlib import Path
import torch
from torch.autograd import Function

_OPS = None


def _get_kmeans_ops():
    global _OPS
    if _OPS is not None:
        return _OPS

    try:
        import kmeans_ops

        _OPS = kmeans_ops
        return _OPS
    except ImportError:
        pass

    try:
        from torch.utils.cpp_extension import load

        pkg_dir = Path(__file__).parent
        _OPS = load(
            name="kmeans_ops",
            sources=[str(pkg_dir / "kmeans_quantize.cu"), str(pkg_dir / "binding.cpp")],
            extra_cuda_cflags=["-O3"],  # inherits torch nvcc arches including sm_80, sm_86, sm_89, sm_90
            verbose=False,
        )
        return _OPS
    except Exception:
        return None


class KMeansQuantizeFunction(Function):
    """Autograd Function wrapping CUDA kmeans_quantize_forward kernel."""

    @staticmethod
    def forward(ctx, W: torch.Tensor, centroids: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(W, centroids)

        ops = _get_kmeans_ops()
        if ops is not None and hasattr(ops, "kmeans_quantize_forward"):
            return ops.kmeans_quantize_forward(W.contiguous(), centroids.contiguous())

        # Level 3 PyTorch reference fallback
        W_flat = W.reshape(-1, 1)
        c_flat = centroids.reshape(1, -1)
        dists = torch.abs(W_flat - c_flat)
        best_idx = torch.argmin(dists, dim=1)
        return centroids[best_idx].reshape(W.shape)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        # STE (Straight-Through Estimator)
        return grad_output, None

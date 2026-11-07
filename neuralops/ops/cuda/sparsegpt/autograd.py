#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Autograd Function for SparseGPT Pruning CUDA Kernel.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
from torch.autograd import Function

_OPS = None


def _get_sparsegpt_ops():
    global _OPS
    if _OPS is not None:
        return _OPS

    try:
        import sparsegpt_ops

        _OPS = sparsegpt_ops
        return _OPS
    except ImportError:
        pass

    try:
        from torch.utils.cpp_extension import load

        pkg_dir = Path(__file__).parent
        _OPS = load(
            name="sparsegpt_ops",
            sources=[str(pkg_dir / "sparsegpt.cu"), str(pkg_dir / "binding.cpp")],
            extra_cuda_cflags=["-O3"],  # inherits torch nvcc arches including sm_80, sm_86, sm_89, sm_90
            verbose=False,
        )
        return _OPS
    except Exception:
        return None


class SparseGPTFunction(Function):
    """Autograd Function wrapping CUDA sparsegpt_prune_forward kernel."""

    @staticmethod
    def forward(
        ctx,
        W: torch.Tensor,
        sparsity: float = 0.5,
        diag_h_inv: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        ctx.save_for_backward(W)

        ops = _get_sparsegpt_ops()
        if ops is not None and hasattr(ops, "sparsegpt_prune_forward"):
            return ops.sparsegpt_prune_forward(W.contiguous(), float(sparsity), diag_h_inv)

        # Level 3 PyTorch reference fallback
        if sparsity <= 0.0:
            return W.clone()
        if diag_h_inv is None:
            diag_h_inv = torch.full((W.shape[1],), 0.5, device=W.device, dtype=W.dtype)
        saliency = (W.float() ** 2) / (2.0 * diag_h_inv.float().clamp(min=1e-8).unsqueeze(0))
        k = max(int(saliency.shape[1] * sparsity), 1)
        threshold = torch.kthvalue(saliency, k, dim=1, keepdim=True).values
        keep = saliency > threshold
        return (W * keep).to(dtype=W.dtype)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        # STE (Straight-Through Estimator)
        return grad_output, None, None

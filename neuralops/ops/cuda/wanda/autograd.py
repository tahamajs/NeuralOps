#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Autograd Function for Wanda Pruning CUDA Kernel.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
from torch.autograd import Function

_OPS = None


def _get_wanda_ops():
    global _OPS
    if _OPS is not None:
        return _OPS

    try:
        import wanda_ops

        _OPS = wanda_ops
        return _OPS
    except ImportError:
        pass

    try:
        from torch.utils.cpp_extension import load

        pkg_dir = Path(__file__).parent
        _OPS = load(
            name="wanda_ops",
            sources=[str(pkg_dir / "wanda.cu"), str(pkg_dir / "binding.cpp")],
            extra_cuda_cflags=["-O3"],  # inherits torch nvcc arches including sm_80, sm_86, sm_89, sm_90
            verbose=False,
        )
        return _OPS
    except Exception:
        return None


class WandaFunction(Function):
    """Autograd Function wrapping CUDA wanda_prune_forward kernel."""

    @staticmethod
    def forward(
        ctx,
        W: torch.Tensor,
        sparsity: float = 0.5,
        act_norm: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        ctx.save_for_backward(W)

        ops = _get_wanda_ops()
        if ops is not None and hasattr(ops, "wanda_prune_forward"):
            return ops.wanda_prune_forward(W.contiguous(), float(sparsity), act_norm)

        # Level 3 PyTorch reference fallback
        if sparsity <= 0.0:
            return W.clone()
        if act_norm is None:
            act_norm = torch.ones(W.shape[1], device=W.device, dtype=W.dtype)
        metric = torch.abs(W.float()) * act_norm.float().unsqueeze(0)
        n_keep = max(1, min(metric.shape[1], int(round(metric.shape[1] * (1.0 - sparsity)))))
        _, top_idx = torch.topk(metric, k=n_keep, dim=1, largest=True)
        mask = torch.zeros_like(W, dtype=torch.bool)
        mask.scatter_(1, top_idx, True)
        return (W * mask).to(dtype=W.dtype)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        # STE (Straight-Through Estimator)
        return grad_output, None, None

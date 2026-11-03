#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Autograd Function for ADMM 2:4 Structured Pruning CUDA Kernel.
"""

from __future__ import annotations

from pathlib import Path
import torch
from torch.autograd import Function

_OPS = None


def _get_admm_ops():
    global _OPS
    if _OPS is not None:
        return _OPS

    try:
        import admm_ops

        _OPS = admm_ops
        return _OPS
    except ImportError:
        pass

    try:
        from torch.utils.cpp_extension import load

        pkg_dir = Path(__file__).parent
        _OPS = load(
            name="admm_ops",
            sources=[str(pkg_dir / "admm.cu"), str(pkg_dir / "binding.cpp")],
            extra_cuda_cflags=["-O3"],  # inherits torch nvcc arches including sm_80, sm_86, sm_89, sm_90
            verbose=False,
        )
        return _OPS
    except Exception:
        return None


class ADMM24Function(Function):
    """Autograd Function wrapping CUDA admm_2of4_forward kernel."""

    @staticmethod
    def forward(ctx, W: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(W)

        ops = _get_admm_ops()
        if ops is not None and hasattr(ops, "admm_2of4_forward"):
            return ops.admm_2of4_forward(W.contiguous())

        # Level 3 PyTorch reference fallback
        from neuralops.core.pruning_utils import apply_24_structured_mask

        return apply_24_structured_mask(W, dim=1)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        # STE (Straight-Through Estimator)
        return grad_output

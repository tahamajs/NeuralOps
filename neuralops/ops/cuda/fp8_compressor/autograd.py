#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Autograd Function for FP8 Compressor CUDA Kernel.
"""

from __future__ import annotations

from pathlib import Path
import torch
from torch.autograd import Function

_OPS = None


def _get_fp8_ops():
    global _OPS
    if _OPS is not None:
        return _OPS

    try:
        import fp8_ops

        _OPS = fp8_ops
        return _OPS
    except ImportError:
        pass

    try:
        from torch.utils.cpp_extension import load

        pkg_dir = Path(__file__).parent
        _OPS = load(
            name="fp8_ops",
            sources=[str(pkg_dir / "fp8_compressor.cu"), str(pkg_dir / "binding.cpp")],
            extra_cuda_cflags=["-gencode=arch=compute_75,code=sm_75", "-O3"],
            verbose=False,
        )
        return _OPS
    except Exception:
        return None


class FP8Function(Function):
    """Autograd Function wrapping CUDA fp8_compress_forward kernel."""

    @staticmethod
    def forward(ctx, W: torch.Tensor, scale: float) -> torch.Tensor:
        ctx.save_for_backward(W)
        ctx.scale = scale

        ops = _get_fp8_ops()
        if ops is not None and hasattr(ops, "fp8_compress_forward"):
            return ops.fp8_compress_forward(W.contiguous(), float(scale))

        # Level 3 PyTorch reference fallback
        w_float = W.to(torch.float32)
        max_val = 448.0
        step_mult = 0.125
        w_scaled = (w_float / scale).clamp(-max_val, max_val)
        exp = torch.floor(torch.log2(w_scaled.abs().clamp(min=1e-8)))
        base = torch.pow(2.0, exp)
        step = base * step_mult
        w_fp8 = torch.round(w_scaled / step) * step
        return (w_fp8 * scale).to(dtype=W.dtype)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        # STE (Straight-Through Estimator)
        return grad_output, None

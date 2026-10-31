#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Autograd Function for LSH Attention CUDA Kernel.
"""

from __future__ import annotations

from pathlib import Path
import torch
from torch.autograd import Function

_OPS = None


def _get_lsh_ops():
    global _OPS
    if _OPS is not None:
        return _OPS

    try:
        import lsh_ops

        _OPS = lsh_ops
        return _OPS
    except ImportError:
        pass

    try:
        from torch.utils.cpp_extension import load

        pkg_dir = Path(__file__).parent
        _OPS = load(
            name="lsh_ops",
            sources=[str(pkg_dir / "lsh_attention.cu"), str(pkg_dir / "binding.cpp")],
            extra_cuda_cflags=["-gencode=arch=compute_75,code=sm_75", "-O3"],
            verbose=False,
        )
        return _OPS
    except Exception:
        return None


class LSHAttentionFunction(Function):
    """Autograd Function wrapping CUDA lsh_attention_forward_cuda kernel."""

    @staticmethod
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        q_hash: torch.Tensor,
        k_hash: torch.Tensor,
        scale: float,
    ) -> torch.Tensor:
        ctx.save_for_backward(q, k, v)

        ops = _get_lsh_ops()
        if ops is not None and hasattr(ops, "lsh_attention_forward_cuda"):
            return ops.lsh_attention_forward_cuda(
                q.contiguous().float(),
                k.contiguous().float(),
                v.contiguous().float(),
                q_hash.contiguous().to(torch.int32),
                k_hash.contiguous().to(torch.int32),
                float(scale),
            )

        # Level 3 PyTorch fallback
        from neuralops.ops.triton.lsh_attention_triton import _lsh_attention_pytorch

        return _lsh_attention_pytorch(q, k, v)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        # STE / Autograd placeholder
        return grad_output, None, None, None, None, None

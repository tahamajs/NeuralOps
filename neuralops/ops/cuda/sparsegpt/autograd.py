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
    """Autograd Function wrapping CUDA sparsegpt_prune_forward kernel with 3-level fallback."""

    @staticmethod
    def forward(
        ctx,
        W: torch.Tensor,
        sparsity: float = 0.5,
        diag_h_inv: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        W = W.contiguous()
        if diag_h_inv is not None:
            diag_h_inv = diag_h_inv.contiguous()
        ctx.save_for_backward(W)

        from neuralops.ops.cuda import _fallback

        # Level 1: Native CUDA C++ extension
        if W.is_cuda and (diag_h_inv is None or diag_h_inv.is_cuda):
            ops = _get_sparsegpt_ops()
            if ops is not None and hasattr(ops, "sparsegpt_prune_forward"):
                try:
                    out = ops.sparsegpt_prune_forward(W, float(sparsity), diag_h_inv)
                    _fallback.set_last_dispatch_backend("sparsegpt", "cuda")
                    return out
                except Exception as e:
                    _fallback.set_last_dispatch_error("sparsegpt", f"Level 1 (CUDA) failed: {e}")
                    _fallback._DISPATCH_STATS["cuda_failures"] += 1

        # Level 2: Triton JIT Kernel
        if W.is_cuda and (diag_h_inv is None or diag_h_inv.is_cuda):
            try:
                from neuralops.ops.triton.prune_sparsegpt import sparsegpt_compress as _triton_compress
                out = _triton_compress(W, sparsity=float(sparsity), diag_h_inv=diag_h_inv)
                _fallback.set_last_dispatch_backend("sparsegpt", "triton")
                return out
            except Exception as e:
                _fallback.set_last_dispatch_error("sparsegpt", f"Level 2 (Triton) failed: {e}")
                _fallback._DISPATCH_STATS["triton_failures"] += 1

        # Level 3: Vectorized PyTorch reference fallback
        _fallback.set_last_dispatch_backend("sparsegpt", "pytorch")
        if sparsity <= 0.0:
            return W.clone()
        if diag_h_inv is None:
            diag_h_inv = torch.full((W.shape[1],), 0.5, device=W.device, dtype=W.dtype)
        saliency = (W.float() ** 2) / (2.0 * diag_h_inv.float().clamp(min=1e-8).unsqueeze(0))
        n_keep = max(1, min(saliency.shape[1], int(round(saliency.shape[1] * (1.0 - sparsity)))))
        _, top_idx = torch.topk(saliency, k=n_keep, dim=1, largest=True)
        mask = torch.zeros_like(W, dtype=torch.bool)
        mask.scatter_(1, top_idx, True)
        return (W * mask).to(dtype=W.dtype)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        # STE (Straight-Through Estimator)
        grad_output = grad_output.contiguous()
        return grad_output, None, None

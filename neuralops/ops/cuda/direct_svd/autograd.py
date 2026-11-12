#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Autograd Function for Direct SVD CUDA Kernel.
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch.autograd import Function

_OPS = None


def _get_direct_svd_ops():
    global _OPS
    if _OPS is not None:
        return _OPS

    try:
        import direct_svd_ops

        _OPS = direct_svd_ops
        return _OPS
    except ImportError:
        pass

    try:
        from torch.utils.cpp_extension import load

        pkg_dir = Path(__file__).parent
        _OPS = load(
            name="direct_svd_ops",
            sources=[str(pkg_dir / "direct_svd.cu"), str(pkg_dir / "binding.cpp")],
            extra_cuda_cflags=["-O3"],  # inherits torch nvcc arches including sm_80, sm_86, sm_89, sm_90
            verbose=False,
        )
        return _OPS
    except Exception:
        return None


class DirectSVDFunction(Function):
    """Autograd Function wrapping CUDA direct_svd_fwd kernel with 3-level fallback."""

    @staticmethod
    def forward(ctx, U: torch.Tensor, V: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
        U = U.contiguous()
        V = V.contiguous()
        ctx.save_for_backward(U, V)
        ctx.scale = float(scale)

        from neuralops.ops.cuda import _fallback

        # Level 1: Native CUDA C++ extension
        if U.is_cuda and V.is_cuda:
            ops = _get_direct_svd_ops()
            if ops is not None and hasattr(ops, "direct_svd_fwd"):
                try:
                    out = ops.direct_svd_fwd(U, V, float(scale))
                    _fallback.set_last_dispatch_backend("direct_svd", "cuda")
                    return out
                except Exception as e:
                    _fallback.set_last_dispatch_error("direct_svd", f"Level 1 (CUDA) failed: {e}")
                    _fallback._DISPATCH_STATS["cuda_failures"] += 1

        # Level 2: Triton JIT Kernel
        if U.is_cuda and V.is_cuda:
            try:
                from neuralops.ops.triton.low_rank import _direct_svd_reconstruct_kernel, HAS_TRITON
                import triton

                if HAS_TRITON and _direct_svd_reconstruct_kernel is not None:
                    M, r = U.shape
                    N = V.shape[1]
                    out = torch.empty((M, N), device=U.device, dtype=torch.float32)
                    grid = lambda META: (
                        triton.cdiv(M, META["BLOCK_M"]),
                        triton.cdiv(N, META["BLOCK_N"]),
                    )
                    _direct_svd_reconstruct_kernel[grid](
                        U,
                        V,
                        out,
                        M,
                        N,
                        r,
                        U.stride(0),
                        U.stride(1),
                        V.stride(0),
                        V.stride(1),
                        out.stride(0),
                        out.stride(1),
                        float(scale),
                    )
                    _fallback.set_last_dispatch_backend("direct_svd", "triton")
                    return out
            except Exception as e:
                _fallback.set_last_dispatch_error("direct_svd", f"Level 2 (Triton) failed: {e}")
                _fallback._DISPATCH_STATS["triton_failures"] += 1

        # Level 3: Vectorized PyTorch reference fallback
        _fallback.set_last_dispatch_backend("direct_svd", "pytorch")
        return torch.matmul(U, V) * float(scale)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        U, V = ctx.saved_tensors
        scale = ctx.scale

        grad_output = grad_output.contiguous()
        grad_scaled = grad_output * scale
        grad_U = torch.matmul(grad_scaled, V.t())
        grad_V = torch.matmul(U.t(), grad_scaled)
        return grad_U, grad_V, None

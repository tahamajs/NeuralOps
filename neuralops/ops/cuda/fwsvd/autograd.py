#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Autograd Function for FWSVD CUDA Kernel with Three-Level Fallback.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Tuple

import torch
from torch.autograd import Function

_OPS = None


def _get_fwsvd_ops() -> Any:
    global _OPS
    if _OPS is not None:
        return _OPS

    try:
        import fwsvd_ops

        _OPS = fwsvd_ops
        return _OPS
    except ImportError:
        pass

    try:
        from torch.utils.cpp_extension import load

        pkg_dir = Path(__file__).parent
        _OPS = load(
            name="fwsvd_ops",
            sources=[str(pkg_dir / "fwsvd.cu"), str(pkg_dir / "binding.cpp")],
            extra_cuda_cflags=["-O3"],  # inherits torch nvcc arches including sm_80, sm_86, sm_89, sm_90
            verbose=False,
        )
        return _OPS
    except Exception:
        return None


class FWSVDFunction(Function):
    """Autograd Function wrapping CUDA fwsvd_fwd kernel with 3-level fallback."""

    @staticmethod
    def forward(ctx: Any, U: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
        U_c = U.contiguous()
        V_c = V.contiguous()
        ctx.save_for_backward(U_c, V_c)

        from neuralops.ops.cuda import _fallback

        # Level 1: Native CUDA C++ extension
        if U.is_cuda and V.is_cuda:
            ops = _get_fwsvd_ops()
            if ops is not None and hasattr(ops, "fwsvd_fwd"):
                try:
                    out = ops.fwsvd_fwd(U_c, V_c)
                    _fallback.set_last_dispatch_backend("fwsvd", "cuda")
                    return out
                except Exception as e:
                    _fallback.set_last_dispatch_error("fwsvd", f"Level 1 (CUDA) failed: {e}")
                    _fallback._DISPATCH_STATS["cuda_failures"] += 1

        # Level 2: Triton JIT Kernel
        if U.is_cuda and V.is_cuda:
            try:
                from neuralops.ops.triton.low_rank import _fwsvd_reconstruct_kernel, HAS_TRITON
                import triton

                if HAS_TRITON and _fwsvd_reconstruct_kernel is not None:
                    M, r = U.shape
                    N = V.shape[1]
                    out = torch.empty((M, N), device=U.device, dtype=torch.float32)
                    grid = lambda META: (
                        triton.cdiv(M, META["BLOCK_M"]),
                        triton.cdiv(N, META["BLOCK_N"]),
                    )
                    _fwsvd_reconstruct_kernel[grid](
                        U_c,
                        V_c,
                        out,
                        M,
                        N,
                        r,
                        U_c.stride(0),
                        U_c.stride(1),
                        V_c.stride(0),
                        V_c.stride(1),
                        out.stride(0),
                        out.stride(1),
                    )
                    _fallback.set_last_dispatch_backend("fwsvd", "triton")
                    return out
            except Exception as e:
                _fallback.set_last_dispatch_error("fwsvd", f"Level 2 (Triton) failed: {e}")
                _fallback._DISPATCH_STATS["triton_failures"] += 1

        # Level 3: Vectorized PyTorch reference fallback
        _fallback.set_last_dispatch_backend("fwsvd", "pytorch")
        return torch.matmul(U_c, V_c)

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        U, V = ctx.saved_tensors
        grad_out_c = grad_output.contiguous().to(dtype=U.dtype)
        grad_U = torch.matmul(grad_out_c, V.t())
        grad_V = torch.matmul(U.t(), grad_out_c)
        return grad_U, grad_V

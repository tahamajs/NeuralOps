#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Fused Kronecker GEMM Autograd Interface

Hardware-accelerated CUDA C++ autograd implementation with analytical gradients and three-level fallback dispatch.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from neuralops.core.cuda_kernels._bindings import load_cuda_module
from neuralops.core.cuda_kernels._fallback import dispatch

_MODULE = None


def _get_module():
    global _MODULE
    if _MODULE is None:
        curr_dir = Path(__file__).parent
        with open(curr_dir / "kronecker_binding.cpp", "r", encoding="utf-8") as f:
            cpp_src = f.read()
        with open(curr_dir / "kronecker_gemm.cu", "r", encoding="utf-8") as f:
            cuda_src = f.read()

        _MODULE = load_cuda_module(
            name="cuda_kronecker_gemm",
            cpp_sources=cpp_src,
            cuda_sources=cuda_src,
            functions=["kronecker_gemm_fwd"],
        )
    return _MODULE


def kronecker_matmul_pytorch_ref(
    x: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """Pure PyTorch reference implementation."""
    from neuralops.core.triton_kernels.low_rank import triton_fused_kronecker_matmul_pytorch_ref
    return triton_fused_kronecker_matmul_pytorch_ref(x, A, B, bias=bias)


def _get_triton_impl():
    try:
        from neuralops.core.triton_kernels.low_rank import triton_fused_kronecker_matmul
        return triton_fused_kronecker_matmul
    except ImportError:
        return None


def kronecker_matmul(
    x: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """3-Level Dispatched Fused Kronecker Product Matmul."""
    def cuda_fn(x, A, B, bias=None):
        mod = _get_module()
        dummy_bias = torch.empty(0, device=x.device, dtype=x.dtype)
        b_arg = bias.contiguous() if bias is not None else dummy_bias
        return mod.kronecker_gemm_fwd(x.contiguous(), A.contiguous(), B.contiguous(), b_arg)

    triton_fn = _get_triton_impl()

    return dispatch(
        cuda_fn,
        triton_fn,
        kronecker_matmul_pytorch_ref,
        x,
        A,
        B,
        bias=bias,
        kernel_name="kronecker_matmul",
    )

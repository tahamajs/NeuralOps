#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Block-Diagonal GEMM Autograd Interface

Hardware-accelerated CUDA C++ autograd implementation with analytical gradients and three-level fallback dispatch.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch

from neuralops.core.cuda_kernels._bindings import load_cuda_module
from neuralops.core.cuda_kernels._fallback import dispatch

_MODULE = None


def _get_module():
    global _MODULE
    if _MODULE is None:
        curr_dir = Path(__file__).parent
        with open(curr_dir / "block_diag_binding.cpp", "r", encoding="utf-8") as f:
            cpp_src = f.read()
        with open(curr_dir / "block_diag_gemm.cu", "r", encoding="utf-8") as f:
            cuda_src = f.read()

        _MODULE = load_cuda_module(
            name="cuda_block_diag_gemm",
            cpp_sources=cpp_src,
            cuda_sources=cuda_src,
            functions=["block_diag_gemm_fwd"],
        )
    return _MODULE


def block_diag_matmul_pytorch_ref(
    x: torch.Tensor,
    weights: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Pure PyTorch reference implementation."""
    orig_shape = x.shape
    G, Ng, Kg = weights.shape
    x_3d = x.reshape(-1, G, Kg)
    y = torch.einsum("mgk,gnk->mgn", x_3d, weights)
    if bias is not None:
        y = y + bias.unsqueeze(0)
    return y.reshape(*orig_shape[:-1], G * Ng)


def _get_triton_impl():
    try:
        from neuralops.core.triton_kernels.low_rank import block_diag_matmul_triton
        return block_diag_matmul_triton
    except (ImportError, AttributeError):
        return None


def block_diag_matmul(
    x: torch.Tensor,
    weights: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """3-Level Dispatched Block-Diagonal GEMM."""
    def cuda_fn(x, weights, bias=None, **kwargs):
        mod = _get_module()
        return mod.block_diag_gemm_fwd(
            x.contiguous(),
            weights.contiguous(),
            bias.contiguous() if bias is not None else None,
        )

    triton_fn = _get_triton_impl()

    return dispatch(
        cuda_fn,
        triton_fn,
        block_diag_matmul_pytorch_ref,
        x,
        weights,
        bias=bias,
        kernel_name="block_diag_matmul",
    )

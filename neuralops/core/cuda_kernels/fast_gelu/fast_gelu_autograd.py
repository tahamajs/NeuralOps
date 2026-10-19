#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Fast GELU Activation Autograd Interface

Hardware-accelerated CUDA C++ autograd implementation with analytical gradients and three-level fallback dispatch.
"""

from __future__ import annotations

import math
from pathlib import Path

import torch

from neuralops.core.cuda_kernels._bindings import load_cuda_module
from neuralops.core.cuda_kernels._fallback import dispatch

_MODULE = None


def _get_module():
    global _MODULE
    if _MODULE is None:
        curr_dir = Path(__file__).parent
        with open(curr_dir / "fast_gelu_binding.cpp", "r", encoding="utf-8") as f:
            cpp_src = f.read()
        with open(curr_dir / "fast_gelu.cu", "r", encoding="utf-8") as f:
            cuda_src = f.read()

        _MODULE = load_cuda_module(
            name="cuda_fast_gelu",
            cpp_sources=cpp_src,
            cuda_sources=cuda_src,
            functions=["fast_gelu_fwd"],
        )
    return _MODULE


def fast_gelu_pytorch_ref(x: torch.Tensor) -> torch.Tensor:
    """Pure PyTorch reference implementation."""
    return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))


def _get_triton_impl():
    try:
        from neuralops.core.triton_kernels.elementwise import fast_gelu_triton
        return fast_gelu_triton
    except (ImportError, AttributeError):
        return None


def fast_gelu(x: torch.Tensor) -> torch.Tensor:
    """3-Level Dispatched Fast-GELU."""
    def cuda_fn(x, **kwargs):
        mod = _get_module()
        return mod.fast_gelu_fwd(x.contiguous())

    triton_fn = _get_triton_impl()

    return dispatch(
        cuda_fn,
        triton_fn,
        fast_gelu_pytorch_ref,
        x,
        kernel_name="fast_gelu",
    )

#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""Bitonic Top-K Selection Autograd Interface.

Hardware-accelerated CUDA C++ autograd implementation with analytical gradients
and three-level fallback dispatch (CUDA -> Triton -> PyTorch).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from neuralops.ops.cuda._bindings import load_cuda_module
from neuralops.ops.cuda._fallback import dispatch

_MODULE: Any = None


def _get_module() -> Any:
    """Lazy loader for CUDA C++ bitonic top-k compiled module.

    Returns:
        Loaded CUDA C++ module containing bitonic_topk_indices_fwd.
    """
    global _MODULE
    if _MODULE is None:
        curr_dir = Path(__file__).parent
        with open(curr_dir / "bitonic_topk_binding.cpp", "r", encoding="utf-8") as f:
            cpp_src = f.read()
        with open(curr_dir / "bitonic_topk.cu", "r", encoding="utf-8") as f:
            cuda_src = f.read()
        _MODULE = load_cuda_module(
            name="cuda_bitonic_topk",
            cpp_sources=cpp_src,
            cuda_sources=cuda_src,
            functions=["bitonic_topk_indices_fwd"],
        )
    return _MODULE


def bitonic_topk_pytorch_ref(x: torch.Tensor, K: int) -> torch.Tensor:
    """Pure PyTorch reference implementation for top-k index selection.

    Args:
        x: Input tensor of arbitrary floating point type.
        K: Number of top elements to select along last dimension.

    Returns:
        Tensor of int64 indices corresponding to top-k values.

    Raises:
        ValueError: If K is negative or exceeds the last dimension of x.
    """
    if K < 0:
        raise ValueError(f"K must be non-negative, got {K}")
    if K > x.size(-1):
        raise ValueError(f"K ({K}) cannot exceed last dimension size ({x.size(-1)})")
    if K == 0:
        out_shape = list(x.shape)
        out_shape[-1] = 0
        return torch.empty(out_shape, dtype=torch.int64, device=x.device)
    return torch.topk(x, k=K, dim=-1).indices


def bitonic_topk_triton_ref(x: torch.Tensor, K: int) -> torch.Tensor:
    """Triton fallback reference for bitonic top-k (delegates to PyTorch reference).

    Args:
        x: Input tensor.
        K: Number of top elements to select.

    Returns:
        Tensor of int64 indices.
    """
    return bitonic_topk_pytorch_ref(x, K)


def _fwd_dispatch(x: torch.Tensor, K: int) -> torch.Tensor:
    """Dispatches top-k forward computation across CUDA, Triton, and PyTorch.

    Args:
        x: Input tensor.
        K: Number of top elements to select.

    Returns:
        Tensor of int64 indices.
    """
    if K < 0:
        raise ValueError(f"K must be non-negative, got {K}")
    if K > x.size(-1):
        raise ValueError(f"K ({K}) cannot exceed last dimension size ({x.size(-1)})")
    if K == 0:
        out_shape = list(x.shape)
        out_shape[-1] = 0
        return torch.empty(out_shape, dtype=torch.int64, device=x.device)

    def cuda_fn(*args: Any, **kwargs: Any) -> torch.Tensor:
        mod = _get_module()
        fn = mod.bitonic_topk_indices_fwd
        return fn(*args, **kwargs)

    return dispatch(
        cuda_fn,
        bitonic_topk_triton_ref,
        bitonic_topk_pytorch_ref,
        x,
        K,
        kernel_name="bitonic_topk",
    )


class _BitonicTopKFunction(torch.autograd.Function):
    """Autograd function wrapper for Bitonic Top-K selection."""

    @staticmethod
    def forward(ctx: Any, x: torch.Tensor, K: int) -> torch.Tensor:
        """Forward pass for bitonic top-k.

        Args:
            ctx: Autograd context.
            x: Input tensor.
            K: Top elements count.

        Returns:
            Indices tensor.
        """
        return _fwd_dispatch(x, K)

    @staticmethod
    def backward(ctx: Any, *grad_outputs: Any) -> tuple[torch.Tensor | None, None]:
        """Backward pass (discrete indices have non-differentiable zero gradients).

        Args:
            ctx: Autograd context.
            grad_outputs: Gradient outputs.

        Returns:
            Gradients with respect to (x, K).
        """
        return None, None


def bitonic_topk(x: torch.Tensor, K: int) -> torch.Tensor:
    """3-Level Dispatched Bitonic Top-K with autograd and torch.amp support.

    Computes the indices of the largest K elements along the last dimension.
    Fully compliant with PyTorch AMP autocast (`torch.amp.autocast`), CPU/MPS fallback,
    and autograd graph integration.

    Args:
        x: Input tensor of shape (..., N) and any floating point dtype (float32, float16, bfloat16).
        K: Number of top elements to select (0 <= K <= N).

    Returns:
        Int64 tensor of shape (..., K) containing the indices of the top-k elements.

    Raises:
        ValueError: If K < 0 or K > x.size(-1).

    Examples:
        >>> x = torch.tensor([[1.0, 5.0, 3.0, 2.0]])
        >>> bitonic_topk(x, 2)
        tensor([[1, 2]])
    """
    if x.requires_grad:
        return _BitonicTopKFunction.apply(x, K)
    return _fwd_dispatch(x, K)

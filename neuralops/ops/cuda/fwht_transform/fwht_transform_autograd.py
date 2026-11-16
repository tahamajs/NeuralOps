#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Fast Walsh-Hadamard Transform Autograd Interface

Hardware-accelerated CUDA C++ autograd implementation with analytical gradients and three-level fallback dispatch.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from neuralops.ops.cuda._bindings import load_cuda_module
from neuralops.ops.cuda._fallback import dispatch
from neuralops.ops.triton.elementwise import fast_hadamard_transform

_MODULE: Any = None


def _get_module() -> Any:
    """Load or retrieve the cached CUDA C++ extension module for FWHT.

    Returns:
        Compiled CUDA module or None if CUDA compilation is unavailable.
    """
    global _MODULE
    if _MODULE is None:
        curr_dir = Path(__file__).parent
        with open(curr_dir / "fwht_transform_binding.cpp", "r", encoding="utf-8") as f:
            cpp_src = f.read()
        with open(curr_dir / "fwht_transform.cu", "r", encoding="utf-8") as f:
            cuda_src = f.read()
        _MODULE = load_cuda_module(
            name="cuda_fwht_transform",
            cpp_sources=cpp_src,
            cuda_sources=cuda_src,
            functions=["fwht_fwd"],
        )
    return _MODULE


def fwht_transform_pytorch_ref(x: torch.Tensor, B_dim: int = 128) -> torch.Tensor:
    """Pure PyTorch reference implementation of Fast Walsh-Hadamard Transform.

    Args:
        x: Input tensor of shape (..., D).
        B_dim: Block dimension for the butterfly transform. Must be a positive power of 2.

    Returns:
        Transformed tensor with the same shape, dtype, and device as x.

    Raises:
        ValueError: If B_dim is not a positive power of 2.
    """
    if B_dim <= 0 or (B_dim & (B_dim - 1)) != 0:
        raise ValueError(f"fwht_transform: B_dim={B_dim} must be a positive power of 2")
    return fast_hadamard_transform(x.contiguous(), block_size=B_dim)


def fwht_transform(x: torch.Tensor, B_dim: int = 128) -> torch.Tensor:
    """Execute normalized Fast Walsh-Hadamard Transform with 3-level fallback.

    Applies the normalized block-diagonal Fast Walsh-Hadamard Transform:
    Y = X @ H where H^T H = I and H = H^T (involutory and orthogonal).

    Args:
        x: Input tensor of shape (..., D).
        B_dim: Block dimension for the butterfly transform. Must be a positive power of 2.

    Returns:
        Transformed tensor with the same shape, dtype, and device as x.

    Raises:
        ValueError: If B_dim is not a positive power of 2.
    """
    if B_dim <= 0 or (B_dim & (B_dim - 1)) != 0:
        raise ValueError(f"fwht_transform: B_dim={B_dim} must be a positive power of 2")

    def cuda_fn(x: torch.Tensor, B_dim: int = 128, **kwargs: Any) -> torch.Tensor:
        mod = _get_module()
        if mod is None:
            raise RuntimeError("cuda_fwht_transform module compilation or loading failed")
        return mod.fwht_fwd(x.contiguous(), B_dim)

    return dispatch(
        cuda_fn,
        fwht_transform_pytorch_ref,
        fwht_transform_pytorch_ref,
        x,
        B_dim=B_dim,
        kernel_name="fwht_transform",
    )


__all__ = ["fwht_transform", "fwht_transform_pytorch_ref", "fast_hadamard_transform"]

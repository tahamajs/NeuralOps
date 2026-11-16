#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""Galerkin Attention Autograd Interface.

Hardware-accelerated CUDA C++ autograd implementation with analytical gradients
and three-level fallback dispatch.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from neuralops.ops.cuda._bindings import load_cuda_module
from neuralops.ops.cuda._fallback import dispatch

_MODULE = None


def _get_module() -> Any:  # reason: dynamically compiled PyTorch C++/CUDA extension module
    global _MODULE
    if _MODULE is None:
        curr_dir = Path(__file__).parent
        with open(curr_dir / "galerkin_attention_binding.cpp", "r", encoding="utf-8") as f:
            cpp_src = f.read()
        with open(curr_dir / "galerkin_attention.cu", "r", encoding="utf-8") as f:
            cuda_src = f.read()
        _MODULE = load_cuda_module("cuda_galerkin", cpp_src, cuda_src, ["galerkin_fwd"])
    return _MODULE


def galerkin_attention_ref(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, h_step: float = 0.01) -> torch.Tensor:
    """Exact reference PyTorch implementation for Galerkin Attention.

    Computes:
        g = sum_{t}(k_t * v_t) * h_step
        out = q * g

    Args:
        q: Query tensor of shape [B, H, N, D].
        k: Key tensor of shape [B, H, N, D].
        v: Value tensor of shape [B, H, N, D].
        h_step: Step size parameter (default: 0.01).

    Returns:
        Tensor of shape [B, H, N, D].
    """
    g = (k * v).sum(dim=-2, keepdim=True) * h_step
    return q * g


galerkin_attention_pytorch_ref = galerkin_attention_ref


def galerkin_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, h_step: float = 0.01) -> torch.Tensor:
    """Galerkin Attention with three-level fallback dispatch (CUDA -> Reference).

    Args:
        q: Query tensor of shape [B, H, N, D].
        k: Key tensor of shape [B, H, N, D].
        v: Value tensor of shape [B, H, N, D].
        h_step: Step size parameter (default: 0.01).

    Returns:
        Output tensor of shape [B, H, N, D].

    Raises:
        RuntimeError: If forced backend fails or shapes are invalid.
    """

    def cuda_fn(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        h_step: float = 0.01,
        **kwargs: Any,  # reason: accommodate dispatch kwargs
    ) -> torch.Tensor:
        mod = _get_module()
        if mod is None:
            raise RuntimeError("CUDA module cuda_galerkin failed to compile or load.")
        return mod.galerkin_fwd(q.contiguous(), k.contiguous(), v.contiguous(), float(h_step))

    return dispatch(
        cuda_fn,
        galerkin_attention_ref,
        galerkin_attention_ref,
        q,
        k,
        v,
        h_step=h_step,
        kernel_name="galerkin_attention",
    )


galerkin_attention_forward = galerkin_attention

__all__ = [
    "galerkin_attention",
    "galerkin_attention_forward",
    "galerkin_attention_ref",
    "galerkin_attention_pytorch_ref",
]

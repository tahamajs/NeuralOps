#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Goldfinch AHN Recurrent Attention Autograd Interface

Hardware-accelerated CUDA C++ autograd implementation with analytical gradients and three-level fallback dispatch.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from neuralops.ops.cuda._bindings import load_cuda_module
from neuralops.ops.cuda._fallback import dispatch

__all__ = ["goldfinch_ahn", "goldfinch_ahn_ref"]

_MODULE: Any = None


def _get_module() -> Any:
    global _MODULE
    if _MODULE is None:
        curr_dir = Path(__file__).parent
        with open(curr_dir / "goldfinch_ahn_binding.cpp", "r", encoding="utf-8") as f:
            cpp_src = f.read()
        with open(curr_dir / "goldfinch_ahn.cu", "r", encoding="utf-8") as f:
            cuda_src = f.read()
        _MODULE = load_cuda_module("cuda_goldfinch_ahn", cpp_src, cuda_src, ["goldfinch_ahn_fwd"])
    return _MODULE


def goldfinch_ahn_ref(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
    """Pure PyTorch reference implementation of Goldfinch AHN recurrent attention.

    Args:
        q: Query tensor of shape (B, H, N, D).
        k: Key tensor of shape (B, H, N, D).
        v: Value tensor of shape (B, H, N, D).
        beta: Step size tensor of shape (B, H, N, D).

    Returns:
        Output tensor of shape (B, H, N, D).

    Raises:
        ValueError: If input tensor dimensions or shapes do not match.
    """
    if q.dim() != 4:
        raise ValueError(f"q must be 4D (B, H, N, D), got shape {q.shape}")
    if k.shape != q.shape:
        raise ValueError(f"k shape {k.shape} does not match q shape {q.shape}")
    if v.shape != q.shape:
        raise ValueError(f"v shape {v.shape} does not match q shape {q.shape}")
    if beta.shape != q.shape:
        raise ValueError(f"beta shape {beta.shape} does not match q shape {q.shape}")

    B, H, N, D = q.shape
    if B == 0 or H == 0 or N == 0 or D == 0:
        return torch.empty_like(q)

    out = torch.empty_like(q)
    s = torch.zeros(B, H, D, device=q.device, dtype=q.dtype)
    for t in range(N):
        pred = k[:, :, t, :] * s
        err = v[:, :, t, :] - pred
        s = s + beta[:, :, t, :] * k[:, :, t, :] * err
        out[:, :, t, :] = q[:, :, t, :] * s
    return out


def goldfinch_ahn(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
    """Hardware-accelerated Goldfinch AHN recurrent attention with fallback.

    Args:
        q: Query tensor of shape (B, H, N, D).
        k: Key tensor of shape (B, H, N, D).
        v: Value tensor of shape (B, H, N, D).
        beta: Step size tensor of shape (B, H, N, D).

    Returns:
        Output tensor of shape (B, H, N, D).
    """

    def cuda_fn(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
        mod = _get_module()
        if mod is None:
            raise RuntimeError("Failed to load CUDA module cuda_goldfinch_ahn")
        return mod.goldfinch_ahn_fwd(q.contiguous(), k.contiguous(), v.contiguous(), beta.contiguous())

    return dispatch(cuda_fn, goldfinch_ahn_ref, goldfinch_ahn_ref, q, k=k, v=v, beta=beta, kernel_name="goldfinch_ahn")

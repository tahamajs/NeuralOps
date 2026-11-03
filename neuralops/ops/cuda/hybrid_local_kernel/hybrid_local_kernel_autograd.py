#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Hybrid Local Kernel Attention Autograd Interface

Hardware-accelerated CUDA C++ autograd implementation with analytical gradients and three-level fallback dispatch.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn.functional as F

from neuralops.ops.cuda._bindings import load_cuda_module
from neuralops.ops.cuda._fallback import dispatch

logger = logging.getLogger(__name__)

_MODULE = None


def _get_module():
    global _MODULE
    if _MODULE is None:
        curr_dir = Path(__file__).parent
        with open(curr_dir / "hybrid_local_kernel_binding.cpp", "r", encoding="utf-8") as f:
            cpp_src = f.read()
        with open(curr_dir / "hybrid_local_kernel.cu", "r", encoding="utf-8") as f:
            cuda_src = f.read()
        _MODULE = load_cuda_module(
            name="cuda_hybrid_local_kernel",
            cpp_sources=cpp_src,
            cuda_sources=cuda_src,
            functions=[
                "hybrid_local_attn_fwd",
                "hybrid_two_branch_blend_fwd",
                "hybrid_three_branch_blend_fwd",
                "hybrid_heterogeneous_branch_fwd",
            ],
        )
    return _MODULE


def hybrid_local_attn_ref(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    radius: int = 32,
    is_causal: bool = False,
) -> torch.Tensor:
    """Pure PyTorch reference implementation."""
    B, H, N, D = q.shape
    scale = 1.0 / math.sqrt(D)
    scores = torch.matmul(q.float(), k.float().transpose(-2, -1)) * scale

    q_idx = torch.arange(N, device=q.device).unsqueeze(1)
    k_idx = torch.arange(N, device=q.device).unsqueeze(0)
    dist = (q_idx - k_idx).abs()
    band = dist <= radius
    if is_causal:
        causal = k_idx <= q_idx
        band = band & causal

    scores = scores.masked_fill(~band.unsqueeze(0).unsqueeze(0), -1e4)
    probs = F.softmax(scores, dim=-1).to(q.dtype)
    return torch.matmul(probs, v)


def hybrid_two_branch_blend_ref(
    b0: torch.Tensor,
    b1: torch.Tensor,
    alpha: torch.Tensor,
) -> torch.Tensor:
    """Pure PyTorch reference blend."""
    return alpha * b0 + (1.0 - alpha) * b1


def hybrid_local_attn(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    radius: int = 32,
    is_causal: bool = False,
) -> torch.Tensor:
    """3-Tier Dispatched Hybrid Local Attention."""
    def cuda_fn(q, k, v, radius=32, is_causal=False):
        mod = _get_module()
        if mod is None:
            return None
        return mod.hybrid_local_attn_fwd(
            q.contiguous(), k.contiguous(), v.contiguous(), int(radius), bool(is_causal)
        )

    def triton_fn(q, k, v, radius=32, is_causal=False):
        try:
            from neuralops.ops.triton._types import check_triton_suitability
            suitable, _ = check_triton_suitability(q, "hybrid_local_attn")
            if not suitable:
                return None
            from neuralops.kernels.triton.hybrid_local_kernel_triton import hybrid_local_attn_triton
            return hybrid_local_attn_triton(q, k, v, radius=radius, is_causal=is_causal)
        except (RuntimeError, MemoryError, AssertionError, ValueError, ImportError, AttributeError) as exc:
            logger.warning("Failed in Triton hybrid_local_attn: %s", exc)
            return None

    if q.shape[-1] > 256:
        cuda_fn_use = None
    else:
        cuda_fn_use = cuda_fn

    return dispatch(
        cuda_fn_use,
        triton_fn,
        hybrid_local_attn_ref,
        q,
        k=k,
        v=v,
        radius=radius,
        is_causal=is_causal,
        kernel_name="hybrid_local_attn",
    )


def hybrid_two_branch_blend(
    b0: torch.Tensor,
    b1: torch.Tensor,
    alpha: torch.Tensor,
) -> torch.Tensor:
    """3-Tier Dispatched Two Branch Gated Blend."""
    def cuda_fn(b0, b1, alpha):
        mod = _get_module()
        if mod is None:
            return None
        return mod.hybrid_two_branch_blend_fwd(
            b0.contiguous(), b1.contiguous(), alpha.contiguous()
        )

    def triton_fn(b0, b1, alpha):
        try:
            from neuralops.ops.triton._types import check_triton_suitability
            suitable, _ = check_triton_suitability(b0, "hybrid_two_branch_blend")
            if not suitable:
                return None
            from neuralops.kernels.triton.hybrid_local_kernel_triton import hybrid_two_branch_blend_triton
            return hybrid_two_branch_blend_triton(b0, b1, alpha)
        except (RuntimeError, MemoryError, AssertionError, ValueError, ImportError, AttributeError) as exc:
            logger.warning("Failed in Triton hybrid_two_branch_blend: %s", exc)
            return None

    return dispatch(
        cuda_fn,
        triton_fn,
        hybrid_two_branch_blend_ref,
        b0,
        b1=b1,
        alpha=alpha,
        kernel_name="hybrid_two_branch_blend",
    )


def hybrid_three_branch_blend_ref(
    b0: torch.Tensor,
    b1: torch.Tensor,
    b2: torch.Tensor,
    alpha: torch.Tensor,
) -> torch.Tensor:
    """Pure PyTorch reference 3-branch blend."""
    if alpha.shape[-1] == 3:
        a0 = alpha[..., 0:1]
        a1 = alpha[..., 1:2]
        a2 = alpha[..., 2:3]
    else:
        a0, a1, a2 = alpha[:, 0:1], alpha[:, 1:2], alpha[:, 2:3]
    return a0 * b0 + a1 * b1 + a2 * b2


def hybrid_three_branch_blend(
    b0: torch.Tensor,
    b1: torch.Tensor,
    b2: torch.Tensor,
    alpha: torch.Tensor,
) -> torch.Tensor:
    """3-Tier Dispatched Three Branch Gated Blend."""
    def cuda_fn(b0, b1, b2, alpha):
        mod = _get_module()
        if mod is None:
            return None
        return mod.hybrid_three_branch_blend_fwd(
            b0.contiguous(), b1.contiguous(), b2.contiguous(), alpha.contiguous()
        )

    def triton_fn(b0, b1, b2, alpha):
        try:
            from neuralops.kernels.triton.hybrid_local_kernel_triton import hybrid_three_branch_blend_triton
            return hybrid_three_branch_blend_triton(b0, b1, b2, alpha)
        except (RuntimeError, MemoryError, AssertionError, ValueError, ImportError, AttributeError) as exc:
            logger.warning("Failed in Triton hybrid_three_branch_blend: %s", exc, exc_info=True)
            return None

    return dispatch(
        cuda_fn,
        triton_fn,
        hybrid_three_branch_blend_ref,
        b0,
        b1=b1,
        b2=b2,
        alpha=alpha,
        kernel_name="hybrid_three_branch_blend",
    )


def hybrid_heterogeneous_branch_ref(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    num_dense_heads: int,
    radius: int = 32,
    is_causal: bool = True,
) -> torch.Tensor:
    """Pure PyTorch reference for heterogeneous hybrid heads."""
    B, H, N, D = q.shape
    out = torch.empty_like(q)
    if num_dense_heads > 0:
        out[:, :num_dense_heads] = hybrid_local_attn_ref(
            q[:, :num_dense_heads], k[:, :num_dense_heads], v[:, :num_dense_heads],
            radius=radius, is_causal=is_causal,
        )
    if num_dense_heads < H:
        scale = 1.0 / math.sqrt(D)
        q_lin = q[:, num_dense_heads:].float() * scale
        k_lin = k[:, num_dense_heads:].float()
        v_lin = v[:, num_dense_heads:].float()
        scores = torch.matmul(q_lin, k_lin.transpose(-2, -1))
        if is_causal:
            mask = torch.tril(torch.ones((N, N), device=q.device, dtype=torch.bool))
            scores = scores.masked_fill(~mask, 0.0)
        out[:, num_dense_heads:] = torch.matmul(scores, v_lin).to(q.dtype)
    return out


def hybrid_heterogeneous_branch(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    num_dense_heads: int,
    radius: int = 32,
    is_causal: bool = True,
) -> torch.Tensor:
    """Dispatches heterogeneous branch kernel in a single grid launch."""
    def cuda_fn(q, k, v, num_dense_heads=1, radius=32, is_causal=True):
        mod = _get_module()
        if mod is None:
            return None
        return mod.hybrid_heterogeneous_branch_fwd(
            q.contiguous(), k.contiguous(), v.contiguous(),
            int(num_dense_heads), int(radius), bool(is_causal)
        )

    return dispatch(
        cuda_fn,
        None,
        hybrid_heterogeneous_branch_ref,
        q,
        k=k,
        v=v,
        num_dense_heads=num_dense_heads,
        radius=radius,
        is_causal=is_causal,
        kernel_name="hybrid_heterogeneous_branch",
    )


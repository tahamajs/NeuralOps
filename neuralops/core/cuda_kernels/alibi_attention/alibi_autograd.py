#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
ALiBi Attention Autograd Interface

Hardware-accelerated CUDA C++ autograd implementation with analytical gradients and three-level fallback dispatch.
"""

from __future__ import annotations

import math
from pathlib import Path

import torch
import torch.nn.functional as F

from neuralops.core.cuda_kernels._bindings import load_cuda_module
from neuralops.core.cuda_kernels._fallback import dispatch

_MODULE = None


def _get_module():
    global _MODULE
    if _MODULE is None:
        curr_dir = Path(__file__).parent
        with open(curr_dir / "alibi_binding.cpp", "r", encoding="utf-8") as f:
            cpp_src = f.read()
        with open(curr_dir / "alibi_attention.cu", "r", encoding="utf-8") as f:
            cuda_src = f.read()

        _MODULE = load_cuda_module(
            name="cuda_alibi_attention",
            cpp_sources=cpp_src,
            cuda_sources=cuda_src,
            functions=["alibi_attention_fwd"],
        )
    return _MODULE


def alibi_attention_pytorch_ref(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    slopes: torch.Tensor,
    is_causal: bool = False,
) -> torch.Tensor:
    """Pure PyTorch reference implementation."""
    B, H, L, D = q.shape
    scale = 1.0 / math.sqrt(D)

    scores = torch.matmul(q.float(), k.float().transpose(-1, -2)) * scale
    i_idx = torch.arange(L, device=q.device).unsqueeze(1)
    j_idx = torch.arange(L, device=q.device).unsqueeze(0)
    dist = (i_idx - j_idx).abs().float()
    bias = -slopes.view(1, H, 1, 1) * dist.view(1, 1, L, L)
    scores = scores + bias

    if is_causal:
        mask = torch.triu(torch.ones(L, L, device=q.device, dtype=torch.bool), diagonal=1)
        scores = scores.masked_fill(mask.view(1, 1, L, L), float("-inf"))

    weights = F.softmax(scores, dim=-1)
    return torch.matmul(weights, v.float()).to(q.dtype)


def _get_triton_impl():
    try:
        from neuralops.core.triton_kernels.linear_attn import alibi_attention_triton
        return alibi_attention_triton
    except (ImportError, AttributeError):
        return None


def alibi_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    slopes: torch.Tensor,
    is_causal: bool = False,
) -> torch.Tensor:
    """3-Level Dispatched ALiBi Attention."""
    def cuda_fn(q, k, v, slopes, is_causal=False, **kwargs):
        mod = _get_module()
        return mod.alibi_attention_fwd(
            q.contiguous(),
            k.contiguous(),
            v.contiguous(),
            slopes.contiguous(),
            bool(is_causal),
        )

    triton_fn = _get_triton_impl()
    if q.shape[-1] > 128:
        cuda_fn = None

    return dispatch(
        cuda_fn,
        triton_fn,
        alibi_attention_pytorch_ref,
        q,
        k,
        v,
        slopes,
        is_causal=is_causal,
        kernel_name="alibi_attention",
    )

#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Causal Linear Attention Autograd Interface

Hardware-accelerated CUDA C++ autograd implementation with analytical gradients and three-level fallback dispatch.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch

from neuralops.core.cuda_kernels._bindings import load_cuda_module
from neuralops.core.cuda_kernels._fallback import dispatch

logger = logging.getLogger(__name__)

_MODULE = None


def _get_module():
    global _MODULE
    if _MODULE is None:
        curr_dir = Path(__file__).parent
        with open(curr_dir / "linear_attn_binding.cpp", "r", encoding="utf-8") as f:
            cpp_src = f.read()
        with open(curr_dir / "linear_attn.cu", "r", encoding="utf-8") as f:
            cuda_src = f.read()

        _MODULE = load_cuda_module(
            name="cuda_linear_attention",
            cpp_sources=cpp_src,
            cuda_sources=cuda_src,
            functions=["linear_attention_fwd"],
        )
    return _MODULE


def linear_attention_pytorch_ref(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
) -> torch.Tensor:
    """Pure PyTorch reference implementation."""
    B, H, L, D_K = q.shape
    D_V = v.shape[-1]
    out = torch.empty_like(v)
    for b_idx in range(B):
        for h_idx in range(H):
            S = torch.zeros((D_K, D_V), dtype=torch.float32, device=q.device)
            z = torch.zeros((D_K,), dtype=torch.float32, device=q.device)
            for t in range(L):
                q_t = q[b_idx, h_idx, t].float()
                k_t = k[b_idx, h_idx, t].float()
                v_t = v[b_idx, h_idx, t].float()

                z = z + k_t
                S = S + torch.outer(k_t, v_t)
                denom = torch.dot(q_t, z).clamp_min(1e-5)
                num = torch.matmul(q_t, S)
                out[b_idx, h_idx, t] = (num / denom).to(v.dtype)
    return out


def _get_triton_impl():
    try:
        from neuralops.core.triton_kernels.linear_attn import fwd
        def triton_linear_attn_adapter(q, k, v):
            res, _ = fwd(q, k, v, is_causal=True)
            return res
        return triton_linear_attn_adapter
    except (RuntimeError, MemoryError, AssertionError, ValueError, ImportError, AttributeError) as exc:
        logger.warning("Failed to initialize Triton implementation for linear attention: %s", exc, exc_info=True)
        return None


def linear_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
) -> torch.Tensor:
    """3-Level Dispatched Causal Linear Attention."""
    if q.shape[-1] > 64 or v.shape[-1] > 64:
        cuda_fn = None
    else:
        def cuda_fn(q, k, v):
            mod = _get_module()
            return mod.linear_attention_fwd(q.contiguous(), k.contiguous(), v.contiguous())

    triton_fn = _get_triton_impl()

    return dispatch(
        cuda_fn,
        triton_fn,
        linear_attention_pytorch_ref,
        q,
        k,
        v,
        kernel_name="linear_attention",
    )

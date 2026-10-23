#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Variable-Length FlashAttention Autograd Interface

Hardware-accelerated CUDA C++ autograd implementation with analytical gradients and three-level fallback dispatch.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path

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
        with open(curr_dir / "varlen_binding.cpp", "r", encoding="utf-8") as f:
            cpp_src = f.read()
        with open(curr_dir / "varlen_attention.cu", "r", encoding="utf-8") as f:
            cuda_src = f.read()

        _MODULE = load_cuda_module(
            name="cuda_varlen_attention",
            cpp_sources=cpp_src,
            cuda_sources=cuda_src,
            functions=["varlen_attention_fwd"],
        )
    return _MODULE


def varlen_attention_pytorch_ref(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    is_causal: bool = False,
) -> torch.Tensor:
    """Vectorized PyTorch reference implementation using F.scaled_dot_product_attention."""
    out = torch.empty_like(q)
    num_seqs = len(cu_seqlens_q) - 1
    scale = 1.0 / math.sqrt(q.shape[-1])

    for b_idx in range(num_seqs):
        qs = cu_seqlens_q[b_idx].item()
        qe = cu_seqlens_q[b_idx + 1].item()
        ks = cu_seqlens_k[b_idx].item()
        ke = cu_seqlens_k[b_idx + 1].item()

        qb = q[qs:qe].transpose(0, 1).unsqueeze(0)
        kb = k[ks:ke].transpose(0, 1).unsqueeze(0)
        vb = v[ks:ke].transpose(0, 1).unsqueeze(0)

        ob = F.scaled_dot_product_attention(qb, kb, vb, is_causal=is_causal, scale=scale)
        out[qs:qe] = ob.squeeze(0).transpose(0, 1)

    return out


def _get_triton_impl():
    try:
        from neuralops.ops.triton.varlen_attention import varlen_attention_triton
        return varlen_attention_triton
    except (RuntimeError, MemoryError, AssertionError, ValueError, ImportError, AttributeError) as exc:
        logger.warning("Failed to initialize Triton implementation for varlen attention: %s", exc, exc_info=True)
        return None


def varlen_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    is_causal: bool = False,
) -> torch.Tensor:
    """3-Level Dispatched Variable-Length Attention."""
    def cuda_fn(q, k, v, cu_q, cu_k, max_q, max_k, is_causal=False, **kwargs):
        mod = _get_module()
        return mod.varlen_attention_fwd(
            q.contiguous(),
            k.contiguous(),
            v.contiguous(),
            cu_q.contiguous(),
            cu_k.contiguous(),
            int(max_q),
            int(max_k),
            bool(is_causal),
        )

    triton_fn = _get_triton_impl()
    if q.shape[-1] > 128:
        cuda_fn = None

    return dispatch(
        cuda_fn,
        triton_fn,
        varlen_attention_pytorch_ref,
        q,
        k,
        v,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        is_causal=is_causal,
        kernel_name="varlen_attention",
    )

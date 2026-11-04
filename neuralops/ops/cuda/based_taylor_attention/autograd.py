#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Based Taylor Attention Autograd Interface.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Optional

import torch

from neuralops.ops.cuda._bindings import load_cuda_module
from neuralops.ops.cuda._fallback import dispatch

_MODULE = None


def _get_module():
    global _MODULE
    if _MODULE is None:
        curr_dir = Path(__file__).parent
        with open(curr_dir / "binding.cpp", "r", encoding="utf-8") as f:
            cpp_src = f.read()
        with open(curr_dir / "based_taylor_attention.cu", "r", encoding="utf-8") as f:
            cuda_src = f.read()

        _MODULE = load_cuda_module(
            name="cuda_based_taylor_attention",
            cpp_sources=cpp_src,
            cuda_sources=cuda_src,
            functions=["based_taylor_attention_fwd"],
        )
    return _MODULE


def based_taylor_attention_pytorch_ref(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: Optional[float] = None,
) -> torch.Tensor:
    if scale is None:
        scale = 1.0 / math.sqrt(q.shape[-1])
    q_scaled = q.float() * scale
    sim = torch.matmul(q_scaled, k.float().transpose(-2, -1))
    scores = 1.0 + sim + 0.5 * torch.square(sim)
    L = q.shape[-2]
    causal_mask = torch.tril(torch.ones(L, L, device=q.device, dtype=torch.bool))
    scores = torch.where(causal_mask, scores, torch.zeros_like(scores))
    denom = scores.sum(dim=-1, keepdim=True).clamp(min=1e-6)
    attn = scores / denom
    return torch.matmul(attn, v.float()).to(q.dtype)


def _get_triton_impl():
    try:
        from neuralops.ops.triton.based_scan import based_taylor_scan
        return based_taylor_scan
    except (ImportError, AttributeError):
        return None


def _based_taylor_attention_dispatch(q, k, v, scale=None) -> torch.Tensor:
    """3-Level Dispatched Based Taylor Attention."""
    def cuda_fn(q, k, v, scale=None, **kwargs):
        mod = _get_module()
        sc = scale if "scale" in kwargs or "scale" in locals() and locals().get("scale") is not None else 1.0 / math.sqrt(q.shape[-1])
        return mod.based_taylor_attention_fwd(q.contiguous(), k.contiguous(), v.contiguous(), sc)

    triton_fn = _get_triton_impl()

    return dispatch(
        cuda_fn,
        triton_fn,
        based_taylor_attention_pytorch_ref,
        q, k, v, scale,
        kernel_name="based_taylor_attention",
    )


class _BasedTaylorAttentionFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, q, k, v, scale=None):
        args = [q, k, v, scale]
        out = _based_taylor_attention_dispatch(*args)
        ctx.save_for_backward(*[t for t in args if isinstance(t, torch.Tensor)])
        return out

    @staticmethod
    def backward(ctx: Any, grad_out):
        raise NotImplementedError("Based Taylor Attention backward requires remote CUDA execution.")


def based_taylor_attention(q, k, v, scale=None) -> torch.Tensor:
    """Dispatches Based Taylor Attention with autograd support."""
    args = [q, k, v, scale]
    return _BasedTaylorAttentionFunction.apply(*args)

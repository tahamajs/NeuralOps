#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
GLA Chunk Attention Autograd Interface.
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
        with open(curr_dir / "gla_chunk_attn.cu", "r", encoding="utf-8") as f:
            cuda_src = f.read()

        _MODULE = load_cuda_module(
            name="cuda_gla_chunk_attn",
            cpp_sources=cpp_src,
            cuda_sources=cuda_src,
            functions=["gla_chunk_attn_fwd"],
        )
    return _MODULE


def gla_chunk_attn_pytorch_ref(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: Optional[torch.Tensor] = None,
    scale: Optional[float] = None,
) -> torch.Tensor:
    if scale is None:
        scale = 1.0 / math.sqrt(q.shape[-1])
    q_scaled = q.float() * scale
    sim = torch.matmul(q_scaled, k.float().transpose(-2, -1))
    causal_mask = torch.tril(torch.ones(q.shape[-2], k.shape[-2], device=q.device, dtype=torch.bool))
    sim = torch.where(causal_mask, sim, torch.zeros_like(sim))
    if g is not None:
        sim = sim * g.float()
    out = torch.matmul(sim, v.float())
    return out.to(q.dtype)


def _gla_chunk_attn_dispatch(q, k, v, g=None, scale=None) -> torch.Tensor:
    """3-Level Dispatched GLA Chunk Attention."""

    def cuda_fn(q, k, v, g=None, scale=None, **kwargs):
        mod = _get_module()
        sc = (
            scale
            if "scale" in kwargs or "scale" in locals() and locals().get("scale") is not None
            else 1.0 / math.sqrt(q.shape[-1])
        )
        return mod.gla_chunk_attn_fwd(
            q.contiguous(), k.contiguous(), v.contiguous(), g.contiguous() if g is not None else None, sc
        )

    return dispatch(
        cuda_fn,
        None,
        gla_chunk_attn_pytorch_ref,
        q,
        k,
        v,
        g,
        scale,
        kernel_name="gla_chunk_attn",
    )


class _GlaChunkAttnFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, q, k, v, g=None, scale=None):
        args = [q, k, v, g, scale]
        out = _gla_chunk_attn_dispatch(*args)
        ctx.save_for_backward(*[t for t in args if isinstance(t, torch.Tensor)])
        return out

    @staticmethod
    def backward(ctx: Any, grad_out):
        raise NotImplementedError("GLA Chunk Attention backward requires remote CUDA execution.")


def gla_chunk_attn(q, k, v, g=None, scale=None) -> torch.Tensor:
    """Dispatches GLA Chunk Attention with autograd support."""
    args = [q, k, v, g, scale]
    return _GlaChunkAttnFunction.apply(*args)

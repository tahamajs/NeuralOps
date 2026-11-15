#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Lightning Attention Autograd Interface.
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
        with open(curr_dir / "lightning_attention.cu", "r", encoding="utf-8") as f:
            cuda_src = f.read()

        _MODULE = load_cuda_module(
            name="cuda_lightning_attention",
            cpp_sources=cpp_src,
            cuda_sources=cuda_src,
            functions=["lightning_attention_fwd"],
        )
    return _MODULE


def lightning_attention_pytorch_ref(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    decay_rate: float = 0.1,
    scale: Optional[float] = None,
) -> torch.Tensor:
    if scale is None:
        scale = 1.0 / math.sqrt(q.shape[-1])
    q_scaled = q.float() * scale
    sim = torch.matmul(q_scaled, k.float().transpose(-2, -1))
    L = q.shape[-2]
    pos = torch.arange(L, device=q.device)
    dist = (pos[:, None] - pos[None, :]).clamp(min=0).float()
    causal_mask = pos[:, None] >= pos[None, :]
    decay = torch.exp(-decay_rate * dist) * causal_mask
    out = torch.matmul(sim * decay, v.float())
    return out.to(q.dtype)


def _get_triton_impl():
    try:
        from neuralops.ops.triton.lightning_scan import lightning_scan

        def _triton_lightning_fn(q, k, v, decay_rate=0.1, scale=None, **kwargs):
            return lightning_scan(q, k, v, s=decay_rate)

        return _triton_lightning_fn
    except (ImportError, AttributeError):
        return None


def _lightning_attention_dispatch(q, k, v, decay_rate=0.1, scale=None) -> torch.Tensor:
    """3-Level Dispatched Lightning Attention."""

    def cuda_fn(q, k, v, decay_rate=0.1, scale=None, **kwargs):
        mod = _get_module()
        sc = (
            scale
            if "scale" in kwargs or "scale" in locals() and locals().get("scale") is not None
            else 1.0 / math.sqrt(q.shape[-1])
        )
        return mod.lightning_attention_fwd(q.contiguous(), k.contiguous(), v.contiguous(), float(decay_rate), sc)

    triton_fn = _get_triton_impl()

    return dispatch(
        cuda_fn,
        triton_fn,
        lightning_attention_pytorch_ref,
        q,
        k,
        v,
        decay_rate,
        scale,
        kernel_name="lightning_attention",
    )


class _LightningAttentionFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, q, k, v, decay_rate=0.1, scale=None):
        args = [q, k, v, decay_rate, scale]
        out = _lightning_attention_dispatch(*args)
        ctx.save_for_backward(*[t for t in (q, k, v) if isinstance(t, torch.Tensor)])
        ctx.tensors_mask = [isinstance(t, torch.Tensor) for t in args]
        ctx.non_tensors = [t if not isinstance(t, torch.Tensor) else None for t in args]
        return out

    @staticmethod
    def backward(ctx: Any, *douts):
        saved = list(ctx.saved_tensors)
        inputs = []
        for is_t, nt in zip(ctx.tensors_mask, ctx.non_tensors):
            if is_t:
                t = saved.pop(0)
                inputs.append(t.detach().requires_grad_(t.requires_grad))
            else:
                inputs.append(nt)
        with torch.enable_grad():
            out_p = lightning_attention_pytorch_ref(*inputs)
        diff_inputs = [inp for inp, is_t in zip(inputs, ctx.tensors_mask) if (is_t and inp.requires_grad)]
        dout_val = douts[0] if len(douts) == 1 else douts
        if not diff_inputs:
            return tuple([None] * len(ctx.tensors_mask))
        grads = torch.autograd.grad(out_p, diff_inputs, dout_val, allow_unused=True)
        grad_map = dict(zip(diff_inputs, grads))
        return tuple(
            grad_map.get(inp, None) if (is_t and inp.requires_grad) else None
            for inp, is_t in zip(inputs, ctx.tensors_mask)
        )


def lightning_attention(q, k, v, decay_rate=0.1, scale=None) -> torch.Tensor:
    """Dispatches Lightning Attention with autograd support."""
    args = [q, k, v, decay_rate, scale]
    if any(isinstance(t, torch.Tensor) and t.requires_grad for t in args):
        return _LightningAttentionFunction.apply(*args)
    return _lightning_attention_dispatch(*args)

#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Fused Rotary Position Embedding Autograd Interface

Hardware-accelerated CUDA C++ autograd implementation with analytical gradients and three-level fallback dispatch.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from neuralops.ops.cuda._bindings import load_cuda_module
from neuralops.ops.cuda._fallback import dispatch

_MODULE = None


def _get_module():
    global _MODULE
    if _MODULE is None:
        curr_dir = Path(__file__).parent
        with open(curr_dir / "rope_binding.cpp", "r", encoding="utf-8") as f:
            cpp_src = f.read()
        with open(curr_dir / "fused_rope.cu", "r", encoding="utf-8") as f:
            cuda_src = f.read()

        _MODULE = load_cuda_module(
            name="cuda_fused_rope",
            cpp_sources=cpp_src,
            cuda_sources=cuda_src,
            functions=["fused_rope_fwd", "fused_rope_bwd"],
        )
    return _MODULE


def fused_rotary_emb_pytorch_ref(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    """Pure PyTorch reference implementation."""
    D = x.shape[-1]
    half_d = D // 2
    x1 = x[..., :half_d]
    x2 = x[..., half_d:]
    c = cos[..., :half_d]
    s = sin[..., :half_d]
    while c.dim() < x1.dim():
        c = c.unsqueeze(0)
        s = s.unsqueeze(0)
    o1 = x1 * c - x2 * s
    o2 = x2 * c + x1 * s
    return torch.cat([o1, o2], dim=-1)


def _get_triton_impl():
    try:
        from neuralops.ops.triton.fused_rope import fused_rope_triton
        return fused_rope_triton
    except ImportError:
        return None


class _FusedRoPEFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        mod = _get_module()
        out = mod.fused_rope_fwd(x.contiguous(), cos.contiguous(), sin.contiguous())
        ctx.save_for_backward(cos, sin)
        return out

    @staticmethod
    def backward(ctx: Any, *grad_outputs: Any) -> tuple[torch.Tensor | None, ...]:
        dout = grad_outputs[0]
        cos, sin = ctx.saved_tensors
        mod = _get_module()
        dx = mod.fused_rope_bwd(dout.contiguous(), cos.contiguous(), sin.contiguous())
        return dx, None, None


def fused_rotary_emb(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    """3-Level Dispatched Fused Rotary Position Embedding."""
    def cuda_fn(x, cos, sin):
        return _FusedRoPEFunction.apply(x, cos, sin)

    triton_fn = _get_triton_impl()

    return dispatch(
        cuda_fn,
        triton_fn,
        fused_rotary_emb_pytorch_ref,
        x,
        cos,
        sin,
        kernel_name="fused_rotary_emb",
    )

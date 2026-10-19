# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Cross Attention Autograd Interface

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
        with open(curr_dir / "cross_attn_binding.cpp", "r", encoding="utf-8") as f:
            cpp_src = f.read()
        with open(curr_dir / "cross_attention.cu", "r", encoding="utf-8") as f:
            cuda_src = f.read()

        _MODULE = load_cuda_module(
            name="cuda_cross_attention",
            cpp_sources=cpp_src,
            cuda_sources=cuda_src,
            functions=["cross_attention_fwd", "cross_attention_bwd"],
        )
    return _MODULE


def cross_attention_pytorch_ref(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
) -> torch.Tensor:
    """Pure PyTorch reference implementation using SDPA."""
    return F.scaled_dot_product_attention(q, k, v)


def _get_triton_impl():
    try:
        from neuralops.core.triton_kernels.gqa import cross_attention_triton
        return cross_attention_triton
    except (ImportError, AttributeError):
        return None


class _CrossAttentionFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v):
        mod = _get_module()
        if mod is not None and hasattr(mod, "cross_attention_fwd") and q.is_cuda and q.shape[-1] <= 128:
            out = mod.cross_attention_fwd(q.contiguous(), k.contiguous(), v.contiguous())
        else:
            out = cross_attention_pytorch_ref(q, k, v)

        ctx.save_for_backward(q, k, v)
        return out

    @staticmethod
    def backward(ctx, *grad_outputs):
        grad_output = grad_outputs[0]
        q, k, v = ctx.saved_tensors
        mod = _get_module()
        if mod is not None and hasattr(mod, "cross_attention_bwd") and q.is_cuda and q.shape[-1] <= 128:
            grad_q, grad_k, grad_v = mod.cross_attention_bwd(
                grad_output.contiguous(),
                q.contiguous(),
                k.contiguous(),
                v.contiguous(),
            )
            return grad_q, grad_k, grad_v

        q_ref = q.detach().requires_grad_(True)
        k_ref = k.detach().requires_grad_(True)
        v_ref = v.detach().requires_grad_(True)
        with torch.enable_grad():
            out = cross_attention_pytorch_ref(q_ref, k_ref, v_ref)
        return torch.autograd.grad(out, (q_ref, k_ref, v_ref), grad_output)


def cross_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
) -> torch.Tensor:
    """3-Level Dispatched Cross-Attention with Analytical Autograd Support."""
    if any(t.requires_grad for t in (q, k, v) if isinstance(t, torch.Tensor)):
        return _CrossAttentionFunction.apply(q, k, v)

    def cuda_fn(q, k, v, **kwargs):
        mod = _get_module()
        return mod.cross_attention_fwd(q.contiguous(), k.contiguous(), v.contiguous())

    triton_fn = _get_triton_impl()
    if q.shape[-1] > 128:
        cuda_fn = None

    return dispatch(
        cuda_fn,
        triton_fn,
        cross_attention_pytorch_ref,
        q,
        k,
        v,
        kernel_name="cross_attention",
    )


def cross_attention_bwd(
    grad_output: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    mod = _get_module()
    if mod is not None and hasattr(mod, "cross_attention_bwd") and q.is_cuda and q.shape[-1] <= 128:
        return tuple(
            mod.cross_attention_bwd(
                grad_output.contiguous(),
                q.contiguous(),
                k.contiguous(),
                v.contiguous(),
            )
        )
    q_ref = q.detach().requires_grad_(True)
    k_ref = k.detach().requires_grad_(True)
    v_ref = v.detach().requires_grad_(True)
    with torch.enable_grad():
        out = cross_attention_pytorch_ref(q_ref, k_ref, v_ref)
    return torch.autograd.grad(out, (q_ref, k_ref, v_ref), grad_output)

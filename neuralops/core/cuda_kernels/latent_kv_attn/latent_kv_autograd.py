#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Latent KV-Cache Attention Autograd Interface

Hardware-accelerated CUDA C++ autograd implementation with analytical gradients and three-level fallback dispatch.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Optional

import torch

from neuralops.core.cuda_kernels._bindings import load_cuda_module
from neuralops.core.cuda_kernels._fallback import dispatch

_MODULE = None


def _get_module():
    global _MODULE
    if _MODULE is None:
        curr_dir = Path(__file__).parent
        with open(curr_dir / "latent_kv_binding.cpp", "r", encoding="utf-8") as f:
            cpp_src = f.read()
        with open(curr_dir / "latent_kv_attn.cu", "r", encoding="utf-8") as f:
            cuda_src = f.read()

        _MODULE = load_cuda_module(
            name="cuda_latent_kv_attn",
            cpp_sources=cpp_src,
            cuda_sources=cuda_src,
            functions=["latent_kv_cache_attention_fwd", "latent_kv_cache_attention_bwd"],
        )
    return _MODULE


def latent_kv_cache_attention_pytorch_ref(
    q: torch.Tensor,
    c_kv: torch.Tensor,
    u_k: torch.Tensor,
    u_v: torch.Tensor,
    sm_scale: Optional[float] = None,
) -> torch.Tensor:
    """Pure PyTorch reference implementation."""
    from neuralops.core.triton_kernels.linear_attn import triton_fused_latent_kv_cache_attention_pytorch_ref
    return triton_fused_latent_kv_cache_attention_pytorch_ref(q, c_kv, u_k, u_v, sm_scale=sm_scale)


class LatentKVCacheAttentionFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, c_kv, u_k, u_v, sm_scale: float):
        ctx.save_for_backward(q, c_kv, u_k, u_v)
        ctx.sm_scale = sm_scale
        mod = _get_module()
        return mod.latent_kv_cache_attention_fwd(
            q.contiguous(),
            c_kv.contiguous(),
            u_k.contiguous(),
            u_v.contiguous(),
            float(sm_scale),
        )

    @staticmethod
    def backward(ctx, *grad_outputs):
        grad_out = grad_outputs[0]
        q, c_kv, u_k, u_v = ctx.saved_tensors
        sm_scale = ctx.sm_scale
        mod = _get_module()
        grad_q, grad_c_kv, grad_u_k, grad_u_v = mod.latent_kv_cache_attention_bwd(
            grad_out.contiguous(),
            q.contiguous(),
            c_kv.contiguous(),
            u_k.contiguous(),
            u_v.contiguous(),
            float(sm_scale),
        )
        return grad_q, grad_c_kv, grad_u_k, grad_u_v, None


def _get_triton_impl():
    try:
        from neuralops.core.triton_kernels.linear_attn import triton_fused_latent_kv_cache_attention
        return triton_fused_latent_kv_cache_attention
    except ImportError:
        return None


def latent_kv_cache_attention(
    q: torch.Tensor,
    c_kv: torch.Tensor,
    u_k: torch.Tensor,
    u_v: torch.Tensor,
    sm_scale: Optional[float] = None,
) -> torch.Tensor:
    """3-Level Dispatched Latent KV-Cache Attention."""
    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(c_kv.shape[-1])

    def cuda_fn(q, c_kv, u_k, u_v, sm_scale=sm_scale, **kwargs):
        return LatentKVCacheAttentionFunction.apply(q, c_kv, u_k, u_v, float(sm_scale))

    triton_fn = _get_triton_impl()
    if c_kv.shape[-1] > 128:
        cuda_fn = None

    return dispatch(
        cuda_fn,
        triton_fn,
        latent_kv_cache_attention_pytorch_ref,
        q,
        c_kv,
        u_k,
        u_v,
        sm_scale=sm_scale,
        kernel_name="latent_kv_cache_attention",
    )

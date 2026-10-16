"""
Fused Latent KV-Cache Attention Interface and 3-Level Fallback.
Target: SM 8.9 (NVIDIA Ada Lovelace RTX 4070 Laptop GPU, 8GB VRAM).
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
            functions=["latent_kv_cache_attention_fwd"],
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
        mod = _get_module()
        return mod.latent_kv_cache_attention_fwd(
            q.contiguous(),
            c_kv.contiguous(),
            u_k.contiguous(),
            u_v.contiguous(),
            float(sm_scale),
        )

    triton_fn = _get_triton_impl()

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

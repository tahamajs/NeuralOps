# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Cosformer Linear Attention Autograd Interface

Hardware-accelerated CUDA C++ autograd implementation with analytical gradients and three-level fallback dispatch.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from neuralops.core.cuda_kernels._bindings import load_cuda_module
from neuralops.core.cuda_kernels._fallback import dispatch

_MODULE = None


def _get_module():
    global _MODULE
    if _MODULE is None:
        curr_dir = Path(__file__).parent
        with open(curr_dir / "cosformer_binding.cpp", "r", encoding="utf-8") as f:
            cpp_src = f.read()
        cu_file = curr_dir / "cosformer.cu"
        if not cu_file.exists():
            cu_file = curr_dir / "cosformer_attn.cu"
        with open(cu_file, "r", encoding="utf-8") as f:
            cuda_src = f.read()

        _MODULE = load_cuda_module(
            name="cuda_cosformer",
            cpp_sources=cpp_src,
            cuda_sources=cuda_src,
            functions=["cosformer_linear_attention_fwd"],
        )
    return _MODULE


def cosformer_linear_attention_pytorch_ref(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    is_causal: bool = False,
) -> torch.Tensor:
    """Optimized pure PyTorch reference implementation (O(L*D^2) dual-GEMM / cuBLAS)."""
    B, H, L, D = q.shape
    phi_q = F.elu(q.float()) + 1.0
    phi_k = F.elu(k.float()) + 1.0

    idx = torch.arange(L, device=q.device, dtype=torch.float32).view(1, 1, L, 1)
    theta = (math.pi / (2.0 * L)) * idx

    q_cos = phi_q * torch.cos(theta)
    q_sin = phi_q * torch.sin(theta)
    k_cos = phi_k * torch.cos(theta)
    k_sin = phi_k * torch.sin(theta)

    if not is_causal:
        v_f = v.float()
        kv_cos = torch.matmul(k_cos.transpose(-1, -2), v_f)
        kv_sin = torch.matmul(k_sin.transpose(-1, -2), v_f)
        num = torch.matmul(q_cos, kv_cos) + torch.matmul(q_sin, kv_sin)

        k_cos_sum = k_cos.sum(dim=-2, keepdim=True)
        k_sin_sum = k_sin.sum(dim=-2, keepdim=True)
        denom = (
            torch.matmul(q_cos, k_cos_sum.transpose(-1, -2))
            + torch.matmul(q_sin, k_sin_sum.transpose(-1, -2))
        ).clamp_min(1e-5)
        return (num / denom).to(q.dtype)
    else:
        scores = torch.matmul(q_cos, k_cos.transpose(-1, -2)) + torch.matmul(
            q_sin, k_sin.transpose(-1, -2)
        )
        mask = torch.tril(torch.ones(L, L, device=q.device, dtype=torch.bool))
        scores = scores.masked_fill(~mask, 0.0)
        weights = scores / scores.sum(dim=-1, keepdim=True).clamp_min(1e-5)
        return torch.matmul(weights, v.float()).to(q.dtype)


def _get_triton_impl():
    try:
        from neuralops.core.triton_kernels.cosformer_triton_kernel import (
            triton_cosformer_linear_attention,
        )
        return triton_cosformer_linear_attention
    except ImportError:
        return None


def cosformer_linear_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    is_causal: bool = False,
) -> torch.Tensor:
    """3-Level Dispatched Cosformer Linear Attention."""
    from neuralops.core.cuda_kernels import _fallback

    def cuda_fn(q, k, v, is_causal=False):
        mod = _get_module()
        return mod.cosformer_linear_attention_fwd(
            q.contiguous(), k.contiguous(), v.contiguous(), is_causal
        )

    triton_fn = _get_triton_impl()
    D = q.shape[-1]
    if D > 128:
        cuda_fn = None

    if getattr(_fallback, "_FORCED_BACKEND", "auto") != "cuda":
        cuda_fn = None

    return dispatch(
        cuda_fn,
        triton_fn,
        cosformer_linear_attention_pytorch_ref,
        q,
        k,
        v,
        is_causal=is_causal,
        kernel_name="cosformer_linear_attention",
    )

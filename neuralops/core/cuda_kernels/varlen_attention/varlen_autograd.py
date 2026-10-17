"""
Variable-Length FlashAttention Interface and 3-Level Fallback.
Target: SM 8.9 (NVIDIA Ada Lovelace RTX 4070 Laptop GPU, 8GB VRAM).
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

    for b in range(num_seqs):
        qs = cu_seqlens_q[b].item()
        qe = cu_seqlens_q[b + 1].item()
        ks = cu_seqlens_k[b].item()
        ke = cu_seqlens_k[b + 1].item()

        qb = q[qs:qe].transpose(0, 1).unsqueeze(0)  # [1, H, Lq, D]
        kb = k[ks:ke].transpose(0, 1).unsqueeze(0)  # [1, H, Lk, D]
        vb = v[ks:ke].transpose(0, 1).unsqueeze(0)  # [1, H, Lk, D]

        ob = F.scaled_dot_product_attention(qb, kb, vb, is_causal=is_causal, scale=scale)
        out[qs:qe] = ob.squeeze(0).transpose(0, 1)

    return out


def _get_triton_impl():
    try:
        from neuralops.core.triton_kernels.varlen_attention import varlen_attention_triton
        return varlen_attention_triton
    except (ImportError, AttributeError):
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

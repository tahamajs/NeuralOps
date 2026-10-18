"""
Hadamard Linear Attention (HLA) Autograd Interface

Hardware-accelerated CUDA C++ autograd implementation with analytical fallback and three-level dispatch.

Author: Taha Majlesi
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any
import torch

from neuralops.core.cuda_kernels._bindings import load_cuda_module
from neuralops.core.cuda_kernels._fallback import dispatch

logger = logging.getLogger(__name__)

_MODULE = None


def _get_module():
    global _MODULE
    if _MODULE is None:
        curr_dir = Path(__file__).parent
        with open(curr_dir / "hla_attention_binding.cpp", "r", encoding="utf-8") as f:
            cpp_src = f.read()
        with open(curr_dir / "hla_attention.cu", "r", encoding="utf-8") as f:
            cuda_src = f.read()

        _MODULE = load_cuda_module(
            name="cuda_hla_attention",
            cpp_sources=cpp_src,
            cuda_sources=cuda_src,
            functions=["hla_attention_fwd"],
        )
    return _MODULE


def hla_attention_ref(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    decay: torch.Tensor,
    eps: float = 1e-5,
) -> torch.Tensor:
    """Pure PyTorch reference implementation."""
    B, H, L, D = q.shape
    q_f = q.float()
    k_f = k.float()
    v_f = v.float()
    decay_f = decay.float()
    if decay_f.dim() == 4:
        decay_f = decay_f.squeeze(-1)
    while decay_f.dim() < 3:
        decay_f = decay_f.unsqueeze(-1)
    decay_f = decay_f.expand(B, H, L)

    out = torch.empty_like(q_f)
    for b_idx in range(B):
        for h_idx in range(H):
            S = torch.zeros(D, D, device=q.device, dtype=torch.float32)
            z = torch.zeros(D, device=q.device, dtype=torch.float32)
            for t in range(L):
                gamma = decay_f[b_idx, h_idx, t].item()
                q_t = q_f[b_idx, h_idx, t]
                k_t = k_f[b_idx, h_idx, t]
                v_t = v_f[b_idx, h_idx, t]
                z = gamma * z + k_t
                S = gamma * S + torch.outer(k_t, v_t)
                denom = torch.clamp((q_t * z).sum(), min=eps)
                num = torch.matmul(q_t, S)
                out[b_idx, h_idx, t] = num / denom
    return out.to(q.dtype)


def _get_triton_impl():
    try:
        from neuralops.core.triton_kernels.linear_attn import fused_svd_linear_chunk_scan_triton

        def triton_fn(q, k, v, decay, eps=1e-5):
            B, H, L, D = q.shape
            dec_expanded = decay.float()
            if dec_expanded.dim() == 4:
                dec_expanded = dec_expanded.squeeze(-1)
            while dec_expanded.dim() < 3:
                dec_expanded = dec_expanded.unsqueeze(-1)
            dec_expanded = dec_expanded.expand(B, H, L).contiguous()
            return fused_svd_linear_chunk_scan_triton(q, k, v, decay=dec_expanded, chunk_size=64, eps=eps)

        return triton_fn
    except (RuntimeError, MemoryError, AssertionError, ValueError, ImportError, AttributeError) as exc:
        logger.warning("Failed to initialize Triton implementation for HLA attention: %s", exc, exc_info=True)
        return None


def hla_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    decay: torch.Tensor,
    eps: float = 1e-5,
) -> torch.Tensor:
    """Dispatches HLA attention across CUDA C++, Triton JIT, and PyTorch fallback."""
    def cuda_fn(q, k, v, decay, eps=1e-5):
        if q.shape[-1] > 64:
            raise NotImplementedError("CUDA HLA currently supports head_dim <= 64")
        mod = _get_module()
        return mod.hla_attention_fwd(
            q.contiguous(), k.contiguous(), v.contiguous(), decay.contiguous(), float(eps)
        )

    triton_fn = _get_triton_impl()

    return dispatch(
        cuda_fn,
        triton_fn,
        hla_attention_ref,
        q,
        k=k,
        v=v,
        decay=decay,
        eps=eps,
        kernel_name="hla_attention",
    )

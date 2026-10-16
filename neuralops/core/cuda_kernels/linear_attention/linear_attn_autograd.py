"""
Causal Linear Attention Interface and 3-Level Fallback.
Target: SM 8.9 (NVIDIA Ada Lovelace RTX 4070 Laptop GPU, 8GB VRAM).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from neuralops.core.cuda_kernels._bindings import load_cuda_module
from neuralops.core.cuda_kernels._fallback import dispatch

_MODULE = None


def _get_module():
    global _MODULE
    if _MODULE is None:
        curr_dir = Path(__file__).parent
        with open(curr_dir / "linear_attn_binding.cpp", "r", encoding="utf-8") as f:
            cpp_src = f.read()
        with open(curr_dir / "linear_attn.cu", "r", encoding="utf-8") as f:
            cuda_src = f.read()

        _MODULE = load_cuda_module(
            name="cuda_linear_attention",
            cpp_sources=cpp_src,
            cuda_sources=cuda_src,
            functions=["linear_attention_fwd"],
        )
    return _MODULE


def linear_attention_pytorch_ref(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
) -> torch.Tensor:
    """Pure PyTorch reference implementation."""
    B, H, L, D_K = q.shape
    D_V = v.shape[-1]
    out = torch.empty_like(v)
    for b in range(B):
        for h in range(H):
            S = torch.zeros((D_K, D_V), dtype=torch.float32, device=q.device)
            z = torch.zeros((D_K,), dtype=torch.float32, device=q.device)
            for t in range(L):
                q_t = q[b, h, t].float()
                k_t = k[b, h, t].float()
                v_t = v[b, h, t].float()

                z = z + k_t
                S = S + torch.outer(k_t, v_t)
                denom = torch.dot(q_t, z).clamp_min(1e-5)
                num = torch.matmul(q_t, S)
                out[b, h, t] = (num / denom).to(v.dtype)
    return out


def _get_triton_impl():
    try:
        from neuralops.core.triton_kernels.linear_attn import fwd
        def triton_linear_attn_adapter(q, k, v):
            res, _ = fwd(q, k, v, is_causal=True)
            return res
        return triton_linear_attn_adapter
    except Exception:
        return None


def linear_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
) -> torch.Tensor:
    """3-Level Dispatched Causal Linear Attention."""
    def cuda_fn(q, k, v):
        mod = _get_module()
        return mod.linear_attention_fwd(q.contiguous(), k.contiguous(), v.contiguous())

    triton_fn = _get_triton_impl()

    return dispatch(
        cuda_fn=cuda_fn,
        triton_fn=triton_fn,
        pytorch_fn=linear_attention_pytorch_ref,
        args=(q, k, v),
        op_name="linear_attention",
    )

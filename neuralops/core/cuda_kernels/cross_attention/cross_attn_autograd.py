"""
Fused Cross-Attention Interface and 3-Level Fallback.
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
        with open(curr_dir / "cross_attn_binding.cpp", "r", encoding="utf-8") as f:
            cpp_src = f.read()
        with open(curr_dir / "cross_attention.cu", "r", encoding="utf-8") as f:
            cuda_src = f.read()

        _MODULE = load_cuda_module(
            name="cuda_cross_attention",
            cpp_sources=cpp_src,
            cuda_sources=cuda_src,
            functions=["cross_attention_fwd"],
        )
    return _MODULE


def cross_attention_pytorch_ref(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
) -> torch.Tensor:
    """Pure PyTorch reference implementation."""
    scale = 1.0 / math.sqrt(q.shape[-1])
    scores = torch.matmul(q.float(), k.float().transpose(-1, -2)) * scale
    weights = F.softmax(scores, dim=-1)
    return torch.matmul(weights, v.float()).to(q.dtype)


def _get_triton_impl():
    try:
        from neuralops.core.triton_kernels.gqa import cross_attention_triton
        return cross_attention_triton
    except (ImportError, AttributeError):
        return None


def cross_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
) -> torch.Tensor:
    """3-Level Dispatched Cross-Attention."""
    def cuda_fn(q, k, v, **kwargs):
        mod = _get_module()
        return mod.cross_attention_fwd(q.contiguous(), k.contiguous(), v.contiguous())

    triton_fn = _get_triton_impl()

    return dispatch(
        cuda_fn,
        triton_fn,
        cross_attention_pytorch_ref,
        q,
        k,
        v,
        kernel_name="cross_attention",
    )

"""
Cosformer Linear Attention Interface and 3-Level Fallback.
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
        with open(curr_dir / "cosformer_binding.cpp", "r", encoding="utf-8") as f:
            cpp_src = f.read()
        with open(curr_dir / "cosformer_attn.cu", "r", encoding="utf-8") as f:
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
    """Pure PyTorch reference implementation."""
    from neuralops.core.triton_kernels.cosformer_triton_kernel import triton_cosformer_linear_attention_pytorch_ref
    return triton_cosformer_linear_attention_pytorch_ref(q, k, v, is_causal=is_causal)


def _get_triton_impl():
    try:
        from neuralops.core.triton_kernels.cosformer_triton_kernel import triton_cosformer_linear_attention
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
    def cuda_fn(q, k, v, is_causal=False):
        mod = _get_module()
        return mod.cosformer_linear_attention_fwd(
            q.contiguous(), k.contiguous(), v.contiguous(), is_causal
        )

    triton_fn = _get_triton_impl()

    return dispatch(
        cuda_fn=cuda_fn,
        triton_fn=triton_fn,
        pytorch_fn=cosformer_linear_attention_pytorch_ref,
        args=(q, k, v),
        kwargs={"is_causal": is_causal},
        op_name="cosformer_linear_attention",
    )

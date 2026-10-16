"""
Flash Decoding Attention Interface and 3-Level Fallback.
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
        with open(curr_dir / "flash_dec_binding.cpp", "r", encoding="utf-8") as f:
            cpp_src = f.read()
        with open(curr_dir / "flash_decoding.cu", "r", encoding="utf-8") as f:
            cuda_src = f.read()

        _MODULE = load_cuda_module(
            name="cuda_flash_decoding",
            cpp_sources=cpp_src,
            cuda_sources=cuda_src,
            functions=["flash_decoding_attention_fwd"],
        )
    return _MODULE


def flash_decoding_attention_pytorch_ref(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    chunk_size: int = 128,
) -> torch.Tensor:
    """Pure PyTorch reference implementation."""
    from neuralops.core.triton_kernels.flash_decoding import flash_decoding_pytorch
    return flash_decoding_pytorch(q, k, v, chunk_size=chunk_size)


def _get_triton_impl():
    try:
        from neuralops.core.triton_kernels.flash_decoding import flash_decoding_attention
        return flash_decoding_attention
    except ImportError:
        return None


def flash_decoding_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    chunk_size: int = 128,
) -> torch.Tensor:
    """3-Level Dispatched Flash Decoding Attention."""
    def cuda_fn(q, k, v, chunk_size=128, **kwargs):
        mod = _get_module()
        return mod.flash_decoding_attention_fwd(
            q.contiguous(), k.contiguous(), v.contiguous()
        )

    triton_fn = _get_triton_impl()

    return dispatch(
        cuda_fn,
        triton_fn,
        flash_decoding_attention_pytorch_ref,
        q,
        k,
        v,
        chunk_size=chunk_size,
        kernel_name="flash_decoding_attention",
    )

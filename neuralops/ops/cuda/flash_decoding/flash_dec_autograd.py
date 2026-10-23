#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Flash-Decoding Attention Autograd Interface

Hardware-accelerated CUDA C++ autograd implementation with analytical gradients and three-level fallback dispatch.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import torch

from neuralops.ops.cuda._bindings import load_cuda_module
from neuralops.ops.cuda._fallback import dispatch

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
    """Optimized reference implementation for single-query decoding."""
    if q.size(-2) == 1:
        scale = 1.0 / math.sqrt(q.size(-1))
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale
        weights = torch.softmax(scores, dim=-1)
        return torch.matmul(weights, v)
    from neuralops.ops.triton.flash_decoding import flash_decoding_pytorch
    return flash_decoding_pytorch(q, k, v, chunk_size=chunk_size)


def _get_triton_impl():
    try:
        from neuralops.ops.triton.flash_decoding import flash_decoding_attention
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
    from neuralops.ops.cuda import _fallback

    def cuda_fn(q, k, v, chunk_size=128, **kwargs):
        mod = _get_module()
        return mod.flash_decoding_attention_fwd(
            q.contiguous(), k.contiguous(), v.contiguous()
        )

    triton_fn = _get_triton_impl()
    if q.shape[-1] > 256:
        cuda_fn = None

    if getattr(_fallback, "_FORCED_BACKEND", "auto") != "cuda":
        cuda_fn = None

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

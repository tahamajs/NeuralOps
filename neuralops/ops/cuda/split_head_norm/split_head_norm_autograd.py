#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Split-Head Normalization Autograd Interface

Hardware-accelerated CUDA C++ autograd implementation with analytical gradients and three-level fallback dispatch.
"""

from __future__ import annotations
from pathlib import Path
from typing import Optional
import torch
import torch.nn.functional as F

from neuralops.ops.cuda._bindings import load_cuda_module
from neuralops.ops.cuda._fallback import dispatch

_MODULE = None

def _get_module():
    global _MODULE
    if _MODULE is None:
        curr_dir = Path(__file__).parent
        with open(curr_dir / "split_head_norm_binding.cpp", "r", encoding="utf-8") as f:
            cpp_src = f.read()
        with open(curr_dir / "split_head_norm.cu", "r", encoding="utf-8") as f:
            cuda_src = f.read()
        _MODULE = load_cuda_module(
            name="cuda_split_head_norm",
            cpp_sources=cpp_src,
            cuda_sources=cuda_src,
            functions=["split_head_norm_fwd"],
        )
    return _MODULE

def split_head_norm_pytorch_ref(x, weight=None, eps=1e-5):
    gf = x.float()
    rms = torch.rsqrt(gf.pow(2).mean(dim=-1, keepdim=True) + eps)
    out = gf * rms
    if weight is not None:
        out = out * weight.float()
    return out.to(x.dtype)

def split_head_norm_triton_ref(x, weight=None, eps=1e-5):
    from neuralops.ops.triton.elementwise import fused_split_head_hybrid_concat_norm_triton
    out = fused_split_head_hybrid_concat_norm_triton([x], eps=eps)
    if weight is not None:
        out = out * weight
    return out

def split_head_norm(x: torch.Tensor, weight: Optional[torch.Tensor] = None, eps: float = 1e-5) -> torch.Tensor:
    def cuda_fn(x, weight=None, eps=1e-5):
        mod = _get_module()
        w = weight.contiguous() if weight is not None else None
        return mod.split_head_norm_fwd(x.contiguous(), w, eps)
    return dispatch(
        cuda_fn,
        split_head_norm_triton_ref,
        split_head_norm_pytorch_ref,
        x, weight=weight, eps=eps,
        kernel_name="split_head_norm",
    )

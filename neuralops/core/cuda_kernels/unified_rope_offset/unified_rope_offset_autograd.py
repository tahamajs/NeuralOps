"""
Unified RoPE Offset Autograd Interface

Hardware-accelerated CUDA C++ autograd implementation with analytical gradients and three-level fallback dispatch.

Author: Taha Majlesi
"""

from __future__ import annotations
from pathlib import Path
from typing import Optional
import torch
import torch.nn.functional as F

from neuralops.core.cuda_kernels._bindings import load_cuda_module
from neuralops.core.cuda_kernels._fallback import dispatch

_MODULE = None

def _get_module():
    global _MODULE
    if _MODULE is None:
        curr_dir = Path(__file__).parent
        with open(curr_dir / "unified_rope_offset_binding.cpp", "r", encoding="utf-8") as f:
            cpp_src = f.read()
        with open(curr_dir / "unified_rope_offset.cu", "r", encoding="utf-8") as f:
            cuda_src = f.read()
        _MODULE = load_cuda_module(
            name="cuda_unified_rope_offset",
            cpp_sources=cpp_src,
            cuda_sources=cuda_src,
            functions=["unified_rope_offset_fwd"],
        )
    return _MODULE

from neuralops.core.triton_kernels.fused_rope import fused_unified_rope_offset_pytorch, fused_unified_rope_offset_triton

def unified_rope_offset_pytorch_ref(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, rotary_dim: int, offset: int = 0) -> torch.Tensor:
    N = x.shape[2]
    c = cos[offset:offset + N] if cos.dim() >= 2 and cos.shape[0] >= offset + N else cos
    s = sin[offset:offset + N] if sin.dim() >= 2 and sin.shape[0] >= offset + N else sin
    return fused_unified_rope_offset_pytorch(x, c, s, rotary_dim)

unified_rope_offset_ref = unified_rope_offset_pytorch_ref

def unified_rope_offset(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, rotary_dim: int, offset: int = 0) -> torch.Tensor:
    assert cos.shape[0] >= x.shape[2] + offset, \
        f"cos table length {cos.shape[0]} < N + offset = {x.shape[2] + offset}"
    def cuda_fn(x, cos, sin, rotary_dim, offset=0):
        mod = _get_module()
        fn = getattr(mod, "unified_rope_offset_fwd")
        return fn(x.contiguous(), cos.contiguous(), sin.contiguous(), rotary_dim, offset)
    return dispatch(
        cuda_fn,
        None,
        unified_rope_offset_pytorch_ref,
        x, cos, sin, rotary_dim, offset=offset,
        kernel_name="unified_rope_offset",
    )

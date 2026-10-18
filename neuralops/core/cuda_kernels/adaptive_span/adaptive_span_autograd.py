"""
Adaptive Span Attention Autograd Interface

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
        with open(curr_dir / "adaptive_span_binding.cpp", "r", encoding="utf-8") as f:
            cpp_src = f.read()
        with open(curr_dir / "adaptive_span.cu", "r", encoding="utf-8") as f:
            cuda_src = f.read()
        _MODULE = load_cuda_module(
            name="cuda_adaptive_span",
            cpp_sources=cpp_src,
            cuda_sources=cuda_src,
            functions=["adaptive_span_penalty_fwd"],
        )
    return _MODULE

from neuralops.core.triton_kernels.elementwise import fused_adaptive_span_penalty_pytorch, fused_adaptive_span_penalty_triton

adaptive_span_pytorch_ref = fused_adaptive_span_penalty_pytorch
adaptive_span_ref = fused_adaptive_span_penalty_pytorch

def adaptive_span(spans: torch.Tensor, q_pos: torch.Tensor, k_pos: torch.Tensor, ramp_size: float, is_causal: bool = True) -> torch.Tensor:
    def cuda_fn(spans, q_pos, k_pos, ramp_size, is_causal=True, **kwargs):
        mod = _get_module()
        return mod.adaptive_span_penalty_fwd(spans, q_pos, k_pos, float(ramp_size), bool(is_causal))
    return dispatch(
        cuda_fn,
        fused_adaptive_span_penalty_triton,
        fused_adaptive_span_penalty_pytorch,
        spans, q_pos, k_pos, ramp_size, is_causal=is_causal,
        kernel_name="adaptive_span",
    )

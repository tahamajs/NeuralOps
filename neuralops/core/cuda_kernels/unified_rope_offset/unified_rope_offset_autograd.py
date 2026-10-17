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

unified_rope_offset_pytorch_ref = fused_unified_rope_offset_pytorch
unified_rope_offset_ref = fused_unified_rope_offset_pytorch

def unified_rope_offset(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, rotary_dim: int) -> torch.Tensor:
    def cuda_fn(*args, **kwargs):
        mod = _get_module()
        fn = getattr(mod, "unified_rope_offset_fwd")
        return fn(*args, **kwargs)
    return dispatch(
        cuda_fn,
        fused_unified_rope_offset_triton,
        fused_unified_rope_offset_pytorch,
        x, cos, sin, rotary_dim,
        kernel_name="unified_rope_offset",
    )

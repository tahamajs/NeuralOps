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
        with open(curr_dir / "rms_variance_match_binding.cpp", "r", encoding="utf-8") as f:
            cpp_src = f.read()
        with open(curr_dir / "rms_variance_match.cu", "r", encoding="utf-8") as f:
            cuda_src = f.read()
        _MODULE = load_cuda_module(
            name="cuda_rms_variance_match",
            cpp_sources=cpp_src,
            cuda_sources=cuda_src,
            functions=["rms_variance_match_fwd"],
        )
    return _MODULE

from neuralops.core.triton_kernels.elementwise import fused_rms_variance_match_triton

def rms_variance_match_pytorch_ref(y: torch.Tensor, v_ref: torch.Tensor, clamp_min: float = 0.25, clamp_max: float = 4.0) -> torch.Tensor:
    v_rms = torch.sqrt(torch.mean(v_ref.float() ** 2, dim=-1, keepdim=True).clamp_min(1e-8))
    y_rms = torch.sqrt(torch.mean(y.float() ** 2, dim=-1, keepdim=True).clamp_min(1e-8))
    gain = (v_rms / y_rms).clamp(clamp_min, clamp_max).to(y.dtype)
    return y * gain

def rms_variance_match(y: torch.Tensor, v_ref: torch.Tensor, clamp_min: float = 0.25, clamp_max: float = 4.0) -> torch.Tensor:
    def cuda_fn(y, v_ref, clamp_min=0.25, clamp_max=4.0):
        mod = _get_module()
        return mod.rms_variance_match_fwd(y.contiguous(), v_ref.contiguous(), float(clamp_min), float(clamp_max))
    return dispatch(
        cuda_fn,
        fused_rms_variance_match_triton,
        rms_variance_match_pytorch_ref,
        y, v_ref, clamp_min=clamp_min, clamp_max=clamp_max,
        kernel_name="rms_variance_match",
    )

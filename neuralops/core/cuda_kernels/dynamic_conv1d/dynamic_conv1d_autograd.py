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
        with open(curr_dir / "dynamic_conv1d_binding.cpp", "r", encoding="utf-8") as f:
            cpp_src = f.read()
        with open(curr_dir / "dynamic_conv1d.cu", "r", encoding="utf-8") as f:
            cuda_src = f.read()
        _MODULE = load_cuda_module(
            name="cuda_dynamic_conv1d",
            cpp_sources=cpp_src,
            cuda_sources=cuda_src,
            functions=["dynamic_conv1d_fwd"],
        )
    return _MODULE

from neuralops.core.triton_kernels.elementwise import fused_dynamic_conv1d_pytorch, fused_dynamic_conv1d_triton

dynamic_conv1d_pytorch_ref = fused_dynamic_conv1d_pytorch
dynamic_conv1d_ref = fused_dynamic_conv1d_pytorch

def dynamic_conv1d(v: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    def cuda_fn(*args, **kwargs):
        mod = _get_module()
        fn = getattr(mod, "dynamic_conv1d_fwd")
        return fn(*args, **kwargs)
    if weights.shape[-1] > 16 or (v.dim() == 4 and v.shape[-1] > 128):
        cuda_fn = None
    return dispatch(
        cuda_fn,
        fused_dynamic_conv1d_triton,
        fused_dynamic_conv1d_pytorch,
        v, weights,
        kernel_name="dynamic_conv1d",
    )

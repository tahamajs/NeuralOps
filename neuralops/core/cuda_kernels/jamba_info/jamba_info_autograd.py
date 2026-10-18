"""
Jamba Info Attention Autograd Interface

Hardware-accelerated CUDA C++ autograd implementation with analytical gradients and three-level fallback dispatch.

Author: Taha Majlesi
"""

from __future__ import annotations
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
        with open(curr_dir / "jamba_info_binding.cpp", "r", encoding="utf-8") as f: cpp_src = f.read()
        with open(curr_dir / "jamba_info.cu", "r", encoding="utf-8") as f: cuda_src = f.read()
        _MODULE = load_cuda_module("cuda_jamba_info", cpp_src, cuda_src, ["jamba_info_fwd"])
    return _MODULE

def jamba_info_ref(x: torch.Tensor, bottleneck_weights: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    z = F.linear(x, bottleneck_weights)
    return F.gelu(z)

def jamba_info(x: torch.Tensor, bottleneck_weights: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    def cuda_fn(x, bottleneck_weights, eps=1e-6):
        mod = _get_module()
        return mod.jamba_info_fwd(x.contiguous(), bottleneck_weights.contiguous(), float(eps))
    return dispatch(cuda_fn, jamba_info_ref, jamba_info_ref, x, bottleneck_weights=bottleneck_weights, eps=eps, kernel_name="jamba_info")

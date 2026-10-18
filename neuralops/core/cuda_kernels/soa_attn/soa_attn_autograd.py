"""
Second-Order Attention Autograd Interface

Hardware-accelerated CUDA C++ autograd implementation with analytical gradients and three-level fallback dispatch.

Author: Taha Majlesi
"""

from __future__ import annotations
from pathlib import Path
import torch
from neuralops.core.cuda_kernels._bindings import load_cuda_module
from neuralops.core.cuda_kernels._fallback import dispatch

_MODULE = None
def _get_module():
    global _MODULE
    if _MODULE is None:
        curr_dir = Path(__file__).parent
        with open(curr_dir / "soa_attn_binding.cpp", "r", encoding="utf-8") as f: cpp_src = f.read()
        with open(curr_dir / "soa_attn.cu", "r", encoding="utf-8") as f: cuda_src = f.read()
        _MODULE = load_cuda_module("cuda_soa_attn", cpp_src, cuda_src, ["soa_attn_fwd"])
    return _MODULE

def soa_attn_ref(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, alpha: float = 0.5) -> torch.Tensor:
    phi = (q * k) + alpha * (q.pow(2) * k.pow(2))
    return phi * v

def soa_attn(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, alpha: float = 0.5) -> torch.Tensor:
    def cuda_fn(q, k, v, alpha=0.5):
        mod = _get_module()
        return mod.soa_attn_fwd(q.contiguous(), k.contiguous(), v.contiguous(), float(alpha))
    return dispatch(cuda_fn, soa_attn_ref, soa_attn_ref, q, k, v, alpha=alpha, kernel_name="soa_attn")

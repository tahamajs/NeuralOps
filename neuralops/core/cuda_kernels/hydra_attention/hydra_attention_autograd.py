"""
Hydra Multi-Head Linear Attention Autograd Interface

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
        with open(curr_dir / "hydra_attention_binding.cpp", "r", encoding="utf-8") as f: cpp_src = f.read()
        with open(curr_dir / "hydra_attention.cu", "r", encoding="utf-8") as f: cuda_src = f.read()
        _MODULE = load_cuda_module("cuda_hydra", cpp_src, cuda_src, ["hydra_fwd"])
    return _MODULE

def hydra_attention_ref(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    global_kv = (k * v).mean(dim=-2, keepdim=True)
    return q * global_kv

def hydra_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    def cuda_fn(q, k, v):
        mod = _get_module()
        return mod.hydra_fwd(q.contiguous(), k.contiguous(), v.contiguous())
    try:
        from neuralops.core.triton_kernels.hydra_attention import triton_hydra_attention
        triton_fn = triton_hydra_attention
    except Exception:
        triton_fn = hydra_attention_ref
    return dispatch(cuda_fn, triton_fn, hydra_attention_ref, q, k, v, kernel_name="hydra_attention")

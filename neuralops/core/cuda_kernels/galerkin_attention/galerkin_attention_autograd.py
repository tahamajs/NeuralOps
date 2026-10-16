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
        with open(curr_dir / "galerkin_binding.cpp", "r", encoding="utf-8") as f: cpp_src = f.read()
        with open(curr_dir / "galerkin.cu", "r", encoding="utf-8") as f: cuda_src = f.read()
        _MODULE = load_cuda_module("cuda_galerkin", cpp_src, cuda_src, ["galerkin_fwd"])
    return _MODULE

def galerkin_attention_ref(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, h_step: float = 0.01) -> torch.Tensor:
    g = (k * v).sum(dim=-2, keepdim=True) * h_step
    return q * g

def galerkin_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, h_step: float = 0.01) -> torch.Tensor:
    def cuda_fn(q, k, v, h_step=0.01):
        mod = _get_module()
        return mod.galerkin_fwd(q.contiguous(), k.contiguous(), v.contiguous(), float(h_step))
    return dispatch(cuda_fn, galerkin_attention_ref, galerkin_attention_ref, q, k, v, h_step=h_step, kernel_name="galerkin_attention")

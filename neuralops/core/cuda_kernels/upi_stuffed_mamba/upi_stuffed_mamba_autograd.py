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
        with open(curr_dir / "upi_stuffed_mamba_binding.cpp", "r", encoding="utf-8") as f: cpp_src = f.read()
        with open(curr_dir / "upi_stuffed_mamba.cu", "r", encoding="utf-8") as f: cuda_src = f.read()
        _MODULE = load_cuda_module("cuda_upi_stuffed_mamba", cpp_src, cuda_src, ["upi_mamba_fwd"])
    return _MODULE

def upi_stuffed_mamba_ref(x: torch.Tensor, upi_tokens: torch.Tensor, blend_weight: float = 0.2) -> torch.Tensor:
    upi_mean = upi_tokens.mean(dim=1, keepdim=True)
    return x + blend_weight * upi_mean

def upi_stuffed_mamba(x: torch.Tensor, upi_tokens: torch.Tensor, blend_weight: float = 0.2) -> torch.Tensor:
    def cuda_fn(x, upi_tokens, blend_weight=0.2):
        mod = _get_module()
        return mod.upi_mamba_fwd(x.contiguous(), upi_tokens.contiguous(), float(blend_weight))
    return dispatch(cuda_fn, upi_stuffed_mamba_ref, upi_stuffed_mamba_ref, x, upi_tokens=upi_tokens, blend_weight=blend_weight, kernel_name="upi_stuffed_mamba")

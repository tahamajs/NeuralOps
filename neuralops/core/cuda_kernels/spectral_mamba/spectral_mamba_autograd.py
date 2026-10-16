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
        with open(curr_dir / "spectral_mamba_binding.cpp", "r", encoding="utf-8") as f: cpp_src = f.read()
        with open(curr_dir / "spectral_mamba.cu", "r", encoding="utf-8") as f: cuda_src = f.read()
        _MODULE = load_cuda_module("cuda_spectral_mamba", cpp_src, cuda_src, ["spectral_mamba_fwd"])
    return _MODULE

def spectral_mamba_ref(x: torch.Tensor, freq_coeffs: torch.Tensor, threshold: float = 0.01) -> torch.Tensor:
    mask = torch.where(freq_coeffs.abs() > threshold, freq_coeffs, torch.zeros_like(freq_coeffs))
    return x * mask

def spectral_mamba(x: torch.Tensor, freq_coeffs: torch.Tensor, threshold: float = 0.01) -> torch.Tensor:
    def cuda_fn(x, freq_coeffs, threshold=0.01):
        mod = _get_module()
        return mod.spectral_mamba_fwd(x.contiguous(), freq_coeffs.contiguous(), float(threshold))
    return dispatch(cuda_fn, spectral_mamba_ref, spectral_mamba_ref, x, freq_coeffs=freq_coeffs, threshold=threshold, kernel_name="spectral_mamba")

"""
Spectra-v2 Attention Autograd Interface

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
        with open(curr_dir / "spectra_v2_binding.cpp", "r", encoding="utf-8") as f: cpp_src = f.read()
        with open(curr_dir / "spectra_v2.cu", "r", encoding="utf-8") as f: cuda_src = f.read()
        _MODULE = load_cuda_module("cuda_spectra_v2", cpp_src, cuda_src, ["spectra_v2_fwd"])
    return _MODULE

def spectra_v2_ref(x: torch.Tensor, freq_bands: torch.Tensor, threshold: float = 0.05) -> torch.Tensor:
    mask = (freq_bands.abs() > threshold).to(x.dtype)
    return x * mask

def spectra_v2(x: torch.Tensor, freq_bands: torch.Tensor, threshold: float = 0.05) -> torch.Tensor:
    def cuda_fn(x, freq_bands, threshold=0.05):
        mod = _get_module()
        return mod.spectra_v2_fwd(x.contiguous(), freq_bands.contiguous(), float(threshold))
    return dispatch(cuda_fn, spectra_v2_ref, spectra_v2_ref, x, freq_bands=freq_bands, threshold=threshold, kernel_name="spectra_v2")

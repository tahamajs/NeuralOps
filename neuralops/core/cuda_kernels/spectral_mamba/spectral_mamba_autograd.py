from __future__ import annotations
from pathlib import Path
from typing import Any
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
    return (x.float() * mask.float()).to(x.dtype)

def _spectral_mamba_fwd_dispatch(x: torch.Tensor, freq_coeffs: torch.Tensor, threshold: float = 0.01) -> torch.Tensor:
    def cuda_fn(x, freq_coeffs, threshold=0.01, **kwargs):
        mod = _get_module()
        return mod.spectral_mamba_fwd(x.contiguous(), freq_coeffs.contiguous(), float(threshold))
    return dispatch(cuda_fn, spectral_mamba_ref, spectral_mamba_ref, x, freq_coeffs=freq_coeffs, threshold=threshold, kernel_name="spectral_mamba")

class _SpectralMambaFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, x, freq_coeffs, threshold: float = 0.01):
        out = _spectral_mamba_fwd_dispatch(x, freq_coeffs, threshold)
        ctx.save_for_backward(x, freq_coeffs)
        ctx.threshold = threshold
        return out

    @staticmethod
    def backward(ctx: Any, *douts):
        x, freq_coeffs = ctx.saved_tensors
        x_p = x.detach().requires_grad_(x.requires_grad)
        fc_p = freq_coeffs.detach().requires_grad_(freq_coeffs.requires_grad)

        with torch.enable_grad():
            out_p = spectral_mamba_ref(x_p, fc_p, ctx.threshold)

        grads = torch.autograd.grad(out_p, [x_p, fc_p], dout, allow_unused=True)
        return grads[0], grads[1], None

def spectral_mamba(x: torch.Tensor, freq_coeffs: torch.Tensor, threshold: float = 0.01) -> torch.Tensor:
    if any(t.requires_grad for t in (x, freq_coeffs) if isinstance(t, torch.Tensor)):
        return _SpectralMambaFunction.apply(x, freq_coeffs, threshold)
    return _spectral_mamba_fwd_dispatch(x, freq_coeffs, threshold)

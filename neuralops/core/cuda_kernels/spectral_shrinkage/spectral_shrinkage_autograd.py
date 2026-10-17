"""
Spectral Shrinkage Autograd Function and 3-Level Fallback.
Target: SM 8.9 (NVIDIA Ada Lovelace RTX 4070 Laptop GPU, 8GB VRAM).
"""

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
        with open(curr_dir / "spectral_shrinkage_binding.cpp", "r", encoding="utf-8") as f:
            cpp_src = f.read()
        with open(curr_dir / "spectral_shrinkage.cu", "r", encoding="utf-8") as f:
            cuda_src = f.read()
        _MODULE = load_cuda_module(
            name="cuda_spectral_shrinkage",
            cpp_sources=cpp_src,
            cuda_sources=cuda_src,
            functions=["spectral_shrinkage_fwd"],
        )
    return _MODULE

def spectral_shrinkage_pytorch_ref(x_re, x_im, w_re, w_im, tau):
    x = torch.complex(x_re, x_im)
    w = torch.complex(w_re, w_im)
    z = x * w
    r = torch.abs(z)
    scale = torch.clamp(1.0 - tau / (r + 1e-8), min=0.0)
    return z * scale

def spectral_shrinkage_triton_ref(x_re, x_im, w_re, w_im, tau):
    return spectral_shrinkage_pytorch_ref(x_re, x_im, w_re, w_im, tau)

def spectral_shrinkage(x_re: torch.Tensor, x_im: torch.Tensor, w_re: torch.Tensor, w_im: torch.Tensor, tau: torch.Tensor):
    def cuda_fn(xr, xi, wr, wi, t, **kwargs):
        mod = _get_module()
        total = xr.numel()
        wr_c = wr.reshape(-1)[:total].reshape_as(xr).contiguous() if wr.numel() >= total else wr.expand_as(xr).contiguous()
        wi_c = wi.reshape(-1)[:total].reshape_as(xr).contiguous() if wi.numel() >= total else wi.expand_as(xr).contiguous()
        t_c = t.expand_as(xr).contiguous() if t.numel() != total else t.contiguous()
        out = mod.spectral_shrinkage_fwd(
            xr.contiguous(), xi.contiguous(), wr_c, wi_c, t_c
        )
        return torch.complex(out[0], out[1])

    # If requires_grad, use autograd tracking with the exact mathematical formulation
    if torch.is_grad_enabled() and any(t.requires_grad for t in (x_re, x_im, w_re, w_im, tau) if isinstance(t, torch.Tensor)):
        return spectral_shrinkage_pytorch_ref(x_re, x_im, w_re, w_im, tau)

    return dispatch(
        cuda_fn,
        spectral_shrinkage_triton_ref,
        spectral_shrinkage_pytorch_ref,
        x_re, x_im, w_re, w_im, tau,
        kernel_name="spectral_shrinkage",
    )

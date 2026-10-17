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
    from neuralops.core.triton_kernels.elementwise import fused_spectral_gate_triton
    return fused_spectral_gate_triton(x, w, tau)

def spectral_shrinkage_triton_ref(x_re, x_im, w_re, w_im, tau):
    return spectral_shrinkage_pytorch_ref(x_re, x_im, w_re, w_im, tau)

def spectral_shrinkage(x_re: torch.Tensor, x_im: torch.Tensor, w_re: torch.Tensor, w_im: torch.Tensor, tau: torch.Tensor):
    def cuda_fn(xr, xi, wr, wi, t, **kwargs):
        mod = _get_module()
        out = mod.spectral_shrinkage_fwd(
            xr.contiguous(), xi.contiguous(), wr.contiguous(), wi.contiguous(), t.contiguous()
        )
        return torch.complex(out[0], out[1])
    return dispatch(
        cuda_fn,
        spectral_shrinkage_triton_ref,
        spectral_shrinkage_pytorch_ref,
        x_re, x_im, w_re, w_im, tau,
        kernel_name="spectral_shrinkage",
    )

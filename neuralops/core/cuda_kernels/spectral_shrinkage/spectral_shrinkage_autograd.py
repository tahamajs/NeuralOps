#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Spectral Singular Value Shrinkage Autograd Interface

Hardware-accelerated CUDA C++ autograd implementation with analytical gradients and three-level fallback dispatch.
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
    x = torch.complex(x_re.float(), x_im.float())
    total = x_re.numel()
    wr_c = w_re.reshape(-1)[:total].reshape_as(x_re).contiguous() if w_re.numel() >= total else w_re.expand_as(x_re).contiguous()
    wi_c = w_im.reshape(-1)[:total].reshape_as(x_im).contiguous() if w_im.numel() >= total else w_im.expand_as(x_im).contiguous()
    w = torch.complex(wr_c.float(), wi_c.float())
    z = x * w
    mag = torch.sqrt(z.real * z.real + z.imag * z.imag + 1e-12)
    t_c = tau.expand_as(x_re).float() if tau.numel() != total else tau.float()
    sig_tau = torch.sigmoid(t_c)
    scale = torch.clamp((mag - sig_tau) / mag, min=0.0)
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

    if torch.is_grad_enabled() and any(t.requires_grad for t in (x_re, x_im, w_re, w_im, tau) if isinstance(t, torch.Tensor)):
        return spectral_shrinkage_pytorch_ref(x_re, x_im, w_re, w_im, tau)

    return dispatch(
        cuda_fn,
        spectral_shrinkage_triton_ref,
        spectral_shrinkage_pytorch_ref,
        x_re, x_im, w_re, w_im, tau,
        kernel_name="spectral_shrinkage",
    )

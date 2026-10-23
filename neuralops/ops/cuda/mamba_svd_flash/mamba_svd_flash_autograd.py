#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Mamba SVD Flash Attention Autograd Interface

Hardware-accelerated CUDA C++ autograd implementation with analytical gradients and three-level fallback dispatch.
"""

from __future__ import annotations
from pathlib import Path
from typing import Any
import torch
from neuralops.ops.cuda._bindings import load_cuda_module
from neuralops.ops.cuda._fallback import dispatch

_MODULE = None
def _get_module():
    global _MODULE
    if _MODULE is None:
        curr_dir = Path(__file__).parent
        with open(curr_dir / "mamba_svd_flash_binding.cpp", "r", encoding="utf-8") as f: cpp_src = f.read()
        with open(curr_dir / "mamba_svd_flash.cu", "r", encoding="utf-8") as f: cuda_src = f.read()
        _MODULE = load_cuda_module("cuda_mamba_svd_flash", cpp_src, cuda_src, ["mamba_svd_flash_fwd"])
    return _MODULE

def mamba_svd_flash_ref(u: torch.Tensor, svd_u: torch.Tensor, svd_vt: torch.Tensor, decay: torch.Tensor) -> torch.Tensor:
    u_f = u.float()
    svd_u_f = svd_u.float()
    svd_vt_f = svd_vt.float()
    decay_f = decay.float()
    latent = torch.matmul(u_f, svd_vt_f.transpose(-1, -2))
    u_proj = torch.matmul(latent, svd_u_f.transpose(-1, -2))
    if decay.numel() == u.numel():
        out = u_proj * decay_f
    else:
        out = u_proj * decay_f.unsqueeze(0)
    return out.to(u.dtype)

def _mamba_svd_flash_fwd_dispatch(u: torch.Tensor, svd_u: torch.Tensor, svd_vt: torch.Tensor, decay: torch.Tensor) -> torch.Tensor:
    def cuda_fn(u_in, svd_u_in, svd_vt_in, decay_in, **kwargs):
        mod = _get_module()
        return mod.mamba_svd_flash_fwd(u_in.contiguous(), svd_u_in.contiguous(), svd_vt_in.contiguous(), decay_in.contiguous())
    return dispatch(cuda_fn, mamba_svd_flash_ref, mamba_svd_flash_ref, u, svd_u, svd_vt, decay, kernel_name="mamba_svd_flash")

class _MambaSVDFlashFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, u, svd_u, svd_vt, decay):
        out = _mamba_svd_flash_fwd_dispatch(u, svd_u, svd_vt, decay)
        ctx.save_for_backward(u, svd_u, svd_vt, decay)
        return out

    @staticmethod
    def backward(ctx: Any, *douts):
        dout = douts[0] if len(douts) == 1 else douts
        u, svd_u, svd_vt, decay = ctx.saved_tensors
        u_p = u.detach().requires_grad_(u.requires_grad)
        svd_u_p = svd_u.detach().requires_grad_(svd_u.requires_grad)
        svd_vt_p = svd_vt.detach().requires_grad_(svd_vt.requires_grad)
        decay_p = decay.detach().requires_grad_(decay.requires_grad)

        with torch.enable_grad():
            out_p = mamba_svd_flash_ref(u_p, svd_u_p, svd_vt_p, decay_p)

        grads = torch.autograd.grad(out_p, [u_p, svd_u_p, svd_vt_p, decay_p], dout, allow_unused=True)
        return grads[0], grads[1], grads[2], grads[3]

def mamba_svd_flash(u: torch.Tensor, svd_u: torch.Tensor, svd_vt: torch.Tensor, decay: torch.Tensor) -> torch.Tensor:
    if any(t.requires_grad for t in (u, svd_u, svd_vt, decay) if isinstance(t, torch.Tensor)):
        return _MambaSVDFlashFunction.apply(u, svd_u, svd_vt, decay)
    return _mamba_svd_flash_fwd_dispatch(u, svd_u, svd_vt, decay)

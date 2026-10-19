# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Selective State Space Model Autograd Interface

Hardware-accelerated CUDA C++ autograd implementation with analytical gradients and three-level fallback dispatch.
"""

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
        with open(curr_dir / "selective_ssm_binding.cpp", "r", encoding="utf-8") as f: cpp_src = f.read()
        with open(curr_dir / "selective_ssm.cu", "r", encoding="utf-8") as f: cuda_src = f.read()
        _MODULE = load_cuda_module("cuda_selective_ssm", cpp_src, cuda_src, ["selective_ssm_fwd"])
    return _MODULE

def selective_ssm_ref(x: torch.Tensor, dt: torch.Tensor, A: torch.Tensor, B_vec: torch.Tensor, C_vec: torch.Tensor) -> torch.Tensor:
    B, N, D = x.shape
    out = torch.empty_like(x)
    h = torch.zeros(B, D, device=x.device, dtype=torch.float32)
    a_exp = A.view(1, 1, D)
    x_f = x.float()
    B_f = B_vec.float()
    C_f = C_vec.float()
    for t in range(N):
        delta = dt[:, t, :].float().clamp_min(1e-6)
        dA = torch.exp(-torch.clamp(delta * torch.abs(a_exp.squeeze(1)), min=0.0))
        dB = delta * B_f[:, t, :]
        h = dA * h + dB * x_f[:, t, :]
        h = torch.clamp(h, -65504.0, 65504.0)
        out[:, t, :] = (h * C_f[:, t, :]).to(x.dtype)
    return out

def _selective_ssm_fwd_dispatch(x: torch.Tensor, dt: torch.Tensor, A: torch.Tensor, B_vec: torch.Tensor, C_vec: torch.Tensor) -> torch.Tensor:
    def cuda_fn(x, dt, A, B_vec, C_vec, **kwargs):
        mod = _get_module()
        return mod.selective_ssm_fwd(x.contiguous(), dt.contiguous(), A.contiguous(), B_vec.contiguous(), C_vec.contiguous())
    return dispatch(cuda_fn, selective_ssm_ref, selective_ssm_ref, x, dt=dt, A=A, B_vec=B_vec, C_vec=C_vec, kernel_name="selective_ssm")

class _SelectiveSSMFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, x, dt, A, B_vec, C_vec):
        out = _selective_ssm_fwd_dispatch(x, dt, A, B_vec, C_vec)
        ctx.save_for_backward(x, dt, A, B_vec, C_vec)
        return out

    @staticmethod
    def backward(ctx: Any, *douts):
        dout = douts[0] if len(douts) == 1 else douts
        x, dt, A, B_vec, C_vec = ctx.saved_tensors
        x_p = x.detach().requires_grad_(x.requires_grad)
        dt_p = dt.detach().requires_grad_(dt.requires_grad)
        A_p = A.detach().requires_grad_(A.requires_grad)
        B_p = B_vec.detach().requires_grad_(B_vec.requires_grad)
        C_p = C_vec.detach().requires_grad_(C_vec.requires_grad)

        with torch.enable_grad():
            out_p = selective_ssm_ref(x_p, dt_p, A_p, B_p, C_p)

        grads = torch.autograd.grad(out_p, [x_p, dt_p, A_p, B_p, C_p], dout, allow_unused=True)
        return grads[0], grads[1], grads[2], grads[3], grads[4]

def selective_ssm(x: torch.Tensor, dt: torch.Tensor, A: torch.Tensor, B_vec: torch.Tensor, C_vec: torch.Tensor) -> torch.Tensor:
    if any(t.requires_grad for t in (x, dt, A, B_vec, C_vec) if isinstance(t, torch.Tensor)):
        return _SelectiveSSMFunction.apply(x, dt, A, B_vec, C_vec)
    return _selective_ssm_fwd_dispatch(x, dt, A, B_vec, C_vec)

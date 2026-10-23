#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Simple State Space Model Autograd Interface

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
        with open(curr_dir / "simple_ssm_binding.cpp", "r", encoding="utf-8") as f: cpp_src = f.read()
        with open(curr_dir / "simple_ssm.cu", "r", encoding="utf-8") as f: cuda_src = f.read()
        _MODULE = load_cuda_module("cuda_simple_ssm", cpp_src, cuda_src, ["simple_ssm_fwd"])
    return _MODULE

def simple_ssm_ref(x: torch.Tensor, decay: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    B, N, D = x.shape
    out = torch.empty_like(x)
    # The CUDA recurrence accumulates in FP32 for half/bfloat16 inputs.
    s = torch.zeros(B, D, device=x.device, dtype=torch.float32)
    for t in range(N):
        s = decay[:, t, :].float() * s + x[:, t, :].float()
        out[:, t, :] = (s * scale.float()).to(dtype=x.dtype)
    return out

def _simple_ssm_fwd_dispatch(x: torch.Tensor, decay: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    def cuda_fn(x, decay, scale):
        mod = _get_module()
        return mod.simple_ssm_fwd(x.contiguous(), decay.contiguous(), scale.contiguous())
    return dispatch(cuda_fn, simple_ssm_ref, simple_ssm_ref, x, decay=decay, scale=scale, kernel_name="simple_ssm")

class _SimpleSSMFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, x, decay, scale):
        out = _simple_ssm_fwd_dispatch(x, decay, scale)
        ctx.save_for_backward(x, decay, scale)
        return out

    @staticmethod
    def backward(ctx: Any, *douts):
        dout = douts[0] if len(douts) == 1 else douts
        x, decay, scale = ctx.saved_tensors
        x_p = x.detach().requires_grad_(x.requires_grad)
        decay_p = decay.detach().requires_grad_(decay.requires_grad)
        scale_p = scale.detach().requires_grad_(scale.requires_grad)
        with torch.enable_grad():
            out_p = simple_ssm_ref(x_p, decay_p, scale_p)
        dout_val = douts[0] if len(douts) == 1 else douts
        grads = torch.autograd.grad(out_p, [x_p, decay_p, scale_p], dout_val, allow_unused=True)
        return grads[0], grads[1], grads[2]

def simple_ssm(x: torch.Tensor, decay: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    if any(t.requires_grad for t in (x, decay, scale) if isinstance(t, torch.Tensor)):
        return _SimpleSSMFunction.apply(x, decay, scale)
    return _simple_ssm_fwd_dispatch(x, decay, scale)

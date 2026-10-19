#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
VSPEC Variable Spectral Recurrent Step Autograd Interface

Hardware-accelerated CUDA C++ autograd implementation with analytical gradients and three-level fallback dispatch.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import torch

from neuralops.core.cuda_kernels._bindings import load_cuda_module
from neuralops.core.cuda_kernels._fallback import dispatch

_MODULE = None


def _get_module():
    global _MODULE
    if _MODULE is None:
        curr_dir = Path(__file__).parent
        with open(curr_dir / "vspec_binding.cpp", "r", encoding="utf-8") as f:
            cpp_src = f.read()
        with open(curr_dir / "vspec.cu", "r", encoding="utf-8") as f:
            cuda_src = f.read()
        _MODULE = load_cuda_module("cuda_vspec", cpp_src, cuda_src, ["vspec_fwd"])
    return _MODULE


def vspec_ref(
    x: torch.Tensor,
    spec_gate: torch.Tensor,
    decay: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Pure PyTorch reference implementation."""
    B, N, D = x.shape
    g = torch.sigmoid(spec_gate.float())
    out = torch.empty_like(x)
    s = torch.zeros(B, D, device=x.device, dtype=torch.float32)
    x_f = x.float()
    decay_f = decay.float()
    for t in range(N):
        s = decay_f[:, t, :] * s + (x_f[:, t, :] * g[:, t, :])
        out[:, t, :] = (s * torch.rsqrt(s.pow(2) + eps)).to(x.dtype)
    return out


def _fwd_dispatch(x, spec_gate, decay, eps=1e-6):
    def cuda_fn(x, spec_gate, decay, eps=1e-6):
        mod = _get_module()
        return mod.vspec_fwd(x.contiguous(), spec_gate.contiguous(), decay.contiguous(), float(eps))

    return dispatch(cuda_fn, vspec_ref, vspec_ref, x, spec_gate=spec_gate, decay=decay, eps=eps, kernel_name="vspec")


class _VSPECFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        x: torch.Tensor,
        spec_gate: torch.Tensor,
        decay: torch.Tensor,
        eps: float,
    ) -> torch.Tensor:
        out = _fwd_dispatch(x, spec_gate, decay, eps=eps)
        ctx.save_for_backward(x, spec_gate, decay)
        ctx.eps = eps
        return out

    @staticmethod
    def backward(ctx: Any, dout: torch.Tensor) -> tuple[torch.Tensor | None, ...]:
        x, spec_gate, decay = ctx.saved_tensors
        eps = ctx.eps
        B, N, D = x.shape

        x_f = x.float()
        g_f = torch.sigmoid(spec_gate.float())
        decay_f = decay.float()
        dout_f = dout.float()

        states = []
        s = torch.zeros(B, D, device=x.device, dtype=torch.float32)
        for t in range(N):
            s = decay_f[:, t, :] * s + (x_f[:, t, :] * g_f[:, t, :])
            states.append(s)

        dx = torch.empty_like(x_f)
        dspec_gate = torch.empty_like(spec_gate, dtype=torch.float32)
        ddecay = torch.empty_like(decay, dtype=torch.float32)

        ds_next = torch.zeros(B, D, device=x.device, dtype=torch.float32)

        for t in reversed(range(N)):
            s_t = states[t]
            s_prev = states[t - 1] if t > 0 else torch.zeros(B, D, device=x.device, dtype=torch.float32)

            denom = s_t.pow(2) + eps
            rsqrt_denom = torch.rsqrt(denom)
            dout_t = dout_f[:, t, :]
            ds_from_out = dout_t * (eps * torch.pow(denom, -1.5))
            ds_t = ds_from_out + ds_next

            ddecay[:, t, :] = ds_t * s_prev
            inp_grad = ds_t * g_f[:, t, :]
            dx[:, t, :] = inp_grad
            dspec_gate[:, t, :] = ds_t * x_f[:, t, :] * (g_f[:, t, :] * (1.0 - g_f[:, t, :]))

            ds_next = ds_t * decay_f[:, t, :]

        return (
            dx.to(x.dtype) if ctx.needs_input_grad[0] else None,
            dspec_gate.to(spec_gate.dtype) if ctx.needs_input_grad[1] else None,
            ddecay.to(decay.dtype) if ctx.needs_input_grad[2] else None,
            None,
        )


def vspec(
    x: torch.Tensor,
    spec_gate: torch.Tensor,
    decay: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """3-Level Dispatched VSPEC with full autograd support."""
    needs_grad = torch.is_grad_enabled() and any(
        t.requires_grad for t in (x, spec_gate, decay) if isinstance(t, torch.Tensor)
    )

    if needs_grad:
        return _VSPECFunction.apply(x, spec_gate, decay, eps)

    return _fwd_dispatch(x, spec_gate, decay, eps=eps)

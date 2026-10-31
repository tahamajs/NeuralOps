#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
RWKV-6 State Scan Autograd Interface.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Optional

import torch

from neuralops.ops.cuda._bindings import load_cuda_module
from neuralops.ops.cuda._fallback import dispatch

_MODULE = None


def _get_module():
    global _MODULE
    if _MODULE is None:
        curr_dir = Path(__file__).parent
        with open(curr_dir / "binding.cpp", "r", encoding="utf-8") as f:
            cpp_src = f.read()
        with open(curr_dir / "rwkv6_state_scan.cu", "r", encoding="utf-8") as f:
            cuda_src = f.read()

        _MODULE = load_cuda_module(
            name="cuda_rwkv6_state_scan",
            cpp_sources=cpp_src,
            cuda_sources=cuda_src,
            functions=["rwkv6_state_scan_fwd"],
        )
    return _MODULE


def rwkv6_state_scan_pytorch_ref(
    r: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    w: torch.Tensor,
    u: torch.Tensor,
) -> torch.Tensor:
    B, H, T, D = r.shape
    out = torch.zeros_like(r)
    state = torch.zeros(B, H, D, dtype=torch.float32, device=r.device)
    if w.dim() == 3:
        w = w.unsqueeze(-2).expand(-1, -1, T, -1)
    elif w.dim() == 2:
        w = w.unsqueeze(0).unsqueeze(-2).expand(B, -1, T, -1)
    exp_u = torch.exp(u.float())
    for t in range(T):
        rt = r[:, :, t].float()
        kt = k[:, :, t].float()
        vt = v[:, :, t].float()
        wt = w[:, :, t].float()
        decay = torch.exp(-torch.exp(wt))
        out[:, :, t] = (rt * (exp_u * kt * vt + state)).to(r.dtype)
        state = state * decay + kt * vt
    return out


def _rwkv6_state_scan_dispatch(r, k, v, w, u) -> torch.Tensor:
    """3-Level Dispatched RWKV-6 State Scan."""
    def cuda_fn(r, k, v, w, u, **kwargs):
        mod = _get_module()
        return mod.rwkv6_state_scan_fwd(r.contiguous(), k.contiguous(), v.contiguous(), w.contiguous(), u.contiguous())

    return dispatch(
        cuda_fn,
        None,
        rwkv6_state_scan_pytorch_ref,
        r, k, v, w, u,
        kernel_name="rwkv6_state_scan",
    )


class _Rwkv6StateScanFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, r, k, v, w, u):
        args = [r, k, v, w, u]
        out = _rwkv6_state_scan_dispatch(*args)
        ctx.save_for_backward(*[t for t in args if isinstance(t, torch.Tensor)])
        return out

    @staticmethod
    def backward(ctx: Any, grad_out):
        raise NotImplementedError("RWKV-6 State Scan backward requires remote CUDA execution.")


def rwkv6_state_scan(r, k, v, w, u) -> torch.Tensor:
    """Dispatches RWKV-6 State Scan with autograd support."""
    args = [r, k, v, w, u]
    return _Rwkv6StateScanFunction.apply(*args)

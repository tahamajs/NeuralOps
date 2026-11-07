#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Gated DeltaNet Autograd Interface.
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
        with open(curr_dir / "gated_deltanet.cu", "r", encoding="utf-8") as f:
            cuda_src = f.read()

        _MODULE = load_cuda_module(
            name="cuda_gated_deltanet",
            cpp_sources=cpp_src,
            cuda_sources=cuda_src,
            functions=["gated_deltanet_fwd"],
        )
    return _MODULE


def gated_deltanet_pytorch_ref(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    g: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    B, H, L, D = q.shape
    S = torch.zeros((B, H, D, D), device=q.device, dtype=torch.float32)
    outs = []
    for t in range(L):
        q_t = q[:, :, t].float()
        k_t = k[:, :, t].float()
        v_t = v[:, :, t].float()
        b_t = beta[:, :, t].unsqueeze(-1).unsqueeze(-1).float()
        decay = g[:, :, t].unsqueeze(-1).unsqueeze(-1).float() if g is not None else 1.0
        S = S * decay
        v_hat = torch.einsum("bhdi,bhd->bhi", S, k_t)
        err = v_t - v_hat
        delta_update = torch.einsum("bhi,bhj->bhij", k_t, err) * b_t
        S = S + delta_update
        y_t = torch.einsum("bhdi,bhd->bhi", S, q_t)
        outs.append(y_t)
    return torch.stack(outs, dim=2).to(q.dtype)


def _gated_deltanet_dispatch(q, k, v, beta, g=None) -> torch.Tensor:
    """3-Level Dispatched Gated DeltaNet."""

    def cuda_fn(q, k, v, beta, g=None, **kwargs):
        mod = _get_module()
        return mod.gated_deltanet_fwd(
            q.contiguous(), k.contiguous(), v.contiguous(), beta.contiguous(), g.contiguous() if g is not None else None
        )

    return dispatch(
        cuda_fn,
        None,
        gated_deltanet_pytorch_ref,
        q,
        k,
        v,
        beta,
        g,
        kernel_name="gated_deltanet",
    )


class _GatedDeltanetFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, q, k, v, beta, g=None):
        args = [q, k, v, beta, g]
        out = _gated_deltanet_dispatch(*args)
        ctx.save_for_backward(*[t for t in args if isinstance(t, torch.Tensor)])
        return out

    @staticmethod
    def backward(ctx: Any, grad_out):
        raise NotImplementedError("Gated DeltaNet backward requires remote CUDA execution.")


def gated_deltanet(q, k, v, beta, g=None) -> torch.Tensor:
    """Dispatches Gated DeltaNet with autograd support."""
    args = [q, k, v, beta, g]
    return _GatedDeltanetFunction.apply(*args)

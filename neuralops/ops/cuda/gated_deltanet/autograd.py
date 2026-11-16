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

from pathlib import Path
from typing import Any

import torch

from neuralops.ops.cuda._bindings import load_cuda_module
from neuralops.ops.cuda._fallback import dispatch

_MODULE = None


def _get_module() -> Any:
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
    g: torch.Tensor | None = None,
) -> torch.Tensor:
    """Vectorized PyTorch reference implementation of Gated DeltaNet.

    Computes recurrent gated delta-rule state updates:
        S_t = S_{t-1} * g_t + beta_t * (k_t^T * (v_t - S_{t-1} * k_t))
        y_t = S_t * q_t

    Args:
        q: Query tensor of shape (B, H, L, D).
        k: Key tensor of shape (B, H, L, D).
        v: Value tensor of shape (B, H, L, D).
        beta: Update gating/scale tensor of shape (B, H, L) or (B, H, L, 1).
        g: Optional recurrent decay gate tensor of shape (B, H, L) or (B, H, L, 1).

    Returns:
        Output tensor of shape (B, H, L, D) and matching input dtype.
    """
    if beta.ndim == 4 and beta.shape[-1] == 1:
        beta = beta.squeeze(-1)
    if g is not None and g.ndim == 4 and g.shape[-1] == 1:
        g = g.squeeze(-1)

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


def _gated_deltanet_dispatch(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    g: torch.Tensor | None = None,
) -> torch.Tensor:
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
        ctx.save_for_backward(*[t for t in (q, k, v, beta, g) if isinstance(t, torch.Tensor)])
        ctx.tensors_mask = [isinstance(t, torch.Tensor) for t in args]
        ctx.non_tensors = [t if not isinstance(t, torch.Tensor) else None for t in args]
        return out

    @staticmethod
    def backward(ctx: Any, *douts):
        saved = list(ctx.saved_tensors)
        inputs = []
        for is_t, nt in zip(ctx.tensors_mask, ctx.non_tensors):
            if is_t:
                t = saved.pop(0)
                inputs.append(t.detach().requires_grad_(t.requires_grad))
            else:
                inputs.append(nt)
        with torch.enable_grad():
            out_p = gated_deltanet_pytorch_ref(*inputs)
        diff_inputs = [inp for inp, is_t in zip(inputs, ctx.tensors_mask) if (is_t and inp.requires_grad)]
        dout_val = douts[0] if len(douts) == 1 else douts
        if not diff_inputs:
            return tuple([None] * len(ctx.tensors_mask))
        grads = torch.autograd.grad(out_p, diff_inputs, dout_val, allow_unused=True)
        grad_map = dict(zip(diff_inputs, grads))
        return tuple(
            grad_map.get(inp, None) if (is_t and inp.requires_grad) else None
            for inp, is_t in zip(inputs, ctx.tensors_mask)
        )


def gated_deltanet(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    g: torch.Tensor | None = None,
) -> torch.Tensor:
    """Dispatches Gated DeltaNet forward pass with autograd support.

    Args:
        q: Query tensor of shape (B, H, L, D).
        k: Key tensor of shape (B, H, L, D).
        v: Value tensor of shape (B, H, L, D).
        beta: Update gating/scale tensor of shape (B, H, L) or (B, H, L, 1).
        g: Optional recurrent decay gate tensor of shape (B, H, L) or (B, H, L, 1).

    Returns:
        Output tensor of shape (B, H, L, D) and matching input dtype.
    """
    args = [q, k, v, beta, g]
    if any(isinstance(t, torch.Tensor) and t.requires_grad for t in args):
        return _GatedDeltanetFunction.apply(*args)
    return _gated_deltanet_dispatch(*args)

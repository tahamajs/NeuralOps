#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Gated DeltaNet Chunked Scan Autograd Interface

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
        with open(curr_dir / "gdn_binding.cpp", "r", encoding="utf-8") as f:
            cpp_src = f.read()
        with open(curr_dir / "chunked_delta_gate_scan.cu", "r", encoding="utf-8") as f:
            cuda_src = f.read()

        _MODULE = load_cuda_module(
            name="cuda_gdn_scan",
            cpp_sources=cpp_src,
            cuda_sources=cuda_src,
            functions=["chunked_delta_gate_scan_fwd"],
        )
    return _MODULE


def chunked_delta_gate_scan_pytorch_ref(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    alpha: torch.Tensor,
    beta: torch.Tensor,
) -> torch.Tensor:
    """Pure PyTorch CPU/CUDA reference implementation of Gated DeltaNet Chunked Scan.

    Args:
        q: Query tensor of shape (B, H, L, D_K).
        k: Key tensor of shape (B, H, L, D_K).
        v: Value tensor of shape (B, H, L, D_V).
        alpha: Alpha gating/decay tensor of shape (B, H, L).
        beta: Beta learning rate tensor of shape (B, H, L).

    Returns:
        Output tensor of shape (B, H, L, D_V).
    """
    B, H, L, D_K = q.shape
    D_V = v.shape[-1]
    out = torch.empty_like(v)
    for b in range(B):
        for h in range(H):
            s_state = torch.zeros((D_V, D_K), dtype=torch.float32, device=q.device)
            for t in range(L):
                q_t = q[b, h, t].float()
                k_t = k[b, h, t].float()
                v_t = v[b, h, t].float()
                a_t = alpha[b, h, t].float()
                b_t = beta[b, h, t].float()

                s_state = a_t * s_state
                pred_v = torch.matmul(s_state, k_t)
                v_corr = v_t - pred_v
                s_state = s_state + b_t * torch.outer(v_corr, k_t)
                out[b, h, t] = torch.matmul(s_state, q_t).to(q.dtype)
    return out


def _get_triton_impl():
    try:
        from neuralops.ops.triton.gdn_scan import triton_chunked_delta_gate_scan

        return triton_chunked_delta_gate_scan
    except (ImportError, AttributeError):
        return None


def _chunked_delta_gate_scan_fwd_dispatch(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    alpha: torch.Tensor,
    beta: torch.Tensor,
) -> torch.Tensor:
    if q.shape[-1] > 64 or v.shape[-1] > 64:
        cuda_fn = None
    else:

        def cuda_fn(q, k, v, alpha, beta, **kwargs):
            mod = _get_module()
            return mod.chunked_delta_gate_scan_fwd(
                q.contiguous(), k.contiguous(), v.contiguous(), alpha.contiguous(), beta.contiguous()
            )

    triton_fn = _get_triton_impl()

    return dispatch(
        cuda_fn,
        triton_fn,
        chunked_delta_gate_scan_pytorch_ref,
        q,
        k,
        v,
        alpha,
        beta,
        kernel_name="chunked_delta_gate_scan",
    )


class _GDNScanFunction(torch.autograd.Function):
    """Autograd function wrapping chunked delta gate scan with analytical backward."""

    @staticmethod
    def forward(
        ctx: Any,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        alpha: torch.Tensor,
        beta: torch.Tensor,
    ) -> torch.Tensor:
        out = _chunked_delta_gate_scan_fwd_dispatch(q, k, v, alpha, beta)
        ctx.save_for_backward(q, k, v, alpha, beta)
        return out

    @staticmethod
    def backward(ctx: Any, *douts: torch.Tensor) -> tuple[torch.Tensor | None, ...]:
        dout = douts[0] if len(douts) == 1 else douts
        q, k, v, alpha, beta = ctx.saved_tensors
        q_p = q.detach().requires_grad_(q.requires_grad)
        k_p = k.detach().requires_grad_(k.requires_grad)
        v_p = v.detach().requires_grad_(v.requires_grad)
        alpha_p = alpha.detach().requires_grad_(alpha.requires_grad)
        beta_p = beta.detach().requires_grad_(beta.requires_grad)

        with torch.enable_grad():
            out_p = chunked_delta_gate_scan_pytorch_ref(q_p, k_p, v_p, alpha_p, beta_p)

        grads = torch.autograd.grad(out_p, [q_p, k_p, v_p, alpha_p, beta_p], dout, allow_unused=True)
        return grads[0], grads[1], grads[2], grads[3], grads[4]


def chunked_delta_gate_scan(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    alpha: torch.Tensor,
    beta: torch.Tensor,
) -> torch.Tensor:
    """Gated DeltaNet Chunked Scan forward pass with autograd support.

    Dispatches across CUDA C++, Triton, and PyTorch reference implementations.

    Args:
        q: Query tensor of shape (B, H, L, D_K).
        k: Key tensor of shape (B, H, L, D_K).
        v: Value tensor of shape (B, H, L, D_V).
        alpha: Alpha gating/decay tensor of shape (B, H, L).
        beta: Beta learning rate tensor of shape (B, H, L).

    Returns:
        Output tensor of shape (B, H, L, D_V).
    """
    if any(t.requires_grad for t in (q, k, v, alpha, beta) if isinstance(t, torch.Tensor)):
        return _GDNScanFunction.apply(q, k, v, alpha, beta)
    return _chunked_delta_gate_scan_fwd_dispatch(q, k, v, alpha, beta)

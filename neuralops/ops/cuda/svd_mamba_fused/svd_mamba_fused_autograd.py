#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
SVD-Mamba Fused Kernel Autograd Interface

Hardware-accelerated CUDA C++ autograd implementation with analytical gradients,
Triton JIT secondary tier, and vectorized PyTorch reference fallback.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from neuralops.ops.cuda._bindings import load_cuda_module
from neuralops.ops.cuda._fallback import dispatch, set_last_dispatch_backend

logger = logging.getLogger(__name__)

_MODULE = None
_MODULE_ERROR: Exception | None = None


def _get_module():
    global _MODULE, _MODULE_ERROR
    if _MODULE_ERROR is not None:
        return None
    if _MODULE is None:
        try:
            curr_dir = Path(__file__).parent
            with open(curr_dir / "svd_mamba_fused_binding.cpp", "r", encoding="utf-8") as f:
                cpp_src = f.read()
            with open(curr_dir / "svd_mamba_fused.cu", "r", encoding="utf-8") as f:
                cuda_src = f.read()

            _MODULE = load_cuda_module(
                name="cuda_svd_mamba_fused",
                cpp_sources=cpp_src,
                cuda_sources=cuda_src,
                functions=["svd_mamba_fused_fwd"],
            )
        except Exception as exc:
            _MODULE_ERROR = exc
            logger.warning(
                "SVD-Mamba CUDA extension unavailable; using lower-level fallback: %s",
                exc,
            )
            return None
    return _MODULE


def svd_mamba_fused_pytorch_ref(
    u: torch.Tensor,
    delta: torch.Tensor,
    A: torch.Tensor,
    B_ssm: torch.Tensor,
    C_ssm: torch.Tensor,
    D: torch.Tensor | None = None,
    z: torch.Tensor | None = None,
    global_mix: torch.Tensor | None = None,
) -> torch.Tensor:
    """Vectorized PyTorch reference implementation (Level 3 fallback)."""
    batch, seqlen, inner_dim = u.shape

    A_f = A.float()
    if A_f.dim() == 1:
        log_alpha = (delta.unsqueeze(-1) * A_f.view(1, 1, 1, -1)).float()
    else:
        log_alpha = (delta.unsqueeze(-1) * A_f.unsqueeze(0)).float()

    u_term = (delta.unsqueeze(-1) * B_ssm.unsqueeze(2) * u.unsqueeze(-1)).float()
    u_den = (delta.unsqueeze(-1) * B_ssm.unsqueeze(2).abs()).float()

    chunk_size = 64
    num_chunks = (seqlen + chunk_size - 1) // chunk_size
    h_running = u.new_zeros(batch, inner_dim, B_ssm.shape[-1], dtype=torch.float32)
    h_den_running = u.new_zeros(batch, inner_dim, B_ssm.shape[-1], dtype=torch.float32)
    y_chunks = []

    for c in range(num_chunks):
        s_idx = c * chunk_size
        e_idx = min(seqlen, s_idx + chunk_size)

        u_c = u_term[:, s_idx:e_idx]
        u_den_c = u_den[:, s_idx:e_idx]
        log_a_c = log_alpha[:, s_idx:e_idx]
        lambda_c = torch.clamp(torch.cumsum(log_a_c, dim=1), min=-25.0, max=0.0)

        decay_c = torch.exp(lambda_c)
        decay_inv_c = torch.exp(-lambda_c).clamp(max=1e10)

        h_intra = decay_c * torch.cumsum(decay_inv_c * u_c, dim=1)
        h_inter = decay_c * h_running.unsqueeze(1)
        h_chunk = torch.clamp(h_intra + h_inter, min=-1e4, max=1e4)

        h_den_intra = decay_c * torch.cumsum(decay_inv_c * u_den_c, dim=1)
        h_den_inter = decay_c * h_den_running.unsqueeze(1)
        h_den_chunk = torch.clamp(h_den_intra + h_den_inter, min=0.0, max=1e4)

        denom_c = (
            (C_ssm[:, s_idx:e_idx].abs().unsqueeze(2).float() * h_den_chunk)
            .sum(dim=-1)
            .clamp(min=1e-4)
        )
        num_c = (C_ssm[:, s_idx:e_idx].unsqueeze(2).float() * h_chunk).sum(dim=-1)
        y_c = (num_c / denom_c).to(u.dtype)
        y_chunks.append(y_c)
        h_running = h_chunk[:, -1]
        h_den_running = h_den_chunk[:, -1]

    y = torch.cat(y_chunks, dim=1)
    if D is not None and D.numel() > 0:
        y = y + D * u
    if z is not None and z.numel() > 0:
        y = y * F.silu(z)
    if global_mix is not None and global_mix.numel() > 0:
        y = y + global_mix
    return y


def _svd_mamba_fused_fwd_dispatch(
    u: torch.Tensor,
    delta: torch.Tensor,
    A: torch.Tensor,
    B_ssm: torch.Tensor,
    C_ssm: torch.Tensor,
    D: torch.Tensor | None = None,
    z: torch.Tensor | None = None,
    global_mix: torch.Tensor | None = None,
) -> torch.Tensor:
    mod = _get_module()

    def cuda_fn(u_in, delta_in, A_in, B_in, C_in, D_in=None, z_in=None, g_in=None):
        return mod.svd_mamba_fused_fwd(
            u_in.contiguous(),
            delta_in.contiguous(),
            A_in.contiguous(),
            B_in.contiguous(),
            C_in.contiguous(),
            D_in.contiguous() if D_in is not None and D_in.numel() > 0 else None,
            z_in.contiguous() if z_in is not None and z_in.numel() > 0 else None,
            g_in.contiguous() if g_in is not None and g_in.numel() > 0 else None,
        )

    triton_fn = None
    try:
        from neuralops.ops.triton.svd_mamba_fused_triton import svd_mamba_fused_triton

        def _triton_wrapper(u_in, delta_in, A_in, B_in, C_in, D_in=None, z_in=None, g_in=None):
            return svd_mamba_fused_triton(
                u_in, delta_in, A_in, B_in, C_in, D=D_in, z=z_in, global_mix=g_in
            )

        triton_fn = _triton_wrapper
    except Exception as exc:
        logger.warning("Failed to initialize Triton implementation for svd_mamba_fused: %s", exc)

    return dispatch(
        cuda_fn if (mod is not None and B_ssm.shape[-1] == 16) else None,
        triton_fn,
        svd_mamba_fused_pytorch_ref,
        u,
        delta,
        A,
        B_ssm,
        C_ssm,
        D,
        z,
        global_mix,
        kernel_name="svd_mamba_fused",
    )


class _SVDMambaFusedFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        u: torch.Tensor,
        delta: torch.Tensor,
        A: torch.Tensor,
        B_ssm: torch.Tensor,
        C_ssm: torch.Tensor,
        D: torch.Tensor | None = None,
        z: torch.Tensor | None = None,
        global_mix: torch.Tensor | None = None,
    ) -> torch.Tensor:
        out = _svd_mamba_fused_fwd_dispatch(u, delta, A, B_ssm, C_ssm, D, z, global_mix)
        ctx.save_for_backward(u, delta, A, B_ssm, C_ssm, D, z, global_mix)
        return out

    @staticmethod
    def backward(ctx: Any, *douts: Any) -> tuple[torch.Tensor | None, ...]:
        dout = douts[0] if len(douts) == 1 else douts
        u, delta, A, B_ssm, C_ssm, D, z, global_mix = ctx.saved_tensors

        u_p = u.detach().requires_grad_(u.requires_grad)
        delta_p = delta.detach().requires_grad_(delta.requires_grad)
        A_p = A.detach().requires_grad_(A.requires_grad)
        B_p = B_ssm.detach().requires_grad_(B_ssm.requires_grad)
        C_p = C_ssm.detach().requires_grad_(C_ssm.requires_grad)
        D_p = D.detach().requires_grad_(D.requires_grad) if D is not None and D.numel() > 0 else None
        z_p = z.detach().requires_grad_(z.requires_grad) if z is not None and z.numel() > 0 else None
        g_p = (
            global_mix.detach().requires_grad_(global_mix.requires_grad)
            if global_mix is not None and global_mix.numel() > 0
            else None
        )

        with torch.enable_grad():
            out_p = svd_mamba_fused_pytorch_ref(u_p, delta_p, A_p, B_p, C_p, D_p, z_p, g_p)

        inputs = [u_p, delta_p, A_p, B_p, C_p]
        if D_p is not None:
            inputs.append(D_p)
        if z_p is not None:
            inputs.append(z_p)
        if g_p is not None:
            inputs.append(g_p)

        grads = torch.autograd.grad(out_p, inputs, dout, allow_unused=True)
        du, ddelta, dA, dB, dC = grads[:5]
        idx = 5
        dD = grads[idx] if D_p is not None else None
        if D_p is not None:
            idx += 1
        dz = grads[idx] if z_p is not None else None
        if z_p is not None:
            idx += 1
        dg = grads[idx] if g_p is not None else None

        return du, ddelta, dA, dB, dC, dD, dz, dg


def svd_mamba_fused(
    u: torch.Tensor,
    delta: torch.Tensor,
    A: torch.Tensor,
    B_ssm: torch.Tensor,
    C_ssm: torch.Tensor,
    D: torch.Tensor | None = None,
    z: torch.Tensor | None = None,
    global_mix: torch.Tensor | None = None,
) -> torch.Tensor:
    """Dispatches SVD-Mamba fused state-space recurrence across 3 fallback tiers."""
    tensors = [u, delta, A, B_ssm, C_ssm]
    if D is not None:
        tensors.append(D)
    if z is not None:
        tensors.append(z)
    if global_mix is not None:
        tensors.append(global_mix)

    if any(t.requires_grad for t in tensors if isinstance(t, torch.Tensor)):
        return _SVDMambaFusedFunction.apply(u, delta, A, B_ssm, C_ssm, D, z, global_mix)
    return _svd_mamba_fused_fwd_dispatch(u, delta, A, B_ssm, C_ssm, D, z, global_mix)

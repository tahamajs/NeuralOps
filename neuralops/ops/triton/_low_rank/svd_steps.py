#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Modularized Low-Rank Submodule
"""

from __future__ import annotations

import logging
import math
import warnings
from typing import Any, Optional, Tuple

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)

from neuralops.ops.triton._types import (
    FORCE_PYTORCH_FALLBACK,
    HAS_TRITON,
    check_triton_suitability,
    low_rank_pytorch_fallback,
    tl,
    triton,
)
from neuralops.ops.triton._low_rank.structured import _fused_hnc_nullspace_spectral_step_fwd_kernel, fused_svd_gemm_triton

def fused_hnc_nullspace_step_pytorch(
    w: torch.Tensor,
    grad_phi: torch.Tensor,
    inv_null_proj: torch.Tensor,
    eta: float,
    restore_grad: torch.Tensor | None = None,
    alpha: float = 0.0,
) -> torch.Tensor:
    """PyTorch reference fallback for fused HNC null-space step."""
    d_proj = torch.matmul(grad_phi.float(), inv_null_proj.float())
    out = w.float() + (eta * d_proj)
    if restore_grad is not None and alpha > 0.0:
        out = out - (alpha * restore_grad.float())
    return out.to(dtype=w.dtype)


class _FusedHNCNullSpaceStepFn(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        w: Any,
        grad_phi: Any,
        inv_null_proj: Any,
        eta: Any,
        restore_grad: Any = None,
        alpha: Any = 0.0,
    ) -> Any:
        ctx.save_for_backward(inv_null_proj)
        ctx.eta = eta
        ctx.alpha = alpha
        ctx.has_restore = restore_grad is not None

        is_suitable, reason = check_triton_suitability(w, kernel_name="fused_hnc_nullspace_step", allow_fp32=True)
        if not is_suitable or not HAS_TRITON:
            logger.warning("Triton kernel 'fused_hnc_nullspace_step' using PyTorch fallback: %s", reason)
            return fused_hnc_nullspace_step_pytorch(w, grad_phi, inv_null_proj, eta, restore_grad, alpha)

        d_out, d_in = w.shape
        w_contig = w.contiguous()
        gp_contig = grad_phi.contiguous()
        res_contig = inv_null_proj.contiguous()
        rg_contig = restore_grad.contiguous() if restore_grad is not None else None

        out = torch.empty_like(w_contig)

        BLOCK_M = 64
        BLOCK_N = 64
        BLOCK_K = 64 if d_in >= 64 else 32

        grid = (
            triton.cdiv(d_out, BLOCK_M),
            triton.cdiv(d_in, BLOCK_N),
        )

        try:
            _fused_hnc_nullspace_spectral_step_fwd_kernel[grid](
                w_contig,
                gp_contig,
                res_contig,
                rg_contig if rg_contig is not None else w_contig,
                out,
                d_out,
                d_in,
                w_contig.stride(0),
                w_contig.stride(1),
                gp_contig.stride(0),
                gp_contig.stride(1),
                res_contig.stride(0),
                res_contig.stride(1),
                rg_contig.stride(0) if rg_contig is not None else 0,
                rg_contig.stride(1) if rg_contig is not None else 0,
                out.stride(0),
                out.stride(1),
                float(eta),
                float(alpha),
                HAS_RESTORE=bool(restore_grad is not None),
                BLOCK_M=BLOCK_M,
                BLOCK_N=BLOCK_N,
                BLOCK_K=BLOCK_K,
            )
            return out
        except Exception as exc:
            logger.warning("Triton kernel 'fused_hnc_nullspace_step' failed (%s); using PyTorch fallback", exc)
            return fused_hnc_nullspace_step_pytorch(w, grad_phi, inv_null_proj, eta, restore_grad, alpha)

    @staticmethod
    def backward(ctx: Any, grad_output: Any) -> tuple[Any, ...]:
        (inv_null_proj,) = ctx.saved_tensors
        grad_w = grad_output.clone()
        grad_grad_phi = ctx.eta * torch.matmul(grad_output.float(), inv_null_proj.t().float()).to(grad_output.dtype)
        grad_inv = None
        grad_restore = -ctx.alpha * grad_output.clone() if ctx.has_restore else None
        return grad_w, grad_grad_phi, grad_inv, None, grad_restore, None


def fused_hnc_nullspace_step_triton(
    w: torch.Tensor,
    grad_phi: torch.Tensor,
    inv_null_proj: torch.Tensor,
    eta: float,
    restore_grad: torch.Tensor | None = None,
    alpha: float = 0.0,
) -> torch.Tensor:
    """Execute fused HNC null-space step with automated Pascal SM < 7.0 fallback."""
    if not w.is_cuda or not HAS_TRITON:
        return fused_hnc_nullspace_step_pytorch(w, grad_phi, inv_null_proj, eta, restore_grad, alpha)
    is_suitable, reason = check_triton_suitability(w, kernel_name="fused_hnc_nullspace_step", allow_fp32=True)
    if not is_suitable:
        logger.warning("Triton kernel 'fused_hnc_nullspace_step' using PyTorch fallback: %s", reason)
        return fused_hnc_nullspace_step_pytorch(w, grad_phi, inv_null_proj, eta, restore_grad, alpha)
    try:
        return _FusedHNCNullSpaceStepFn.apply(w, grad_phi, inv_null_proj, eta, restore_grad, alpha)
    except Exception as exc:
        logger.warning("Triton kernel 'fused_hnc_nullspace_step' failed (%s); using PyTorch fallback", exc)
        return fused_hnc_nullspace_step_pytorch(w, grad_phi, inv_null_proj, eta, restore_grad, alpha)


def block_diag_matmul_triton(
    x: torch.Tensor,
    weights: torch.Tensor,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """Block diagonal matrix multiplication."""
    orig_shape = x.shape
    G, Ng, Kg = weights.shape
    M = x.numel() // (G * Kg)
    x_3d = x.reshape(M, G, Kg)
    y = torch.einsum("mgk,gnk->mgn", x_3d, weights)
    if bias is not None:
        y = y + bias.unsqueeze(0)
    return y.reshape(*orig_shape[:-1], G * Ng)


def rotational_svd_triton(
    x: torch.Tensor,
    r_in: torch.Tensor,
    v_r: torch.Tensor,
    u_r: torch.Tensor,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """Rotational SVD matrix multiplication."""
    orig_shape = x.shape
    K = orig_shape[-1]
    x_2d = x.reshape(-1, K)
    V = v_r if v_r.shape[0] != K else v_r.t()
    R = V.shape[0]
    U = u_r if u_r.shape[1] == R else u_r.t()
    N = U.shape[0]
    x_rot = torch.matmul(x_2d, r_in)
    z = torch.matmul(x_rot, V.t())
    out_2d = torch.matmul(z, U.t())
    if bias is not None:
        out_2d = out_2d + bias
    return out_2d.reshape(*orig_shape[:-1], N)



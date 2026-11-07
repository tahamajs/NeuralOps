#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Fused SVD Projection Triton Kernel (Level 2 Fallback).

Implements single-pass SRAM intermediate projection:
    Y = alpha * ((X * U) * diag(S) * V_T) + beta * X_res + bias
Uses explicit 2D pointer arithmetic (zero block pointers: offs_m[:, None] * stride_m).
Autotuned across BLOCK_M, BLOCK_N, BLOCK_K, num_warps, num_stages.
"""

from __future__ import annotations

import logging
from typing import Any

import torch

from neuralops.ops.cuda._fallback import set_last_dispatch_backend
from neuralops.ops.triton._types import HAS_TRITON, tl, triton

logger = logging.getLogger(__name__)

if HAS_TRITON and tl is not None:
    _CONFIGS = [
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 32, "BLOCK_K": 16}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 32}, num_warps=8, num_stages=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 32}, num_warps=8, num_stages=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 64}, num_warps=8, num_stages=4),
    ]

    @triton.autotune(
        configs=_CONFIGS,
        key=["M", "D_in", "Rank", "D_out"],
    )
    @triton.jit
    def _fused_svd_projection_fwd_kernel(
        X_ptr,
        U_ptr,
        S_ptr,
        VT_ptr,
        X_res_ptr,
        Bias_ptr,
        Out_ptr,
        M,
        D_in,
        Rank,
        D_out,
        alpha,
        beta,
        stride_xm,
        stride_xd,
        stride_uk,
        stride_ur,
        stride_vr,
        stride_vn,
        stride_res_m,
        stride_res_n,
        stride_om,
        stride_on,
        HAS_S: tl.constexpr,
        HAS_RES: tl.constexpr,
        HAS_BIAS: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        """Single-pass Fused SVD Low-Rank Projection Kernel.

        Mainline Triton compliant: zero block pointer usage (no tl.make_block_ptr).
        Accumulates latent intermediate Z = X * U in SRAM tiles.
        """
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

        mask_m = offs_m < M
        mask_n = offs_n < D_out

        out_acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        # Loop over Rank dimension in chunks of BLOCK_K
        for r_start in range(0, Rank, BLOCK_K):
            offs_r = r_start + tl.arange(0, BLOCK_K)
            mask_r = offs_r < Rank

            # Compute latent tile: Z = (X * U) of shape [BLOCK_M, BLOCK_K]
            z_tile = tl.zeros((BLOCK_M, BLOCK_K), dtype=tl.float32)

            for k_start in range(0, D_in, BLOCK_K):
                offs_k = k_start + tl.arange(0, BLOCK_K)
                mask_k = offs_k < D_in

                x_ptrs = X_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xd
                x_val = tl.load(x_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0)

                u_ptrs = U_ptr + offs_k[:, None] * stride_uk + offs_r[None, :] * stride_ur
                u_val = tl.load(u_ptrs, mask=mask_k[:, None] & mask_r[None, :], other=0.0)

                z_tile += tl.dot(x_val, u_val, allow_tf32=True)

            # Apply singular value scaling: diag(S)
            if HAS_S:
                s_val = tl.load(S_ptr + offs_r, mask=mask_r, other=1.0)
                z_tile = z_tile * s_val[None, :]

            # Up-projection: out_acc += Z @ V_T
            vt_ptrs = VT_ptr + offs_r[:, None] * stride_vr + offs_n[None, :] * stride_vn
            vt_val = tl.load(vt_ptrs, mask=mask_r[:, None] & mask_n[None, :], other=0.0)

            out_acc += tl.dot(z_tile.to(vt_val.dtype), vt_val, allow_tf32=True)

        # Scale by alpha
        out_acc = out_acc * alpha

        # Add residual: + beta * X_res
        if HAS_RES:
            res_ptrs = X_res_ptr + offs_m[:, None] * stride_res_m + offs_n[None, :] * stride_res_n
            res_val = tl.load(res_ptrs, mask=mask_m[:, None] & mask_n[None, :], other=0.0)
            out_acc = out_acc + beta * res_val

        # Add bias
        if HAS_BIAS:
            bias_val = tl.load(Bias_ptr + offs_n, mask=mask_n, other=0.0)
            out_acc = out_acc + bias_val[None, :]

        out_ptrs = Out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
        tl.store(
            out_ptrs,
            out_acc.to(Out_ptr.dtype.element_ty),
            mask=mask_m[:, None] & mask_n[None, :],
        )


def fused_svd_projection_triton(
    x: torch.Tensor,
    U: torch.Tensor,
    S: torch.Tensor | None = None,
    V_T: torch.Tensor | None = None,
    X_res: torch.Tensor | None = None,
    bias: torch.Tensor | None = None,
    alpha: float = 1.0,
    beta: float = 1.0,
) -> torch.Tensor:
    """Triton Level 2 Fused SVD Projection Kernel Entrypoint."""
    if not HAS_TRITON or not x.is_cuda:
        from neuralops.ops.cuda.fused_svd_projection.fused_svd_projection_autograd import (
            fused_svd_projection_pytorch_ref,
        )

        return fused_svd_projection_pytorch_ref(x, U, S=S, V_T=V_T, X_res=X_res, bias=bias, alpha=alpha, beta=beta)

    orig_shape = x.shape
    x_2d = x.reshape(-1, x.shape[-1]).contiguous()
    U_c = U.contiguous()
    VT_c = V_T.contiguous()

    M, D_in = x_2d.shape
    _, Rank = U_c.shape
    _, D_out = VT_c.shape

    has_s = S is not None and S.numel() > 0
    S_c = S.contiguous() if has_s else x_2d
    has_res = X_res is not None and X_res.numel() > 0
    X_res_2d = X_res.reshape(-1, X_res.shape[-1]).contiguous() if has_res else x_2d
    has_bias = bias is not None and bias.numel() > 0
    bias_c = bias.contiguous() if has_bias else x_2d

    out = torch.empty((M, D_out), device=x.device, dtype=x.dtype)

    def grid(meta):
        return (
            triton.cdiv(M, meta["BLOCK_M"]),
            triton.cdiv(D_out, meta["BLOCK_N"]),
        )

    _fused_svd_projection_fwd_kernel[grid](
        x_2d,
        U_c,
        S_c,
        VT_c,
        X_res_2d,
        bias_c,
        out,
        M,
        D_in,
        Rank,
        D_out,
        float(alpha),
        float(beta),
        x_2d.stride(0),
        x_2d.stride(1),
        U_c.stride(0),
        U_c.stride(1),
        VT_c.stride(0),
        VT_c.stride(1),
        X_res_2d.stride(0) if has_res else 0,
        X_res_2d.stride(1) if has_res else 0,
        out.stride(0),
        out.stride(1),
        HAS_S=has_s,
        HAS_RES=has_res,
        HAS_BIAS=has_bias,
    )

    set_last_dispatch_backend("fused_svd_projection", "triton")
    return out.reshape(*orig_shape[:-1], D_out)


__all__ = ["fused_svd_projection_triton"]

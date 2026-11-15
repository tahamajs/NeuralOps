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

if HAS_TRITON and tl is not None:

    @triton.jit
    def _triton_gemm_bias_kernel(
        A_ptr,
        B_ptr,
        Bias_ptr,
        C_ptr,
        M,
        N,
        K,
        stride_am,
        stride_ak,
        stride_bk,
        stride_bn,
        stride_cm,
        stride_cn,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
        HAS_BIAS: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)

        offs_am = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_bn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, BLOCK_K)

        a_ptrs = A_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
        b_ptrs = B_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

        accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        for k in range(0, K, BLOCK_K):
            mask_a = (offs_am[:, None] < M) & (offs_k[None, :] < K)
            mask_b = (offs_k[:, None] < K) & (offs_bn[None, :] < N)

            a = tl.load(a_ptrs, mask=mask_a, other=0.0)
            b = tl.load(b_ptrs, mask=mask_b, other=0.0)
            accumulator += tl.dot(a, b)

            a_ptrs += BLOCK_K * stride_ak
            b_ptrs += BLOCK_K * stride_bk
            offs_k += BLOCK_K

        if HAS_BIAS:
            bias = tl.load(Bias_ptr + offs_bn, mask=offs_bn < N, other=0.0)
            accumulator += bias[None, :]

        c_ptrs = C_ptr + (offs_am[:, None] * stride_cm + offs_bn[None, :] * stride_cn)
        mask_c = (offs_am[:, None] < M) & (offs_bn[None, :] < N)
        tl.store(c_ptrs, accumulator.to(C_ptr.dtype.element_ty), mask=mask_c)

    @triton.jit
    def _fused_low_rank_gemm_fwd_kernel(
        X_ptr,
        V_ptr,
        U_ptr,
        S_ptr,
        Bias_ptr,
        Out_ptr,
        M,
        D_in,
        Rank,
        D_out,
        stride_xm,
        stride_xd,
        stride_vd,
        stride_vr,
        stride_ur,
        stride_un,
        stride_om,
        stride_on,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
        BLOCK_R: tl.constexpr,
        HAS_S: tl.constexpr,
        HAS_BIAS: tl.constexpr,
    ):
        """Single-pass Fused Low-Rank GEMM in SRAM:
        Out[m, n] = sum_r ( (sum_d X[m, d] * V[d, r]) * (S[r] if HAS_S else 1.0) * U[r, n] ) + bias[n]
        Accumulates latent rank-r representations strictly within threadblock registers/SRAM.
        Zero intermediate bytes written to HBM!
        """
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        mask_m = offs_m < M
        mask_n = offs_n < D_out

        out_acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        for r_start in range(0, Rank, BLOCK_R):
            offs_r = r_start + tl.arange(0, BLOCK_R)
            mask_r = offs_r < Rank

            z_tile = tl.zeros((BLOCK_M, BLOCK_R), dtype=tl.float32)
            for k_start in range(0, D_in, BLOCK_K):
                offs_k = k_start + tl.arange(0, BLOCK_K)
                mask_k = offs_k < D_in

                x_ptrs = X_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xd
                x_val = tl.load(x_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0)

                v_ptrs = V_ptr + offs_k[:, None] * stride_vd + offs_r[None, :] * stride_vr
                v_val = tl.load(v_ptrs, mask=mask_k[:, None] & mask_r[None, :], other=0.0)

                z_tile += tl.dot(x_val, v_val)

            if HAS_S:
                s_val = tl.load(S_ptr + offs_r, mask=mask_r, other=1.0)
                z_tile = z_tile * s_val[None, :]

            u_ptrs = U_ptr + offs_r[:, None] * stride_ur + offs_n[None, :] * stride_un
            u_val = tl.load(u_ptrs, mask=mask_r[:, None] & mask_n[None, :], other=0.0)

            out_acc += tl.dot(z_tile.to(u_val.dtype), u_val)

        if HAS_BIAS:
            bias_val = tl.load(Bias_ptr + offs_n, mask=mask_n, other=0.0)
            out_acc += bias_val[None, :]

        out_ptrs = Out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
        tl.store(out_ptrs, out_acc.to(Out_ptr.dtype.element_ty), mask=mask_m[:, None] & mask_n[None, :])

    @triton.jit
    def _fused_low_rank_gemm_bwd_dx_kernel(
        GradOut_ptr,
        U_ptr,
        V_ptr,
        S_ptr,
        GradX_ptr,
        M,
        D_in,
        Rank,
        D_out,
        stride_gm,
        stride_gn,
        stride_ur,
        stride_un,
        stride_vd,
        stride_vr,
        stride_xm,
        stride_xd,
        BLOCK_M: tl.constexpr,
        BLOCK_D: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_R: tl.constexpr,
        HAS_S: tl.constexpr,
    ):
        """Single-pass Fused Low-Rank Backward for dx:
        dZ = GradOut @ U^T  [M, Rank]
        dX = (dZ * S) @ V^T  [M, D_in]
        Zero intermediate bytes written to HBM!
        """
        pid_m = tl.program_id(0)
        pid_d = tl.program_id(1)

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
        mask_m = offs_m < M
        mask_d = offs_d < D_in

        dx_acc = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)

        for r_start in range(0, Rank, BLOCK_R):
            offs_r = r_start + tl.arange(0, BLOCK_R)
            mask_r = offs_r < Rank

            dz_tile = tl.zeros((BLOCK_M, BLOCK_R), dtype=tl.float32)
            for n_start in range(0, D_out, BLOCK_N):
                offs_n = n_start + tl.arange(0, BLOCK_N)
                mask_n = offs_n < D_out

                go_ptrs = GradOut_ptr + offs_m[:, None] * stride_gm + offs_n[None, :] * stride_gn
                go_val = tl.load(go_ptrs, mask=mask_m[:, None] & mask_n[None, :], other=0.0)

                u_ptrs = U_ptr + offs_r[None, :] * stride_ur + offs_n[:, None] * stride_un
                u_val = tl.load(u_ptrs, mask=mask_r[None, :] & mask_n[:, None], other=0.0)

                dz_tile += tl.dot(go_val, u_val)

            if HAS_S:
                s_val = tl.load(S_ptr + offs_r, mask=mask_r, other=1.0)
                dz_tile = dz_tile * s_val[None, :]

            v_ptrs = V_ptr + offs_d[None, :] * stride_vd + offs_r[:, None] * stride_vr
            v_val = tl.load(v_ptrs, mask=mask_d[None, :] & mask_r[:, None], other=0.0)

            dx_acc += tl.dot(dz_tile.to(v_val.dtype), v_val)

        gx_ptrs = GradX_ptr + offs_m[:, None] * stride_xm + offs_d[None, :] * stride_xd
        tl.store(gx_ptrs, dx_acc.to(GradX_ptr.dtype.element_ty), mask=mask_m[:, None] & mask_d[None, :])

    @triton.jit
    def _fused_nystrom_pinv_step_fwd_kernel(
        F1_ptr,
        F2_inv_ptr,
        F3_ptr,
        V_ptr,
        Out_ptr,
        M,
        K1,
        K2,
        N,
        stride_f1_m,
        stride_f1_k,
        stride_f2_k1,
        stride_f2_k2,
        stride_f3_k2,
        stride_f3_n,
        stride_v_n,
        stride_v_d,
        stride_o_m,
        stride_o_d,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        """Fused Nystrom pinv landmark contraction step in SRAM."""
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for k in range(0, K1, BLOCK_K):
            offs_k = k + tl.arange(0, BLOCK_K)
            f1 = tl.load(
                F1_ptr + offs_m[:, None] * stride_f1_m + offs_k[None, :] * stride_f1_k,
                mask=offs_m[:, None] < M and offs_k[None, :] < K1,
                other=0.0,
            )
            f2 = tl.load(
                F2_inv_ptr + offs_k[:, None] * stride_f2_k1 + offs_n[None, :] * stride_f2_k2,
                mask=offs_k[:, None] < K1 and offs_n[None, :] < K2,
                other=0.0,
            )
            acc += tl.dot(f1, f2)
        tl.store(
            Out_ptr + offs_m[:, None] * stride_o_m + offs_n[None, :] * stride_o_d,
            acc.to(Out_ptr.dtype.element_ty),
            mask=offs_m[:, None] < M and offs_n[None, :] < N,
        )

    @triton.jit
    def _fused_rpc_sketch_attention_fwd_kernel(
        Q_ptr,
        K_ptr,
        Out_ptr,
        scale,
        M,
        N,
        K,
        stride_qm,
        stride_qk,
        stride_kn,
        stride_kk,
        stride_om,
        stride_on,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        """Fused RPC sketched query-key contraction."""
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for k in range(0, K, BLOCK_K):
            offs_k = k + tl.arange(0, BLOCK_K)
            q = tl.load(
                Q_ptr + offs_m[:, None] * stride_qm + offs_k[None, :] * stride_qk,
                mask=offs_m[:, None] < M and offs_k[None, :] < K,
                other=0.0,
            )
            k_val = tl.load(
                K_ptr + offs_n[:, None] * stride_kn + offs_k[None, :] * stride_kk,
                mask=offs_n[:, None] < N and offs_k[None, :] < K,
                other=0.0,
            )
            acc += tl.dot(q, tl.trans(k_val))
        acc = acc * scale
        tl.store(
            Out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on,
            acc.to(Out_ptr.dtype.element_ty),
            mask=offs_m[:, None] < M and offs_n[None, :] < N,
        )

    @triton.jit
    def _fused_primal_dual_landmark_fwd_kernel(
        F1_ptr,
        X_ptr,
        F3_ptr,
        Out_ptr,
        M,
        N,
        K,
        stride_f1_m,
        stride_f1_k,
        stride_x_k,
        stride_x_n,
        stride_f3_n,
        stride_f3_d,
        stride_om,
        stride_od,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        """Fused Primal-Dual landmark factor contraction."""
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for k in range(0, K, BLOCK_K):
            offs_k = k + tl.arange(0, BLOCK_K)
            f1 = tl.load(
                F1_ptr + offs_m[:, None] * stride_f1_m + offs_k[None, :] * stride_f1_k,
                mask=offs_m[:, None] < M and offs_k[None, :] < K,
                other=0.0,
            )
            x = tl.load(
                X_ptr + offs_k[:, None] * stride_x_k + offs_n[None, :] * stride_x_n,
                mask=offs_k[:, None] < K and offs_n[None, :] < N,
                other=0.0,
            )
            acc += tl.dot(f1, x)
        tl.store(
            Out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_od,
            acc.to(Out_ptr.dtype.element_ty),
            mask=offs_m[:, None] < M and offs_n[None, :] < N,
        )

    @triton.jit
    def _fused_svd_mamba_proj_fwd_kernel(
        X_ptr,
        V_ptr,
        S_ptr,
        U_ptr,
        Bias_ptr,
        Out_ptr,
        M,
        D_in,
        Rank,
        D_out,
        stride_xm,
        stride_xd,
        stride_vd,
        stride_vr,
        stride_ur,
        stride_un,
        stride_om,
        stride_on,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
        BLOCK_R: tl.constexpr,
        HAS_BIAS: tl.constexpr,
    ):
        """Fused SVD Mamba low-rank projection in SRAM."""
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_r = tl.arange(0, BLOCK_R)
        mask_m = offs_m < M
        mask_n = offs_n < D_out
        mask_r = offs_r < Rank
        s_val = tl.load(S_ptr + offs_r, mask=mask_r, other=1.0)
        z = tl.zeros((BLOCK_M, BLOCK_R), dtype=tl.float32)
        for k in range(0, D_in, BLOCK_K):
            offs_k = k + tl.arange(0, BLOCK_K)
            mask_k = offs_k < D_in
            x = tl.load(
                X_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xd,
                mask=mask_m[:, None] & mask_k[None, :],
                other=0.0,
            )
            v = tl.load(
                V_ptr + offs_k[:, None] * stride_vd + offs_r[None, :] * stride_vr,
                mask=mask_k[:, None] & mask_r[None, :],
                other=0.0,
            )
            z += tl.dot(x, v)
        z = z * s_val[None, :]
        u = tl.load(
            U_ptr + offs_r[:, None] * stride_ur + offs_n[None, :] * stride_un,
            mask=mask_r[:, None] & mask_n[None, :],
            other=0.0,
        )
        out = tl.dot(z.to(u.dtype), u)
        if HAS_BIAS:
            bias = tl.load(Bias_ptr + offs_n, mask=mask_n, other=0.0)
            out = out + bias[None, :]
        tl.store(
            Out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on,
            out.to(Out_ptr.dtype.element_ty),
            mask=mask_m[:, None] & mask_n[None, :],
        )


else:
    _triton_gemm_bias_kernel = None
    _fused_low_rank_gemm_fwd_kernel = None
    _fused_low_rank_gemm_bwd_dx_kernel = None
    _fused_nystrom_pinv_step_fwd_kernel = None
    _fused_rpc_sketch_attention_fwd_kernel = None
    _fused_primal_dual_landmark_fwd_kernel = None
    _fused_svd_mamba_proj_fwd_kernel = None


class _TritonLowRankAutogradFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        x: Any,
        U: Any,
        V: Any,
        bias: Any,
        S: Any = None,
    ) -> Any:
        orig_shape = x.shape
        D_in = orig_shape[-1]
        x_2d = x.contiguous().view(-1, D_in)
        M = x_2d.shape[0]

        V_mat = (V.t() if V.shape[-1] == D_in else V).contiguous()
        Rank = V_mat.shape[1]

        U_mat = (U.t() if U.shape[-1] == Rank else U).contiguous()
        D_out = U_mat.shape[1]

        has_bias = bias is not None
        has_s = S is not None
        bias_t = bias.contiguous() if has_bias else x_2d
        s_t = S.contiguous() if has_s else x_2d

        if HAS_TRITON and x.is_cuda and Rank <= 128:
            y = torch.empty((M, D_out), dtype=x.dtype, device=x.device)
            BLOCK_M = 64
            BLOCK_N = 64
            BLOCK_K = 32
            BLOCK_R = max(16, min(32, triton.next_power_of_2(Rank)))
            grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(D_out, BLOCK_N))

            _fused_low_rank_gemm_fwd_kernel[grid](
                x_2d,
                V_mat,
                U_mat,
                s_t,
                bias_t,
                y,
                M,
                D_in,
                Rank,
                D_out,
                x_2d.stride(0),
                x_2d.stride(1),
                V_mat.stride(0),
                V_mat.stride(1),
                U_mat.stride(0),
                U_mat.stride(1),
                y.stride(0),
                y.stride(1),
                BLOCK_M=BLOCK_M,
                BLOCK_N=BLOCK_N,
                BLOCK_K=BLOCK_K,
                BLOCK_R=BLOCK_R,
                HAS_S=has_s,
                HAS_BIAS=has_bias,
            )
            ctx.save_for_backward(x, U, V, bias, S)
            return y.view(*orig_shape[:-1], D_out)

        z = torch.empty((M, Rank), dtype=x.dtype, device=x.device)
        grid_z = (triton.cdiv(M, 64), triton.cdiv(Rank, 64))
        _triton_gemm_bias_kernel[grid_z](
            x_2d,
            V_mat,
            V_mat,
            z,
            M,
            Rank,
            D_in,
            x_2d.stride(0),
            x_2d.stride(1),
            V_mat.stride(0),
            V_mat.stride(1),
            z.stride(0),
            z.stride(1),
            BLOCK_M=64,
            BLOCK_N=64,
            BLOCK_K=32,
            HAS_BIAS=False,
        )
        if has_s:
            z = z * S.view(1, -1)

        y = torch.empty((M, D_out), dtype=x.dtype, device=x.device)
        grid_y = (triton.cdiv(M, 64), triton.cdiv(D_out, 64))
        _triton_gemm_bias_kernel[grid_y](
            z,
            U_mat,
            bias_t,
            y,
            M,
            D_out,
            Rank,
            z.stride(0),
            z.stride(1),
            U_mat.stride(0),
            U_mat.stride(1),
            y.stride(0),
            y.stride(1),
            BLOCK_M=64,
            BLOCK_N=64,
            BLOCK_K=32,
            HAS_BIAS=has_bias,
        )

        ctx.save_for_backward(x, U, V, bias, S)
        return y.view(*orig_shape[:-1], D_out)

    @staticmethod
    def backward(ctx: Any, grad_out: Any) -> tuple[Any, ...]:
        x, U, V, bias, S = ctx.saved_tensors
        orig_shape = x.shape
        D_in = orig_shape[-1]
        x_2d = x.contiguous().view(-1, D_in)
        M = x_2d.shape[0]

        V_mat = (V.t() if V.shape[-1] == D_in else V).contiguous()
        Rank = V_mat.shape[1]

        U_mat = (U.t() if U.shape[-1] == Rank else U).contiguous()
        D_out = U_mat.shape[1]

        grad_out_2d = grad_out.contiguous().view(-1, D_out)

        g_x = None
        g_U = None
        g_V = None
        g_bias = None
        g_S = None

        if bias is not None and ctx.needs_input_grad[3]:
            g_bias = grad_out_2d.sum(dim=0).to(bias.dtype)

        has_s = S is not None
        s_t = S.contiguous() if has_s else x_2d

        if ctx.needs_input_grad[0]:
            if HAS_TRITON and x.is_cuda and Rank <= 128:
                g_x_2d = torch.empty((M, D_in), dtype=x.dtype, device=x.device)
                BLOCK_M = 64
                BLOCK_D = 64
                BLOCK_N = 32
                BLOCK_R = max(16, min(32, triton.next_power_of_2(Rank)))
                grid_dx = (triton.cdiv(M, BLOCK_M), triton.cdiv(D_in, BLOCK_D))
                _fused_low_rank_gemm_bwd_dx_kernel[grid_dx](
                    grad_out_2d,
                    U_mat,
                    V_mat,
                    s_t,
                    g_x_2d,
                    M,
                    D_in,
                    Rank,
                    D_out,
                    grad_out_2d.stride(0),
                    grad_out_2d.stride(1),
                    U_mat.stride(0),
                    U_mat.stride(1),
                    V_mat.stride(0),
                    V_mat.stride(1),
                    g_x_2d.stride(0),
                    g_x_2d.stride(1),
                    BLOCK_M=BLOCK_M,
                    BLOCK_D=BLOCK_D,
                    BLOCK_N=BLOCK_N,
                    BLOCK_R=BLOCK_R,
                    HAS_S=has_s,
                )
                g_x = g_x_2d.view(*orig_shape).to(x.dtype)
            else:
                dz = torch.matmul(grad_out_2d, U_mat.t())
                if has_s:
                    dz = dz * S.view(1, -1)
                g_x_2d = torch.matmul(dz, V_mat.t())
                g_x = g_x_2d.view(*orig_shape).to(x.dtype)

        if ctx.needs_input_grad[1] or ctx.needs_input_grad[2] or (has_s and ctx.needs_input_grad[4]):
            z = torch.matmul(x_2d, V_mat)
            dz = torch.matmul(grad_out_2d, U_mat.t())

            if has_s:
                if ctx.needs_input_grad[4]:
                    g_S = (z * dz).sum(dim=0).to(S.dtype)
                z_scaled = z * S.view(1, -1)
                dz_scaled = dz * S.view(1, -1)
            else:
                z_scaled = z
                dz_scaled = dz

            if ctx.needs_input_grad[1]:
                g_U_mat = torch.matmul(z_scaled.t(), grad_out_2d)
                g_U = g_U_mat.t().to(U.dtype) if U.shape[-1] == Rank else g_U_mat.to(U.dtype)

            if ctx.needs_input_grad[2]:
                g_V_mat = torch.matmul(x_2d.t(), dz_scaled)
                g_V = g_V_mat.t().to(V.dtype) if V.shape[-1] == D_in else g_V_mat.to(V.dtype)

        return g_x, g_U, g_V, g_bias, g_S


def is_supported(
    x: torch.Tensor,
    U: torch.Tensor,
    V: torch.Tensor,
    bias: torch.Tensor | None = None,
    **kwargs: Any,
) -> bool:
    """Check whether low-rank projection can run on the Triton kernel."""
    suitable, _ = check_triton_suitability(x, "low_rank")
    return suitable and U.is_cuda and V.is_cuda


def fwd(
    x: torch.Tensor,
    U: torch.Tensor,
    V: torch.Tensor,
    bias: torch.Tensor | None = None,
    S: torch.Tensor | None = None,
    **kwargs: Any,
) -> torch.Tensor:
    """Forward entrypoint for single-pass fused Triton low-rank projection: y = (x @ V.T) @ U.T + bias."""
    if not is_supported(x, U, V, bias=bias, **kwargs):
        return low_rank_pytorch_fallback(x, U, V, bias=bias, **kwargs)
    return _TritonLowRankAutogradFunction.apply(x, U, V, bias, S)


def bwd(
    x: torch.Tensor,
    U: torch.Tensor,
    V: torch.Tensor,
    grad_out: torch.Tensor,
    bias: torch.Tensor | None = None,
    S: torch.Tensor | None = None,
    **kwargs: Any,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Backward entrypoint computing gradients for x, U, V, and bias."""
    x_req = x.detach().requires_grad_(True)
    U_req = U.detach().requires_grad_(True)
    V_req = V.detach().requires_grad_(True)
    b_req = bias.detach().requires_grad_(True) if bias is not None else None
    s_req = S.detach().requires_grad_(True) if S is not None else None

    diff_vars = [x_req, U_req, V_req]
    if b_req is not None:
        diff_vars.append(b_req)
    if s_req is not None:
        diff_vars.append(s_req)

    out = _TritonLowRankAutogradFunction.apply(x_req, U_req, V_req, b_req, s_req)
    grads = torch.autograd.grad(out, diff_vars, grad_out)
    return grads[0], grads[1], grads[2], (grads[3] if b_req is not None else None)


def low_rank_forward(
    x: torch.Tensor,
    V: torch.Tensor,
    S: torch.Tensor,
    U: torch.Tensor,
    gate: torch.Tensor | None = None,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """Forward entrypoint for low-rank bottleneck with singular values and optional dynamic gate.

    Computes:
      z = (x @ V)
      z = z * (S * gate)
      out = z @ U.T + bias
    """
    from neuralops.ops.triton._types import FORCE_PYTORCH_FALLBACK, check_triton_suitability

    if x.is_cuda and HAS_TRITON and not FORCE_PYTORCH_FALLBACK and gate is None:
        suitable, _ = check_triton_suitability(x, "low_rank")
        if suitable and is_supported(x, U, V, bias=bias):
            return fwd(x, U, V, bias=bias, S=S)

    v = V.to(x.dtype)
    s = S.to(x.dtype)
    u = U.to(x.dtype)
    z = torch.matmul(x, v)
    if gate is not None:
        g = gate.to(x.dtype)
        z = z * s.unsqueeze(0).unsqueeze(0) * g.unsqueeze(1)
    else:
        z = z * s.unsqueeze(0).unsqueeze(0)
    out = torch.matmul(z, u.t().contiguous())
    if bias is not None:
        out = out + bias.to(x.dtype)
    return out



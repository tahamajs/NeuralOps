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


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_N": 32, "BLOCK_K": 32, "BLOCK_R": 16}, num_stages=2, num_warps=2),
        triton.Config({"BLOCK_N": 64, "BLOCK_K": 32, "BLOCK_R": 16}, num_stages=3, num_warps=4),
        triton.Config({"BLOCK_N": 32, "BLOCK_K": 64, "BLOCK_R": 16}, num_stages=3, num_warps=4),
        triton.Config({"BLOCK_N": 64, "BLOCK_K": 64, "BLOCK_R": 32}, num_stages=4, num_warps=4),
    ],
    key=["N", "K", "R"],
)
@triton.jit
def _strided_tt_reconstruct_kernel(
    core0_ptr,
    core1_ptr,
    w_ptr,
    N,
    K,
    R,
    p0_dim,
    p1_dim,
    q0_dim,
    q1_dim,
    stride_c0_p,
    stride_c0_q,
    stride_c0_r,
    stride_c1_r,
    stride_c1_p,
    stride_c1_q,
    stride_wn,
    stride_wk,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_R: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)

    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)

    mask_n = offs_n < N
    mask_k = offs_k < K

    q0 = offs_n // q1_dim
    q1 = offs_n % q1_dim
    p0 = offs_k // p1_dim
    p1 = offs_k % p1_dim

    acc = tl.zeros((BLOCK_N, BLOCK_K), dtype=tl.float32)

    num_r_blocks = tl.cdiv(R, BLOCK_R)
    for r_idx in range(num_r_blocks):
        offs_r = r_idx * BLOCK_R + tl.arange(0, BLOCK_R)
        mask_r = offs_r < R

        c0_ptrs = (
            core0_ptr
            + p0[None, :, None] * stride_c0_p
            + q0[:, None, None] * stride_c0_q
            + offs_r[None, None, :] * stride_c0_r
        )
        c1_ptrs = (
            core1_ptr
            + offs_r[None, None, :] * stride_c1_r
            + p1[None, :, None] * stride_c1_p
            + q1[:, None, None] * stride_c1_q
        )

        mask_3d = mask_n[:, None, None] & mask_k[None, :, None] & mask_r[None, None, :]

        c0_vals = tl.load(c0_ptrs, mask=mask_3d, other=0.0)
        c1_vals = tl.load(c1_ptrs, mask=mask_3d, other=0.0)

        acc += tl.sum(c0_vals * c1_vals, axis=2)

    w_ptrs = w_ptr + offs_n[:, None] * stride_wn + offs_k[None, :] * stride_wk
    tl.store(w_ptrs, acc.to(w_ptr.dtype.element_ty), mask=mask_n[:, None] & mask_k[None, :])


def triton_strided_tt_contraction_pytorch_ref(
    core0: torch.Tensor,
    core1: torch.Tensor,
    p_dims: Tuple[int, int],
    q_dims: Tuple[int, int],
) -> torch.Tensor:
    c0 = core0.squeeze(0) if core0.dim() == 4 and core0.shape[0] == 1 else core0
    c1 = core1.squeeze(-1) if core1.dim() == 4 and core1.shape[-1] == 1 else core1
    p0, p1 = p_dims
    q0, q1 = q_dims
    N = q0 * q1
    K = p0 * p1
    res = torch.tensordot(c0, c1, dims=([-1], [0]))
    res = res.permute(1, 3, 0, 2).contiguous()
    return res.view(N, K)


def triton_strided_tt_contraction(
    core0: torch.Tensor,
    core1: torch.Tensor,
    p_dims: Tuple[int, int],
    q_dims: Tuple[int, int],
) -> torch.Tensor:
    c0 = core0.squeeze(0) if core0.dim() == 4 and core0.shape[0] == 1 else core0
    c1 = core1.squeeze(-1) if core1.dim() == 4 and core1.shape[-1] == 1 else core1

    p0, p1 = p_dims
    q0, q1 = q_dims
    R = c0.shape[-1]
    K = p0 * p1
    N = q0 * q1

    if not c0.is_cuda:
        return triton_strided_tt_contraction_pytorch_ref(core0, core1, p_dims, q_dims)

    W = torch.empty((N, K), device=c0.device, dtype=c0.dtype)
    grid = lambda META: (
        triton.cdiv(N, META["BLOCK_N"]),
        triton.cdiv(K, META["BLOCK_K"]),
    )

    c0_c = c0.contiguous()
    c1_c = c1.contiguous()

    _strided_tt_reconstruct_kernel[grid](
        c0_c,
        c1_c,
        W,
        N,
        K,
        R,
        p0,
        p1,
        q0,
        q1,
        c0_c.stride(0),
        c0_c.stride(1),
        c0_c.stride(2),
        c1_c.stride(0),
        c1_c.stride(1),
        c1_c.stride(2),
        W.stride(0),
        W.stride(1),
    )

    return W


def triton_strided_tt_linear(
    x: torch.Tensor,
    core0: torch.Tensor,
    core1: torch.Tensor,
    p_dims: Tuple[int, int],
    q_dims: Tuple[int, int],
    bias: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    orig_shape = x.shape
    in_features = orig_shape[-1]
    x_2d = x.view(-1, in_features)

    W = triton_strided_tt_contraction(core0, core1, p_dims, q_dims)
    out = torch.matmul(x_2d, W.t())
    if bias is not None:
        out = out + bias.to(dtype=out.dtype, device=out.device)

    out_features = q_dims[0] * q_dims[1]
    return out.view(*orig_shape[:-1], out_features)


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 16}, num_stages=2, num_warps=2),
        triton.Config({"BLOCK_M": 32}, num_stages=3, num_warps=4),
        triton.Config({"BLOCK_M": 64}, num_stages=3, num_warps=4),
    ],
    key=["M", "M1", "M2", "N1", "N2"],
)
@triton.jit
def _fused_kronecker_gemm_kernel(
    x_ptr,
    a_ptr,
    b_ptr,
    bias_ptr,
    y_ptr,
    M,
    M1: tl.constexpr,
    M2: tl.constexpr,
    N1: tl.constexpr,
    N2: tl.constexpr,
    stride_xm,
    stride_xk,
    stride_a_m,
    stride_a_n,
    stride_b_m,
    stride_b_n,
    stride_ym,
    stride_yn,
    HAS_BIAS: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    pid_m = tl.program_id(0)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_m = offs_m < M

    offs_m1 = tl.arange(0, M1)
    offs_n1 = tl.arange(0, N1)
    offs_m2 = tl.arange(0, M2)
    offs_n2 = tl.arange(0, N2)

    a_tile = tl.load(a_ptr + offs_m1[:, None] * stride_a_m + offs_n1[None, :] * stride_a_n)
    b_tile = tl.load(b_ptr + offs_m2[:, None] * stride_b_m + offs_n2[None, :] * stride_b_n)

    for m_idx in range(BLOCK_M):
        curr_m = pid_m * BLOCK_M + m_idx
        if curr_m < M:
            flat_offs_k = offs_n1[:, None] * N2 + offs_n2[None, :]
            x_ptrs = x_ptr + curr_m * stride_xm + flat_offs_k * stride_xk
            x_slice = tl.load(x_ptrs)

            t = tl.dot(x_slice.to(tl.float32), tl.trans(b_tile).to(tl.float32), allow_tf32=False)
            y_slice = tl.dot(a_tile.to(tl.float32), t.to(tl.float32), allow_tf32=False)

            flat_offs_out = offs_m1[:, None] * M2 + offs_m2[None, :]
            y_ptrs = y_ptr + curr_m * stride_ym + flat_offs_out * stride_yn

            out_val = y_slice
            if HAS_BIAS:
                b_val = tl.load(bias_ptr + flat_offs_out)
                out_val += b_val.to(tl.float32)

            tl.store(y_ptrs, out_val.to(x_ptr.dtype.element_ty))


def triton_fused_kronecker_matmul_pytorch_ref(
    x: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    orig_shape = x.shape
    x_2d = x.view(-1, orig_shape[-1])
    M, K = x_2d.shape
    M1, N1 = A.shape
    M2, N2 = B.shape
    out_features = M1 * M2
    X_reshaped = x_2d.view(M, N1, N2)
    T = torch.matmul(X_reshaped, B.t())
    Y = torch.matmul(A, T)
    Y_flat = Y.reshape(M, out_features)
    if bias is not None:
        Y_flat = Y_flat + bias.to(dtype=Y_flat.dtype, device=Y_flat.device)
    return Y_flat.view(*orig_shape[:-1], out_features)


def triton_fused_kronecker_matmul(
    x: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    orig_shape = x.shape
    x_2d = x.view(-1, orig_shape[-1])
    M, K = x_2d.shape
    M1, N1 = A.shape
    M2, N2 = B.shape
    out_features = M1 * M2

    assert K == N1 * N2, f"Input feature mismatch: {K} != {N1} * {N2}"

    if (
        not x.is_cuda
        or N1 < 16
        or N2 < 16
        or M1 < 16
        or M2 < 16
        or (N1 & (N1 - 1)) != 0
        or (N2 & (N2 - 1)) != 0
        or (M1 & (M1 - 1)) != 0
        or (M2 & (M2 - 1)) != 0
        or M1 > 64
        or M2 > 64
        or N1 > 64
        or N2 > 64
    ):
        return triton_fused_kronecker_matmul_pytorch_ref(x, A, B, bias)

    Y = torch.empty((M, out_features), device=x.device, dtype=x.dtype)
    grid = lambda META: (triton.cdiv(M, META["BLOCK_M"]),)

    has_bias = bias is not None
    bias_t = bias.contiguous() if has_bias else torch.empty(0, device=x.device, dtype=x.dtype)
    x_c = x_2d.contiguous()
    A_c = A.contiguous()
    B_c = B.contiguous()

    try:
        _fused_kronecker_gemm_kernel[grid](
            x_c,
            A_c,
            B_c,
            bias_t,
            Y,
            M,
            M1=M1,
            M2=M2,
            N1=N1,
            N2=N2,
            stride_xm=x_c.stride(0),
            stride_xk=x_c.stride(1),
            stride_a_m=A_c.stride(0),
            stride_a_n=A_c.stride(1),
            stride_b_m=B_c.stride(0),
            stride_b_n=B_c.stride(1),
            stride_ym=Y.stride(0),
            stride_yn=Y.stride(1),
            HAS_BIAS=has_bias,
        )
        return Y.view(*orig_shape[:-1], out_features)
    except Exception as exc:
        logger.warning("Triton kernel 'fused_kronecker_gemm' failed (%s); using PyTorch fallback", exc)
        return triton_fused_kronecker_matmul_pytorch_ref(x, A, B, bias)


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 16, "BLOCK_N": 64}, num_stages=2, num_warps=4),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 64}, num_stages=3, num_warps=4),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 128}, num_stages=3, num_warps=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64}, num_stages=3, num_warps=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128}, num_stages=3, num_warps=4),
    ],
    key=["M", "N", "eff_r"],
)
@triton.jit
def _spectral_entropy_svd_proj_kernel(
    z_ptr,
    u_ptr,
    b_ptr,
    y_ptr,
    M,
    N,
    eff_r,
    stride_zm,
    stride_zr,
    stride_ur,
    stride_un,
    stride_ym,
    stride_yn,
    HAS_BIAS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_R: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_r = tl.arange(0, BLOCK_R)

    mask_m = offs_m < M
    mask_n = offs_n < N
    mask_r = offs_r < eff_r

    z_ptrs = z_ptr + offs_m[:, None] * stride_zm + offs_r[None, :] * stride_zr
    u_ptrs = u_ptr + offs_r[:, None] * stride_ur + offs_n[None, :] * stride_un

    z_tile = tl.load(z_ptrs, mask=mask_m[:, None] & mask_r[None, :], other=0.0)
    u_tile = tl.load(u_ptrs, mask=mask_r[:, None] & mask_n[None, :], other=0.0)

    y_acc = tl.dot(z_tile, u_tile, allow_tf32=True)

    if HAS_BIAS:
        b_tile = tl.load(b_ptr + offs_n, mask=mask_n, other=0.0)
        y_acc += b_tile[None, :].to(tl.float32)

    y_ptrs = y_ptr + offs_m[:, None] * stride_ym + offs_n[None, :] * stride_yn
    tl.store(y_ptrs, y_acc.to(y_ptr.dtype.element_ty), mask=mask_m[:, None] & mask_n[None, :])


def triton_spectral_entropy_svd_gate_pytorch_ref(
    x: torch.Tensor,
    V_r: Optional[torch.Tensor] = None,
    U_r: Optional[torch.Tensor] = None,
    effective_rank: Optional[int] = None,
    s_weights: Optional[torch.Tensor] = None,
    bias: Optional[torch.Tensor] = None,
    threshold: float = 1e-6,
    **kwargs: Any,
) -> Any:
    if V_r is None or U_r is None or effective_rank is None:
        total_s = x.sum() + threshold
        p = torch.clamp(x / total_s, min=threshold, max=1.0)
        ent = -torch.sum(p * torch.log2(p))
        return ent, None
    orig_shape = x.shape
    x_2d = x.view(-1, orig_shape[-1])
    M, K = x_2d.shape
    if V_r.shape[0] == K:
        V_mat = V_r
    else:
        V_mat = V_r.t().contiguous()
    R_MAX = V_mat.shape[1]
    eff_r = min(int(effective_rank), R_MAX)
    if U_r.shape[0] == R_MAX:
        U_mat = U_r
    else:
        U_mat = U_r.t().contiguous()
    N = U_mat.shape[1]

    V_sub = V_mat[:, :eff_r]
    U_sub = U_mat[:eff_r, :]
    z = torch.matmul(x_2d, V_sub)
    if s_weights is not None:
        z = z * s_weights[:eff_r].view(1, -1).to(dtype=z.dtype, device=z.device)
    y = torch.matmul(z, U_sub)
    if bias is not None:
        y = y + bias.to(dtype=y.dtype, device=y.device)
    return y.view(*orig_shape[:-1], N)


def triton_spectral_entropy_svd_gate(
    x: torch.Tensor,
    V_r: Optional[torch.Tensor] = None,
    U_r: Optional[torch.Tensor] = None,
    effective_rank: Optional[int] = None,
    s_weights: Optional[torch.Tensor] = None,
    bias: Optional[torch.Tensor] = None,
    threshold: float = 1e-6,
    **kwargs: Any,
) -> Any:
    if V_r is None or U_r is None or effective_rank is None:
        return triton_spectral_entropy_svd_gate_pytorch_ref(
            x, V_r, U_r, effective_rank, s_weights=s_weights, bias=bias, threshold=threshold, **kwargs
        )
    orig_shape = x.shape
    x_2d = x.view(-1, orig_shape[-1])
    M, K = x_2d.shape

    if V_r.shape[0] == K:
        V_mat = V_r
    else:
        V_mat = V_r.t().contiguous()

    R_MAX = V_mat.shape[1]
    eff_r = min(int(effective_rank), R_MAX)

    if U_r.shape[0] == R_MAX:
        U_mat = U_r
    else:
        U_mat = U_r.t().contiguous()

    N = U_mat.shape[1]

    if not x.is_cuda or R_MAX > 256:
        return triton_spectral_entropy_svd_gate_pytorch_ref(x, V_r, U_r, effective_rank, s_weights, bias)

    V_sub = V_mat[:, :eff_r]
    U_sub = U_mat[:eff_r, :].contiguous()
    if s_weights is not None:
        V_sub = V_sub * s_weights[:eff_r].view(1, -1).to(dtype=V_sub.dtype, device=V_sub.device)

    z = torch.matmul(x_2d, V_sub)

    y = torch.empty((M, N), device=x.device, dtype=x.dtype)
    BLOCK_R = max(16, triton.next_power_of_2(eff_r))

    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_M"]),
        triton.cdiv(N, META["BLOCK_N"]),
    )

    has_bias = bias is not None
    b_t = bias.contiguous() if has_bias else torch.empty(0, device=x.device, dtype=x.dtype)

    _spectral_entropy_svd_proj_kernel[grid](
        z,
        U_sub,
        b_t,
        y,
        M,
        N,
        eff_r,
        z.stride(0),
        z.stride(1),
        U_sub.stride(0),
        U_sub.stride(1),
        y.stride(0),
        y.stride(1),
        HAS_BIAS=has_bias,
        BLOCK_R=BLOCK_R,
    )

    return y.view(*orig_shape[:-1], N)



#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Fused Low-Rank Gated GEMM

Fuses low-rank bottleneck projections and non-linear gating mechanisms into a single memory-bandwidth efficient kernel pass.
"""

from __future__ import annotations

from typing import Optional

import torch
import triton
import triton.language as tl

__all__ = ["triton_fused_lowrank_gated_gemm", "fused_lowrank_gemm"]


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 64, "BLOCK_K": 32}, num_stages=3, num_warps=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32}, num_stages=3, num_warps=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 32}, num_stages=4, num_warps=8),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 32}, num_stages=4, num_warps=8),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 32, "BLOCK_K": 64}, num_stages=2, num_warps=4),
    ],
    key=["M", "N", "K", "R"],
)
@triton.jit
def _fused_lowrank_gated_gemm_1d_kernel(
    x_ptr,
    v_ptr,
    u_ptr,
    alpha_ptr,
    b_ptr,
    y_ptr,
    M,
    N,
    K,
    R,
    stride_xm,
    stride_xk,
    stride_vk,
    stride_vr,
    stride_ur,
    stride_un,
    stride_ym,
    stride_yn,
    HAS_ALPHA: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_R: tl.constexpr,
):
    """
    1D Grid Kernel across M (token dimension).
    Each thread block retains Z_tile in SRAM/registers across the entire N dimension,
    achieving 100% SRAM residency for the intermediate low-rank representation.
    """
    pid_m = tl.program_id(0)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_m = offs_m < M

    offs_r = tl.arange(0, BLOCK_R)
    mask_r = offs_r < R

    z_acc = tl.zeros((BLOCK_M, BLOCK_R), dtype=tl.float32)

    num_k_blocks = tl.cdiv(K, BLOCK_K)
    for k_idx in range(0, num_k_blocks):
        offs_k = k_idx * BLOCK_K + tl.arange(0, BLOCK_K)
        mask_k = offs_k < K

        x_ptrs = x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk
        v_ptrs = v_ptr + offs_k[:, None] * stride_vk + offs_r[None, :] * stride_vr

        x_tile = tl.load(x_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0)
        v_tile = tl.load(v_ptrs, mask=mask_k[:, None] & mask_r[None, :], other=0.0)

        z_acc += tl.dot(x_tile, v_tile, allow_tf32=True)

    if HAS_ALPHA:
        alpha_ptrs = alpha_ptr + offs_r
        alpha_tile = tl.load(alpha_ptrs, mask=mask_r, other=0.0)
        gate = tl.sigmoid(alpha_tile.to(tl.float32))
        z_acc = z_acc * gate[None, :]

    z_tile = z_acc.to(x_ptr.dtype.element_ty)

    num_n_blocks = tl.cdiv(N, BLOCK_N)
    for n_idx in range(0, num_n_blocks):
        offs_n = n_idx * BLOCK_N + tl.arange(0, BLOCK_N)
        mask_n = offs_n < N

        u_ptrs = u_ptr + offs_r[:, None] * stride_ur + offs_n[None, :] * stride_un
        u_tile = tl.load(u_ptrs, mask=mask_r[:, None] & mask_n[None, :], other=0.0)

        y_tile = tl.dot(z_tile, u_tile, allow_tf32=True)

        if HAS_BIAS:
            b_ptrs = b_ptr + offs_n
            b_tile = tl.load(b_ptrs, mask=mask_n, other=0.0)
            y_tile += b_tile[None, :]

        y_ptrs = y_ptr + offs_m[:, None] * stride_ym + offs_n[None, :] * stride_yn
        tl.store(y_ptrs, y_tile, mask=mask_m[:, None] & mask_n[None, :])


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 16, "BLOCK_N": 32, "BLOCK_K": 32}, num_stages=2, num_warps=2),
        triton.Config({"BLOCK_M": 16, "BLOCK_N": 64, "BLOCK_K": 32}, num_stages=3, num_warps=4),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 32, "BLOCK_K": 32}, num_stages=2, num_warps=2),
    ],
    key=["M", "N", "K", "R"],
)
@triton.jit
def _fused_lowrank_gated_gemm_2d_kernel(
    x_ptr,
    v_ptr,
    u_ptr,
    alpha_ptr,
    b_ptr,
    y_ptr,
    M,
    N,
    K,
    R,
    stride_xm,
    stride_xk,
    stride_vk,
    stride_vr,
    stride_ur,
    stride_un,
    stride_ym,
    stride_yn,
    HAS_ALPHA: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_R: tl.constexpr,
):
    """
    2D Grid Kernel across (M, N) for small-M (e.g., token decoding steps where M < 16).
    Ensures all 46 SMs of the RTX 4070 remain fully saturated.
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_r = tl.arange(0, BLOCK_R)

    mask_m = offs_m < M
    mask_n = offs_n < N
    mask_r = offs_r < R

    z_acc = tl.zeros((BLOCK_M, BLOCK_R), dtype=tl.float32)
    num_k_blocks = tl.cdiv(K, BLOCK_K)
    for k_idx in range(0, num_k_blocks):
        offs_k = k_idx * BLOCK_K + tl.arange(0, BLOCK_K)
        mask_k = offs_k < K

        x_ptrs = x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk
        v_ptrs = v_ptr + offs_k[:, None] * stride_vk + offs_r[None, :] * stride_vr

        x_tile = tl.load(x_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0)
        v_tile = tl.load(v_ptrs, mask=mask_k[:, None] & mask_r[None, :], other=0.0)

        z_acc += tl.dot(x_tile, v_tile, allow_tf32=True)

    if HAS_ALPHA:
        alpha_ptrs = alpha_ptr + offs_r
        alpha_tile = tl.load(alpha_ptrs, mask=mask_r, other=0.0)
        gate = tl.sigmoid(alpha_tile.to(tl.float32))
        z_acc = z_acc * gate[None, :]

    z_tile = z_acc.to(x_ptr.dtype.element_ty)

    u_ptrs = u_ptr + offs_r[:, None] * stride_ur + offs_n[None, :] * stride_un
    u_tile = tl.load(u_ptrs, mask=mask_r[:, None] & mask_n[None, :], other=0.0)

    y_tile = tl.dot(z_tile, u_tile, allow_tf32=True)

    if HAS_BIAS:
        b_ptrs = b_ptr + offs_n
        b_tile = tl.load(b_ptrs, mask=mask_n, other=0.0)
        y_tile += b_tile[None, :]

    y_ptrs = y_ptr + offs_m[:, None] * stride_ym + offs_n[None, :] * stride_yn
    tl.store(y_ptrs, y_tile, mask=mask_m[:, None] & mask_n[None, :])


def triton_fused_lowrank_gated_gemm(
    x: torch.Tensor,
    v_r: torch.Tensor,
    u_r: torch.Tensor,
    alpha: Optional[torch.Tensor] = None,
    b_drift: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    High-performance Triton wrapper for fused low-rank gated GEMM:
        y = (x @ V_r^T * sigmoid(alpha)) @ U_r^T + b_drift

    Args:
        x: Input tensor of shape (M, K) or (B, Seq, K).
        v_r: Down-projection weight. Shape (r, K) or (K, r).
        u_r: Up-projection weight. Shape (N, r) or (r, N).
        alpha: Optional spectral gating parameter. Shape (r,).
        b_drift: Optional bias vector. Shape (N,).

    Returns:
        Output tensor y of shape (*x.shape[:-1], N) matching input dtype and device.
    """
    orig_shape = x.shape
    x_2d = x.view(-1, orig_shape[-1])
    M, K = x_2d.shape

    if v_r.shape[0] == K:
        V_mat = v_r
    elif v_r.shape[1] == K:
        V_mat = v_r.t().contiguous()
    else:
        raise ValueError(f"Shape mismatch for V_r: expected dimension {K}, got {v_r.shape}")

    R = V_mat.shape[1]

    if u_r.shape[0] == R:
        U_mat = u_r
    elif u_r.shape[1] == R:
        U_mat = u_r.t().contiguous()
    else:
        raise ValueError(f"Shape mismatch for U_r: expected rank {R}, got {u_r.shape}")

    N = U_mat.shape[1]

    if not x.is_cuda or R > 256:
        z = torch.matmul(x_2d, V_mat)
        if alpha is not None:
            gate = torch.sigmoid(alpha.view(1, -1).to(dtype=z.dtype, device=z.device))
            z = z * gate
        y = torch.matmul(z, U_mat)
        if b_drift is not None:
            y = y + b_drift.to(dtype=y.dtype, device=y.device)
        return y.view(*orig_shape[:-1], N)

    y = torch.empty((M, N), device=x.device, dtype=x.dtype)

    BLOCK_R = max(16, triton.next_power_of_2(R))

    has_alpha = alpha is not None
    has_bias = b_drift is not None

    alpha_t = alpha.contiguous() if has_alpha else torch.empty(0, device=x.device, dtype=x.dtype)
    b_t = b_drift.contiguous() if has_bias else torch.empty(0, device=x.device, dtype=x.dtype)
    x_c = x_2d.contiguous()
    V_c = V_mat.contiguous()
    U_c = U_mat.contiguous()

    if M >= 16:
        grid = lambda META: (triton.cdiv(M, META["BLOCK_M"]),)
        _fused_lowrank_gated_gemm_1d_kernel[grid](
            x_c,
            V_c,
            U_c,
            alpha_t,
            b_t,
            y,
            M,
            N,
            K,
            R,
            x_c.stride(0),
            x_c.stride(1),
            V_c.stride(0),
            V_c.stride(1),
            U_c.stride(0),
            U_c.stride(1),
            y.stride(0),
            y.stride(1),
            HAS_ALPHA=has_alpha,
            HAS_BIAS=has_bias,
            BLOCK_R=BLOCK_R,
        )
    else:
        grid = lambda META: (triton.cdiv(M, META["BLOCK_M"]), triton.cdiv(N, META["BLOCK_N"]))
        _fused_lowrank_gated_gemm_2d_kernel[grid](
            x_c,
            V_c,
            U_c,
            alpha_t,
            b_t,
            y,
            M,
            N,
            K,
            R,
            x_c.stride(0),
            x_c.stride(1),
            V_c.stride(0),
            V_c.stride(1),
            U_c.stride(0),
            U_c.stride(1),
            y.stride(0),
            y.stride(1),
            HAS_ALPHA=has_alpha,
            HAS_BIAS=has_bias,
            BLOCK_R=BLOCK_R,
        )

    return y.view(*orig_shape[:-1], N)


def fused_lowrank_gemm(
    x: torch.Tensor,
    w: torch.Tensor,
    u_r: Optional[torch.Tensor] = None,
    alpha: Optional[torch.Tensor] = None,
    b_drift: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Fused low-rank GEMM supporting both 2-argument packed weight (w = [V; U]) and explicit (x, v_r, u_r).

    Args:
        x: Input activations of shape (*, K).
        w: Either packed weight matrix [V; U] of shape (2*K, R) or down-projection V_r of shape (K, R).
        u_r: Optional up-projection matrix U_r. If None and w is packed, unpacked from w.
        alpha: Optional spectral gating tensor.
        b_drift: Optional bias drift tensor.

    Returns:
        Projected output tensor.
    """
    if u_r is None:
        K = x.shape[-1]
        if w.shape[0] == 2 * K:
            v_mat = w[:K, :]
            u_mat = w[K:, :]
            return triton_fused_lowrank_gated_gemm(x, v_mat, u_mat, alpha=alpha, b_drift=b_drift)
        elif w.shape[0] == K:
            return torch.matmul(x, w)
        elif w.shape[1] == K:
            return torch.matmul(x, w.t())
        else:
            half = w.shape[0] // 2
            v_mat = w[:half, :]
            u_mat = w[half:, :]
            return triton_fused_lowrank_gated_gemm(x, v_mat, u_mat, alpha=alpha, b_drift=b_drift)
    return triton_fused_lowrank_gated_gemm(x, w, u_r, alpha=alpha, b_drift=b_drift)

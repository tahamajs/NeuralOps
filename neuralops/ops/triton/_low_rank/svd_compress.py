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

    @triton.autotune(
        configs=[
            triton.Config({"BLOCK_M": 32, "BLOCK_N": 32, "BLOCK_K": 16}, num_stages=2, num_warps=2),
            triton.Config({"BLOCK_M": 64, "BLOCK_N": 32, "BLOCK_K": 32}, num_stages=3, num_warps=4),
            triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32}, num_stages=3, num_warps=4),
        ],
        key=["M", "N", "K"],
    )
    @triton.jit
    def _direct_svd_reconstruct_kernel(
        U_ptr,
        V_ptr,
        Out_ptr,
        M,
        N,
        K,
        stride_um,
        stride_uk,
        stride_vk,
        stride_vn,
        stride_om,
        stride_on,
        scale,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, BLOCK_K)

        u_ptrs = U_ptr + (offs_m[:, None] * stride_um + offs_k[None, :] * stride_uk)
        v_ptrs = V_ptr + (offs_k[:, None] * stride_vk + offs_n[None, :] * stride_vn)

        accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        for k in range(tl.cdiv(K, BLOCK_K)):
            k_mask = (k * BLOCK_K + offs_k) < K
            u = tl.load(u_ptrs, mask=(offs_m[:, None] < M) & k_mask[None, :], other=0.0)
            v = tl.load(v_ptrs, mask=k_mask[:, None] & (offs_n[None, :] < N), other=0.0)
            accumulator += tl.dot(u, v)

            u_ptrs += BLOCK_K * stride_uk
            v_ptrs += BLOCK_K * stride_vk

        if scale != 1.0:
            accumulator = accumulator * scale

        out_ptrs = Out_ptr + (offs_m[:, None] * stride_om + offs_n[None, :] * stride_on)
        out_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
        tl.store(out_ptrs, accumulator, mask=out_mask)

else:
    _direct_svd_reconstruct_kernel = None


def direct_svd_compress(
    W: torch.Tensor,
    rank: int = 32,
    max_error: float = 0.35,
) -> torch.Tensor:
    """Direct Truncated Singular Value Decomposition (SVD) projection.

    Decomposes target 2D projection matrix W into low-rank factor pair (U_r, V_r)
    with rank truncation, symmetric singular energy allocation, and energy scaling,
    evaluating reconstruction with Triton kernel.

    Args:
        W: 2D weight tensor of shape (M, N)
        rank: Target compression rank
        max_error: Error tolerance threshold (reference compatibility)

    Returns:
        Reconstructed weight tensor matching W.shape and W.dtype
    """
    if W.ndim != 2:
        raise ValueError(f"direct_svd_compress requires a 2D weight matrix, got shape {W.shape}")

    M, N = W.shape
    r = max(1, min(rank, min(M, N) - 1)) if min(M, N) > 1 else 1

    W_fp32 = W.float()
    norm_w = torch.norm(W_fp32, p="fro").item()

    # Step 1: Truncated SVD matching reference DirectSVDCompressor
    U, S, Vt = torch.linalg.svd(W_fp32, full_matrices=False)
    sqrt_S = torch.sqrt(torch.clamp(S[:r], min=1e-12))
    U_r = (U[:, :r] * sqrt_S.unsqueeze(0)).contiguous()
    V_r = (sqrt_S.unsqueeze(1) * Vt[:r, :]).contiguous()

    # Energy normalization scale factor (matching DirectSVDCompressor energy_scale^2)
    W_recon_pre = torch.matmul(U_r, V_r)
    recon_norm = torch.norm(W_recon_pre, p="fro").item()
    scale = 1.0
    if recon_norm > 1e-12 and norm_w > 1e-12:
        scale = norm_w / recon_norm

    # Step 2: Dispatch to Triton or PyTorch fallback
    if (
        HAS_TRITON
        and W.is_cuda
        and check_triton_suitability(W, "direct_svd")
        and _direct_svd_reconstruct_kernel is not None
    ):
        out = torch.empty((M, N), device=W.device, dtype=torch.float32)
        grid = lambda META: (
            triton.cdiv(M, META["BLOCK_M"]),
            triton.cdiv(N, META["BLOCK_N"]),
        )
        _direct_svd_reconstruct_kernel[grid](
            U_r,
            V_r,
            out,
            M,
            N,
            r,
            U_r.stride(0),
            U_r.stride(1),
            V_r.stride(0),
            V_r.stride(1),
            out.stride(0),
            out.stride(1),
            float(scale),
        )
        return out.to(dtype=W.dtype)

    logger.warning(
        "direct_svd_compress: Triton unavailable or unsuitable on current device/shape, using PyTorch fallback."
    )
    return (W_recon_pre * scale).to(dtype=W.dtype)


def direct_svd_compress_pytorch(
    W: torch.Tensor,
    rank: int = 32,
    max_error: float = 0.35,
) -> torch.Tensor:
    """Pure PyTorch reference implementation of direct SVD compression."""
    if W.ndim != 2:
        raise ValueError(f"direct_svd_compress_pytorch requires a 2D weight matrix, got shape {W.shape}")

    M, N = W.shape
    r = max(1, min(rank, min(M, N) - 1)) if min(M, N) > 1 else 1

    W_fp32 = W.float()
    norm_w = torch.norm(W_fp32, p="fro").item()

    U, S, Vt = torch.linalg.svd(W_fp32, full_matrices=False)
    sqrt_S = torch.sqrt(torch.clamp(S[:r], min=1e-12))
    U_r = (U[:, :r] * sqrt_S.unsqueeze(0)).contiguous()
    V_r = (sqrt_S.unsqueeze(1) * Vt[:r, :]).contiguous()

    W_recon_pre = torch.matmul(U_r, V_r)
    recon_norm = torch.norm(W_recon_pre, p="fro").item()
    scale = 1.0
    if recon_norm > 1e-12 and norm_w > 1e-12:
        scale = norm_w / recon_norm

    return (W_recon_pre * scale).to(dtype=W.dtype)


if HAS_TRITON and tl is not None:

    @triton.autotune(
        configs=[
            triton.Config({"BLOCK_M": 32, "BLOCK_N": 32, "BLOCK_K": 16}, num_stages=2, num_warps=2),
            triton.Config({"BLOCK_M": 64, "BLOCK_N": 32, "BLOCK_K": 32}, num_stages=3, num_warps=4),
            triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32}, num_stages=3, num_warps=4),
        ],
        key=["M", "N", "K"],
    )
    @triton.jit
    def _asvd_reconstruct_kernel(
        U_ptr,
        V_ptr,
        Row_gain_ptr,
        Out_ptr,
        M,
        N,
        K,
        stride_um,
        stride_uk,
        stride_vk,
        stride_vn,
        stride_om,
        stride_on,
        HAS_ROW_GAIN: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, BLOCK_K)

        u_ptrs = U_ptr + (offs_m[:, None] * stride_um + offs_k[None, :] * stride_uk)
        v_ptrs = V_ptr + (offs_k[:, None] * stride_vk + offs_n[None, :] * stride_vn)

        accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        for k in range(tl.cdiv(K, BLOCK_K)):
            k_mask = (k * BLOCK_K + offs_k) < K
            u = tl.load(u_ptrs, mask=(offs_m[:, None] < M) & k_mask[None, :], other=0.0)
            v = tl.load(v_ptrs, mask=k_mask[:, None] & (offs_n[None, :] < N), other=0.0)
            accumulator += tl.dot(u, v)

            u_ptrs += BLOCK_K * stride_uk
            v_ptrs += BLOCK_K * stride_vk

        if HAS_ROW_GAIN:
            gain_ptrs = Row_gain_ptr + offs_m
            gain = tl.load(gain_ptrs, mask=offs_m < M, other=1.0)
            accumulator = accumulator * gain[:, None]

        out_ptrs = Out_ptr + (offs_m[:, None] * stride_om + offs_n[None, :] * stride_on)
        out_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
        tl.store(out_ptrs, accumulator, mask=out_mask)

else:
    _asvd_reconstruct_kernel = None


def asvd_compress(
    W: torch.Tensor,
    cov: Optional[torch.Tensor] = None,
    rank: int = 32,
    max_error: float = 0.35,
) -> torch.Tensor:
    """Activation-Aware Singular Value Decomposition (ASVD) projection.

    Scales weight matrix W by empirical input activation covariance factors,
    computes truncated SVD, applies inverse covariance whitening pullback to V_r,
    and reconstructs with row-norm rebalancing via Triton kernel.

    Args:
        W: 2D weight tensor of shape (M, N)
        cov: Optional activation covariance matrix of shape (N, N)
        rank: Target compression rank
        max_error: Error tolerance threshold (reference compatibility)

    Returns:
        Reconstructed weight tensor matching W.shape and W.dtype
    """
    if W.ndim != 2:
        raise ValueError(f"asvd_compress requires a 2D weight matrix, got shape {W.shape}")

    M, N = W.shape
    r = max(1, min(rank, min(M, N) - 1)) if min(M, N) > 1 else 1
    W_fp32 = W.float()

    cov_sqrt = cov_inv_sqrt = None
    if cov is not None and cov.shape == (N, N):
        try:
            cov_fp32 = cov.to(W.device, torch.float32)
            cov_sym = 0.5 * (cov_fp32 + cov_fp32.T)
            trace_val = torch.trace(cov_sym).clamp(min=1e-8)
            damping = 0.05 * (trace_val / float(N))
            cov_damped = cov_sym + damping * torch.eye(N, device=cov_sym.device, dtype=torch.float32)
            eigvals, eigvecs = torch.linalg.eigh(cov_damped)
            eps = 1e-5 * torch.max(eigvals).clamp(min=1e-12) + 1e-7
            eigvals_reg = eigvals.clamp(min=eps)
            cov_sqrt = eigvecs @ torch.diag(torch.sqrt(eigvals_reg)) @ eigvecs.T
            cov_inv_sqrt = eigvecs @ torch.diag(1.0 / torch.sqrt(eigvals_reg)) @ eigvecs.T
        except Exception as e:
            logger.debug("asvd_compress: covariance sqrt failed: %s", e)
            cov_sqrt, cov_inv_sqrt = None, None

    W_weighted = torch.matmul(W_fp32, cov_sqrt) if cov_sqrt is not None else W_fp32
    U, S, Vt = torch.linalg.svd(W_weighted, full_matrices=False)
    selected_r = min(r, len(S))
    sqrt_S = torch.sqrt(torch.clamp(S[:selected_r], min=1e-12))
    U_r = (U[:, :selected_r] * sqrt_S.unsqueeze(0)).contiguous()
    V_r = (sqrt_S.unsqueeze(1) * Vt[:selected_r, :]).contiguous()
    if cov_inv_sqrt is not None:
        V_r = torch.matmul(V_r, cov_inv_sqrt).contiguous()

    # Row-wise norm rebalancing
    W_recon_pre = torch.matmul(U_r, V_r)
    orig_row_norms = torch.norm(W_fp32, dim=-1, keepdim=True)
    comp_row_norms = torch.norm(W_recon_pre, dim=-1, keepdim=True).clamp_min(1e-6)
    row_gain = (orig_row_norms / comp_row_norms).clamp(0.2, 5.0).squeeze(-1).contiguous()

    if HAS_TRITON and W.is_cuda and check_triton_suitability(W, "asvd") and _asvd_reconstruct_kernel is not None:
        out = torch.empty((M, N), device=W.device, dtype=torch.float32)
        grid = lambda META: (
            triton.cdiv(M, META["BLOCK_M"]),
            triton.cdiv(N, META["BLOCK_N"]),
        )
        _asvd_reconstruct_kernel[grid](
            U_r,
            V_r,
            row_gain,
            out,
            M,
            N,
            selected_r,
            U_r.stride(0),
            U_r.stride(1),
            V_r.stride(0),
            V_r.stride(1),
            out.stride(0),
            out.stride(1),
            HAS_ROW_GAIN=True,
        )
        return out.to(dtype=W.dtype)

    logger.warning("asvd_compress: Triton unavailable or unsuitable, using PyTorch fallback.")
    return (W_recon_pre * row_gain.unsqueeze(-1)).to(dtype=W.dtype)


def asvd_compress_pytorch(
    W: torch.Tensor,
    cov: Optional[torch.Tensor] = None,
    rank: int = 32,
    max_error: float = 0.35,
) -> torch.Tensor:
    """Pure PyTorch reference implementation of ASVD compression."""
    if W.ndim != 2:
        raise ValueError(f"asvd_compress_pytorch requires a 2D weight matrix, got shape {W.shape}")

    M, N = W.shape
    r = max(1, min(rank, min(M, N) - 1)) if min(M, N) > 1 else 1
    W_fp32 = W.float()

    cov_sqrt = cov_inv_sqrt = None
    if cov is not None and cov.shape == (N, N):
        try:
            cov_fp32 = cov.to(W.device, torch.float32)
            cov_sym = 0.5 * (cov_fp32 + cov_fp32.T)
            trace_val = torch.trace(cov_sym).clamp(min=1e-8)
            damping = 0.05 * (trace_val / float(N))
            cov_damped = cov_sym + damping * torch.eye(N, device=cov_sym.device, dtype=torch.float32)
            eigvals, eigvecs = torch.linalg.eigh(cov_damped)
            eps = 1e-5 * torch.max(eigvals).clamp(min=1e-12) + 1e-7
            eigvals_reg = eigvals.clamp(min=eps)
            cov_sqrt = eigvecs @ torch.diag(torch.sqrt(eigvals_reg)) @ eigvecs.T
            cov_inv_sqrt = eigvecs @ torch.diag(1.0 / torch.sqrt(eigvals_reg)) @ eigvecs.T
        except Exception as e:
            logger.debug("asvd_compress_pytorch: covariance sqrt failed: %s", e)
            cov_sqrt, cov_inv_sqrt = None, None

    W_weighted = torch.matmul(W_fp32, cov_sqrt) if cov_sqrt is not None else W_fp32
    U, S, Vt = torch.linalg.svd(W_weighted, full_matrices=False)
    selected_r = min(r, len(S))
    sqrt_S = torch.sqrt(torch.clamp(S[:selected_r], min=1e-12))
    U_r = (U[:, :selected_r] * sqrt_S.unsqueeze(0)).contiguous()
    V_r = (sqrt_S.unsqueeze(1) * Vt[:selected_r, :]).contiguous()
    if cov_inv_sqrt is not None:
        V_r = torch.matmul(V_r, cov_inv_sqrt).contiguous()

    # Row-wise norm rebalancing
    W_recon_pre = torch.matmul(U_r, V_r)
    orig_row_norms = torch.norm(W_fp32, dim=-1, keepdim=True)
    comp_row_norms = torch.norm(W_recon_pre, dim=-1, keepdim=True).clamp_min(1e-6)
    row_gain = (orig_row_norms / comp_row_norms).clamp(0.2, 5.0).squeeze(-1).contiguous()
    return (W_recon_pre * row_gain.unsqueeze(-1)).to(dtype=W.dtype)


if HAS_TRITON and tl is not None:

    @triton.autotune(
        configs=[
            triton.Config({"BLOCK_M": 32, "BLOCK_N": 32, "BLOCK_K": 16}, num_stages=2, num_warps=2),
            triton.Config({"BLOCK_M": 64, "BLOCK_N": 32, "BLOCK_K": 32}, num_stages=3, num_warps=4),
            triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32}, num_stages=3, num_warps=4),
        ],
        key=["M", "N", "K"],
    )
    @triton.jit
    def _fwsvd_reconstruct_kernel(
        U_ptr,
        V_ptr,
        Out_ptr,
        M,
        N,
        K,
        stride_um,
        stride_uk,
        stride_vk,
        stride_vn,
        stride_om,
        stride_on,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, BLOCK_K)

        u_ptrs = U_ptr + (offs_m[:, None] * stride_um + offs_k[None, :] * stride_uk)
        v_ptrs = V_ptr + (offs_k[:, None] * stride_vk + offs_n[None, :] * stride_vn)

        accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        for k in range(tl.cdiv(K, BLOCK_K)):
            k_mask = (k * BLOCK_K + offs_k) < K
            u = tl.load(u_ptrs, mask=(offs_m[:, None] < M) & k_mask[None, :], other=0.0)
            v = tl.load(v_ptrs, mask=k_mask[:, None] & (offs_n[None, :] < N), other=0.0)
            accumulator += tl.dot(u, v)

            u_ptrs += BLOCK_K * stride_uk
            v_ptrs += BLOCK_K * stride_vk

        out_ptrs = Out_ptr + (offs_m[:, None] * stride_om + offs_n[None, :] * stride_on)
        out_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
        tl.store(out_ptrs, accumulator, mask=out_mask)

else:
    _fwsvd_reconstruct_kernel = None


def fwsvd_compress(
    W: torch.Tensor,
    fisher_diag: Optional[torch.Tensor] = None,
    fisher_matrix: Optional[torch.Tensor] = None,
    rank: int = 32,
    damping: float = 1e-4,
    max_error: float = 1.0,
) -> torch.Tensor:
    """Fisher-Weighted Singular Value Decomposition (FWSVD) projection.

    Preconditions weight matrix W by empirical Fisher Information curvature
    before SVD factorization, applies inverse Fisher transformation to V_r,
    and reconstructs low-rank approximation via Triton kernel.

    Args:
        W: 2D weight tensor of shape (M, N)
        fisher_diag: Diagonal empirical Fisher vector of shape (N,)
        fisher_matrix: Full empirical Fisher matrix of shape (N, N)
        rank: Target compression rank
        damping: Tikhonov damping coefficient
        max_error: Error tolerance threshold (reference compatibility)

    Returns:
        Reconstructed weight tensor matching W.shape and W.dtype
    """
    if W.ndim != 2:
        raise ValueError(f"fwsvd_compress requires a 2D weight matrix, got shape {W.shape}")

    M, N = W.shape
    r = max(1, min(rank, min(M, N) - 1)) if min(M, N) > 1 else 1
    W_fp32 = W.float()

    F_mat = fisher_matrix
    if F_mat is None and fisher_diag is not None and fisher_diag.ndim == 2:
        F_mat = fisher_diag

    if F_mat is not None and F_mat.shape == (N, N):
        F_dev = F_mat.to(device=W.device, dtype=torch.float32)
        F_sym = 0.5 * (F_dev + F_dev.t())
        tr_F = torch.trace(F_sym).item()
        damp_val = max(1e-7, float(damping * tr_F / max(1, N)))
        F_damped = F_sym + damp_val * torch.eye(N, device=F_dev.device, dtype=torch.float32)

        try:
            eigvals, eigvecs = torch.linalg.eigh(F_damped)
            eigvals_clamped = torch.clamp(eigvals, min=damp_val)
            sqrt_eig = torch.sqrt(eigvals_clamped)
            inv_sqrt_eig = 1.0 / sqrt_eig
            F_sqrt = eigvecs @ torch.diag(sqrt_eig) @ eigvecs.t()
            F_inv_sqrt = eigvecs @ torch.diag(inv_sqrt_eig) @ eigvecs.t()

            W_weighted = torch.matmul(W_fp32, F_sqrt)
            U, S, Vt = torch.linalg.svd(W_weighted, full_matrices=False)
            sqrt_S = torch.sqrt(torch.clamp(S[:r], min=1e-12))
            U_r = (U[:, :r] * sqrt_S.unsqueeze(0)).contiguous()
            V_r = torch.matmul(sqrt_S.unsqueeze(1) * Vt[:r, :], F_inv_sqrt).contiguous()
        except Exception as e:
            logger.debug("fwsvd_compress: full Fisher whitening failed (%s), fallback unweighted", e)
            U, S, Vt = torch.linalg.svd(W_fp32, full_matrices=False)
            sqrt_S = torch.sqrt(torch.clamp(S[:r], min=1e-12))
            U_r = (U[:, :r] * sqrt_S.unsqueeze(0)).contiguous()
            V_r = (sqrt_S.unsqueeze(1) * Vt[:r, :]).contiguous()
    elif fisher_diag is not None and fisher_diag.shape == (N,):
        f_clamped = fisher_diag.to(device=W.device, dtype=torch.float32).clamp(min=1e-8)
        sqrt_f = torch.sqrt(f_clamped)
        inv_sqrt_f = 1.0 / sqrt_f
        W_weighted = W_fp32 * sqrt_f.unsqueeze(0)
        U, S, Vt = torch.linalg.svd(W_weighted, full_matrices=False)
        sqrt_S = torch.sqrt(torch.clamp(S[:r], min=1e-12))
        U_r = (U[:, :r] * sqrt_S.unsqueeze(0)).contiguous()
        V_r = ((sqrt_S.unsqueeze(1) * Vt[:r, :]) * inv_sqrt_f.unsqueeze(0)).contiguous()
    else:
        U, S, Vt = torch.linalg.svd(W_fp32, full_matrices=False)
        sqrt_S = torch.sqrt(torch.clamp(S[:r], min=1e-12))
        U_r = (U[:, :r] * sqrt_S.unsqueeze(0)).contiguous()
        V_r = (sqrt_S.unsqueeze(1) * Vt[:r, :]).contiguous()

    if HAS_TRITON and W.is_cuda and check_triton_suitability(W, "fwsvd") and _fwsvd_reconstruct_kernel is not None:
        out = torch.empty((M, N), device=W.device, dtype=torch.float32)
        grid = lambda META: (
            triton.cdiv(M, META["BLOCK_M"]),
            triton.cdiv(N, META["BLOCK_N"]),
        )
        _fwsvd_reconstruct_kernel[grid](
            U_r,
            V_r,
            out,
            M,
            N,
            r,
            U_r.stride(0),
            U_r.stride(1),
            V_r.stride(0),
            V_r.stride(1),
            out.stride(0),
            out.stride(1),
        )
        return out.to(dtype=W.dtype)

    logger.warning("fwsvd_compress: Triton unavailable or unsuitable, using PyTorch fallback.")
    return torch.matmul(U_r, V_r).to(dtype=W.dtype)


def fwsvd_compress_pytorch(
    W: torch.Tensor,
    fisher_diag: Optional[torch.Tensor] = None,
    fisher_matrix: Optional[torch.Tensor] = None,
    rank: int = 32,
    damping: float = 1e-4,
    max_error: float = 1.0,
) -> torch.Tensor:
    """Pure PyTorch reference implementation of Fisher-Weighted SVD compression."""
    if W.ndim != 2:
        raise ValueError(f"fwsvd_compress_pytorch requires a 2D weight matrix, got shape {W.shape}")

    M, N = W.shape
    r = max(1, min(rank, min(M, N) - 1)) if min(M, N) > 1 else 1
    W_fp32 = W.float()

    F_mat = fisher_matrix
    if F_mat is None and fisher_diag is not None and fisher_diag.ndim == 2:
        F_mat = fisher_diag

    if F_mat is not None and F_mat.shape == (N, N):
        F_dev = F_mat.to(device=W.device, dtype=torch.float32)
        F_sym = 0.5 * (F_dev + F_dev.t())
        tr_F = torch.trace(F_sym).item()
        damp_val = max(1e-7, float(damping * tr_F / max(1, N)))
        F_damped = F_sym + damp_val * torch.eye(N, device=F_dev.device, dtype=torch.float32)

        try:
            eigvals, eigvecs = torch.linalg.eigh(F_damped)
            eigvals_clamped = torch.clamp(eigvals, min=damp_val)
            sqrt_eig = torch.sqrt(eigvals_clamped)
            inv_sqrt_eig = 1.0 / sqrt_eig
            F_sqrt = eigvecs @ torch.diag(sqrt_eig) @ eigvecs.t()
            F_inv_sqrt = eigvecs @ torch.diag(inv_sqrt_eig) @ eigvecs.t()

            W_weighted = torch.matmul(W_fp32, F_sqrt)
            U, S, Vt = torch.linalg.svd(W_weighted, full_matrices=False)
            sqrt_S = torch.sqrt(torch.clamp(S[:r], min=1e-12))
            U_r = (U[:, :r] * sqrt_S.unsqueeze(0)).contiguous()
            V_r = torch.matmul(sqrt_S.unsqueeze(1) * Vt[:r, :], F_inv_sqrt).contiguous()
        except Exception:
            U, S, Vt = torch.linalg.svd(W_fp32, full_matrices=False)
            sqrt_S = torch.sqrt(torch.clamp(S[:r], min=1e-12))
            U_r = (U[:, :r] * sqrt_S.unsqueeze(0)).contiguous()
            V_r = (sqrt_S.unsqueeze(1) * Vt[:r, :]).contiguous()
    elif fisher_diag is not None and fisher_diag.shape == (N,):
        f_clamped = fisher_diag.to(device=W.device, dtype=torch.float32).clamp(min=1e-8)
        sqrt_f = torch.sqrt(f_clamped)
        inv_sqrt_f = 1.0 / sqrt_f
        W_weighted = W_fp32 * sqrt_f.unsqueeze(0)
        U, S, Vt = torch.linalg.svd(W_weighted, full_matrices=False)
        sqrt_S = torch.sqrt(torch.clamp(S[:r], min=1e-12))
        U_r = (U[:, :r] * sqrt_S.unsqueeze(0)).contiguous()
        V_r = ((sqrt_S.unsqueeze(1) * Vt[:r, :]) * inv_sqrt_f.unsqueeze(0)).contiguous()
    else:
        U, S, Vt = torch.linalg.svd(W_fp32, full_matrices=False)
        sqrt_S = torch.sqrt(torch.clamp(S[:r], min=1e-12))
        U_r = (U[:, :r] * sqrt_S.unsqueeze(0)).contiguous()
        V_r = (sqrt_S.unsqueeze(1) * Vt[:r, :]).contiguous()

    return torch.matmul(U_r, V_r).to(dtype=W.dtype)


if HAS_TRITON and tl is not None:

    @triton.autotune(
        configs=[
            triton.Config({"BLOCK_M": 32, "BLOCK_N": 32, "BLOCK_K": 16}, num_stages=2, num_warps=2),
            triton.Config({"BLOCK_M": 64, "BLOCK_N": 32, "BLOCK_K": 32}, num_stages=3, num_warps=4),
            triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32}, num_stages=3, num_warps=4),
        ],
        key=["M", "N", "K"],
    )
    @triton.jit
    def _hnc_svd_reconstruct_kernel(
        A_ptr,
        B_ptr,
        Out_ptr,
        M,
        N,
        K,
        stride_am,
        stride_ak,
        stride_bk,
        stride_bn,
        stride_om,
        stride_on,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, BLOCK_K)

        a_ptrs = A_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
        b_ptrs = B_ptr + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)

        accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        for k in range(tl.cdiv(K, BLOCK_K)):
            k_mask = (k * BLOCK_K + offs_k) < K
            a = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & k_mask[None, :], other=0.0)
            b = tl.load(b_ptrs, mask=k_mask[:, None] & (offs_n[None, :] < N), other=0.0)
            accumulator += tl.dot(a, b)

            a_ptrs += BLOCK_K * stride_ak
            b_ptrs += BLOCK_K * stride_bk

        out_ptrs = Out_ptr + (offs_m[:, None] * stride_om + offs_n[None, :] * stride_on)
        out_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
        tl.store(out_ptrs, accumulator, mask=out_mask)

else:
    _hnc_svd_reconstruct_kernel = None


def hnc_svd_compress(
    W: torch.Tensor,
    cov: Optional[torch.Tensor] = None,
    rank: int = 32,
    max_error: float = 0.35,
) -> torch.Tensor:
    """Hierarchical Newton-Correction SVD (HNC-SVD) projection.

    Executes Cholesky-whitened SVD truncation on weight matrix W preconditioned
    by input activation covariance H_x, solving the triangular pullback for factor B_r
    and evaluating low-rank reconstruction via Triton kernel.

    Args:
        W: 2D weight tensor of shape (M, N)
        cov: Optional empirical activation covariance H_x of shape (N, N)
        rank: Target compression rank
        max_error: Error tolerance threshold (reference compatibility)

    Returns:
        Reconstructed weight tensor matching W.shape and W.dtype
    """
    if W.ndim != 2:
        raise ValueError(f"hnc_svd_compress requires a 2D weight matrix, got shape {W.shape}")

    M, N = W.shape
    r = max(1, min(rank, min(M, N) - 1)) if min(M, N) > 1 else 1
    W_fp32 = W.float()

    if cov is not None and cov.shape == (N, N):
        Hx = cov.to(device=W.device, dtype=torch.float32)
        try:
            reg = 1e-3 * (torch.trace(Hx) / float(N)).item() + 1e-6
            Sx = Hx + reg * torch.eye(N, device=W.device, dtype=torch.float32)
            L = torch.linalg.cholesky(Sx)
            W_whitened = torch.matmul(W_fp32, L)
            U, S, Vh = torch.linalg.svd(W_whitened, full_matrices=False)
            U_r = U[:, :r] * torch.sqrt(S[:r]).unsqueeze(0)
            V_unwhitened = torch.linalg.solve_triangular(
                L.t(), (torch.sqrt(S[:r]).unsqueeze(1) * Vh[:r, :]).t(), upper=True
            ).t()
            A_r = U_r.contiguous()
            B_r = V_unwhitened.contiguous()
        except Exception as e:
            logger.debug("hnc_svd_compress: Cholesky whitening failed (%s), fallback unwhitened", e)
            U, S, Vh = torch.linalg.svd(W_fp32, full_matrices=False)
            A_r = (U[:, :r] * torch.sqrt(S[:r]).unsqueeze(0)).contiguous()
            B_r = (torch.sqrt(S[:r]).unsqueeze(1) * Vh[:r, :]).contiguous()
    else:
        U, S, Vh = torch.linalg.svd(W_fp32, full_matrices=False)
        A_r = (U[:, :r] * torch.sqrt(S[:r]).unsqueeze(0)).contiguous()
        B_r = (torch.sqrt(S[:r]).unsqueeze(1) * Vh[:r, :]).contiguous()

    if HAS_TRITON and W.is_cuda and check_triton_suitability(W, "hnc_svd") and _hnc_svd_reconstruct_kernel is not None:
        out = torch.empty((M, N), device=W.device, dtype=torch.float32)
        grid = lambda META: (
            triton.cdiv(M, META["BLOCK_M"]),
            triton.cdiv(N, META["BLOCK_N"]),
        )
        _hnc_svd_reconstruct_kernel[grid](
            A_r,
            B_r,
            out,
            M,
            N,
            r,
            A_r.stride(0),
            A_r.stride(1),
            B_r.stride(0),
            B_r.stride(1),
            out.stride(0),
            out.stride(1),
        )
        return out.to(dtype=W.dtype)

    logger.warning("hnc_svd_compress: Triton unavailable or unsuitable, using PyTorch fallback.")
    return torch.matmul(A_r, B_r).to(dtype=W.dtype)


def hnc_svd_compress_pytorch(
    W: torch.Tensor,
    cov: Optional[torch.Tensor] = None,
    rank: int = 32,
    max_error: float = 0.35,
) -> torch.Tensor:
    """Pure PyTorch reference implementation of HNC-SVD compression."""
    if W.ndim != 2:
        raise ValueError(f"hnc_svd_compress_pytorch requires a 2D weight matrix, got shape {W.shape}")

    M, N = W.shape
    r = max(1, min(rank, min(M, N) - 1)) if min(M, N) > 1 else 1
    W_fp32 = W.float()

    if cov is not None and cov.shape == (N, N):
        Hx = cov.to(device=W.device, dtype=torch.float32)
        try:
            reg = 1e-3 * (torch.trace(Hx) / float(N)).item() + 1e-6
            Sx = Hx + reg * torch.eye(N, device=W.device, dtype=torch.float32)
            L = torch.linalg.cholesky(Sx)
            W_whitened = torch.matmul(W_fp32, L)
            U, S, Vh = torch.linalg.svd(W_whitened, full_matrices=False)
            U_r = U[:, :r] * torch.sqrt(S[:r]).unsqueeze(0)
            V_unwhitened = torch.linalg.solve_triangular(
                L.t(), (torch.sqrt(S[:r]).unsqueeze(1) * Vh[:r, :]).t(), upper=True
            ).t()
            A_r = U_r.contiguous()
            B_r = V_unwhitened.contiguous()
        except Exception:
            U, S, Vh = torch.linalg.svd(W_fp32, full_matrices=False)
            A_r = (U[:, :r] * torch.sqrt(S[:r]).unsqueeze(0)).contiguous()
            B_r = (torch.sqrt(S[:r]).unsqueeze(1) * Vh[:r, :]).contiguous()
    else:
        U, S, Vh = torch.linalg.svd(W_fp32, full_matrices=False)
        A_r = (U[:, :r] * torch.sqrt(S[:r]).unsqueeze(0)).contiguous()
        B_r = (torch.sqrt(S[:r]).unsqueeze(1) * Vh[:r, :]).contiguous()

    return torch.matmul(A_r, B_r).to(dtype=W.dtype)

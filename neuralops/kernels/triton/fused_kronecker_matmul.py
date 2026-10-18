"""
Fused Kronecker Matrix Multiplication

Fuses dual batched matrix multiplications and axis transpositions for Kronecker factorized linear layers directly in SRAM.

Author: Taha Majlesi
"""

from __future__ import annotations

import math
from typing import Any, Optional, Tuple

import torch
import triton
import triton.language as tl


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

    # Load factor matrices A [M1, N1] and B [M2, N2] into SRAM once per CTA
    offs_m1 = tl.arange(0, M1)
    offs_n1 = tl.arange(0, N1)
    offs_m2 = tl.arange(0, M2)
    offs_n2 = tl.arange(0, N2)

    a_tile = tl.load(a_ptr + offs_m1[:, None] * stride_a_m + offs_n1[None, :] * stride_a_n)
    b_tile = tl.load(b_ptr + offs_m2[:, None] * stride_b_m + offs_n2[None, :] * stride_b_n)

    # For each row in the BLOCK_M tile:
    for m_idx in range(0, BLOCK_M):
        curr_m = pid_m * BLOCK_M + m_idx
        if curr_m < M:
            # Load X_m of shape (N1, N2) or (N2, N1)
            # In standard Kronecker: X_m has shape (N1, N2) where flat index = n1 * N2 + n2
            flat_offs_k = offs_n1[:, None] * N2 + offs_n2[None, :]
            x_ptrs = x_ptr + curr_m * stride_xm + flat_offs_k * stride_xk
            x_slice = tl.load(x_ptrs)  # (N1, N2)

            # 1. Compute T = x_slice @ B^T -> shape (N1, M2)
            # x_slice is (N1, N2), b_tile is (M2, N2) -> dot(x_slice, trans(b_tile))
            t = tl.dot(x_slice.to(tl.float32), tl.trans(b_tile).to(tl.float32), allow_tf32=True)

            # 2. Compute Y_slice = A @ t -> shape (M1, M2)
            # a_tile is (M1, N1), t is (N1, M2) -> dot(a_tile, t)
            y_slice = tl.dot(a_tile.to(tl.float32), t.to(tl.float32), allow_tf32=True)

            # Flatten Y_slice (M1, M2) -> flat_offs_out = m1 * M2 + m2
            flat_offs_out = offs_m1[:, None] * M2 + offs_m2[None, :]
            y_ptrs = y_ptr + curr_m * stride_ym + flat_offs_out * stride_yn

            out_val = y_slice
            if HAS_BIAS:
                b_val = tl.load(bias_ptr + flat_offs_out)
                out_val += b_val.to(tl.float32)

            tl.store(y_ptrs, out_val.to(x_ptr.dtype.element_ty))


def triton_fused_kronecker_matmul(
    x: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    High-performance Triton fused Kronecker GEMM:
        Y = X @ (A (x) B)^T + bias

    Args:
        x: Input tensor of shape (..., N1 * N2).
        A: Factor A of shape (M1, N1).
        B: Factor B of shape (M2, N2).
        bias: Optional bias of shape (M1 * M2,).

    Returns:
        Tensor of shape (..., M1 * M2).
    """
    orig_shape = x.shape
    x_2d = x.view(-1, orig_shape[-1])
    M, K = x_2d.shape

    M1, N1 = A.shape
    M2, N2 = B.shape
    out_features = M1 * M2

    assert K == N1 * N2, f"Input feature mismatch: {K} != {N1} * {N2}"

    if not x.is_cuda:
        # PyTorch reference
        X_reshaped = x_2d.view(M, N1, N2)
        # T = X @ B.T -> (M, N1, M2)
        T = torch.matmul(X_reshaped, B.t())
        # Y = A @ T -> (M, M1, M2)
        Y = torch.matmul(A, T.transpose(1, 2)).transpose(1, 2)
        Y_flat = Y.reshape(M, out_features)
        if bias is not None:
            Y_flat = Y_flat + bias.to(dtype=Y_flat.dtype, device=Y_flat.device)
        return Y_flat.view(*orig_shape[:-1], out_features)

    # If dimensions exceed small-matrix register capacity, use optimized batched GEMMs
    if M1 > 64 or M2 > 64 or N1 > 64 or N2 > 64:
        X_reshaped = x_2d.view(M, N1, N2)
        T = torch.matmul(X_reshaped, B.t())
        Y = torch.matmul(A, T.transpose(1, 2)).transpose(1, 2)
        Y_flat = Y.reshape(M, out_features)
        if bias is not None:
            Y_flat = Y_flat + bias.to(dtype=Y_flat.dtype, device=Y_flat.device)
        return Y_flat.view(*orig_shape[:-1], out_features)

    Y = torch.empty((M, out_features), device=x.device, dtype=x.dtype)

    grid = lambda META: (triton.cdiv(M, META["BLOCK_M"]),)

    has_bias = bias is not None
    bias_t = bias.contiguous() if has_bias else torch.empty(0, device=x.device, dtype=x.dtype)
    x_c = x_2d.contiguous()
    A_c = A.contiguous()
    B_c = B.contiguous()

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

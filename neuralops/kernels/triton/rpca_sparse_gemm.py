#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
RPCA Low-Rank and Sparse GEMM

Fuses low-rank factor projection with structured sparse outlier matrix multiplication into a single unified pass.
"""

from __future__ import annotations

from typing import Optional, Tuple, Union
import warnings

import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 64, "BLOCK_K": 32}, num_stages=3, num_warps=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32}, num_stages=3, num_warps=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 32}, num_stages=4, num_warps=8),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 32, "BLOCK_K": 64}, num_stages=2, num_warps=4),
    ],
    key=["M", "N", "K", "R"],
)
@triton.jit
def _fused_rpca_gemm_kernel(
    x_ptr,
    v_ptr,
    u_ptr,
    crow_ptr,
    col_ptr,
    val_ptr,
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
    HAS_SPARSE: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_R: tl.constexpr,
):
    """
    Fused Low-Rank + Sparse CSR GEMM Kernel.
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

    z_tile = z_acc.to(x_ptr.dtype.element_ty)

    u_ptrs = u_ptr + offs_r[:, None] * stride_ur + offs_n[None, :] * stride_un
    u_tile = tl.load(u_ptrs, mask=mask_r[:, None] & mask_n[None, :], other=0.0)

    y_acc = tl.dot(z_tile, u_tile, allow_tf32=True)

    if HAS_SPARSE:
        for i in range(0, BLOCK_N):
            curr_n = pid_n * BLOCK_N + i
            if curr_n < N:
                row_start = tl.load(crow_ptr + curr_n)
                row_end = tl.load(crow_ptr + curr_n + 1)

                for idx in range(row_start, row_end):
                    col = tl.load(col_ptr + idx)
                    val = tl.load(val_ptr + idx)
                    x_vals = tl.load(x_ptr + offs_m * stride_xm + col * stride_xk, mask=mask_m, other=0.0)

                    y_acc = y_acc + tl.where(
                        (tl.arange(0, BLOCK_N) == i)[None, :],
                        (x_vals * val).to(tl.float32)[:, None],
                        0.0,
                    )

    if HAS_BIAS:
        b_ptrs = b_ptr + offs_n
        b_tile = tl.load(b_ptrs, mask=mask_n, other=0.0)
        y_acc += b_tile[None, :].to(tl.float32)

    y_ptrs = y_ptr + offs_m[:, None] * stride_ym + offs_n[None, :] * stride_yn
    tl.store(y_ptrs, y_acc.to(y_ptr.dtype.element_ty), mask=mask_m[:, None] & mask_n[None, :])


def triton_fused_lowrank_sparse_gemm(
    x: torch.Tensor,
    v_r: torch.Tensor,
    u_r: torch.Tensor,
    s: Optional[Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]] = None,
    bias: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    High-performance Triton wrapper for fused low-rank + sparse GEMM:
        y = (x @ V_r^T) @ U_r^T + (x @ S^T) + bias

    Args:
        x: Input tensor of shape (M, K) or (B, Seq, K).
        v_r: Down-projection low-rank factor. Shape (r, K) or (K, r).
        u_r: Up-projection low-rank factor. Shape (N, r) or (r, N).
        s: Sparse component. Can be:
           - A torch CSR tensor (is_sparse_csr=True)
           - A tuple (crow_indices, col_indices, values)
           - A dense torch.Tensor with sparse non-zeros
           - None
        bias: Optional bias vector of shape (N,).

    Returns:
        Tensor y of shape (*x.shape[:-1], N).
    """
    orig_shape = x.shape
    x_2d = x.view(-1, orig_shape[-1])
    M, K = x_2d.shape

    if v_r.shape[0] == K:
        V_mat = v_r
    elif v_r.shape[1] == K:
        V_mat = v_r.t().contiguous()
    else:
        raise ValueError(f"Shape mismatch for V_r: expected K={K}, got {v_r.shape}")

    R = V_mat.shape[1]

    if u_r.shape[0] == R:
        U_mat = u_r
    elif u_r.shape[1] == R:
        U_mat = u_r.t().contiguous()
    else:
        raise ValueError(f"Shape mismatch for U_r: expected R={R}, got {u_r.shape}")

    N = U_mat.shape[1]

    has_sparse = s is not None
    crow_indices = None
    col_indices = None
    values = None

    if has_sparse:
        if isinstance(s, tuple) and len(s) == 3:
            crow_indices, col_indices, values = s
        elif hasattr(s, "is_sparse_csr") and s.is_sparse_csr:
            crow_indices = s.crow_indices()
            col_indices = s.col_indices()
            values = s.values()
        elif isinstance(s, torch.Tensor) and not s.is_sparse:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UserWarning, message=".*Sparse CSR tensor support.*")
                s_csr = s.to_sparse_csr()
            crow_indices = s_csr.crow_indices()
            col_indices = s_csr.col_indices()
            values = s_csr.values()
        else:
            has_sparse = False

    if not x.is_cuda or R > 256:
        y_lr = torch.matmul(torch.matmul(x_2d, V_mat), U_mat)
        if has_sparse:
            if isinstance(s, torch.Tensor) and not s.is_sparse:
                y_sp = torch.matmul(x_2d, s.t())
            elif hasattr(s, "is_sparse_csr") and s.is_sparse_csr:
                y_sp = torch.matmul(x_2d, s.to_dense().t())
            else:
                s_dense = torch.zeros((N, K), dtype=x.dtype, device=x.device)
                for i in range(N):
                    st = crow_indices[i].item()
                    ed = crow_indices[i + 1].item()
                    s_dense[i, col_indices[st:ed]] = values[st:ed]
                y_sp = torch.matmul(x_2d, s_dense.t())
            y = y_lr + y_sp
        else:
            y = y_lr

        if bias is not None:
            y = y + bias.to(dtype=y.dtype, device=y.device)
        return y.view(*orig_shape[:-1], N)

    y = torch.empty((M, N), device=x.device, dtype=x.dtype)
    BLOCK_R = max(16, triton.next_power_of_2(R))

    has_bias = bias is not None
    b_t = bias.contiguous() if has_bias else torch.empty(0, device=x.device, dtype=x.dtype)
    x_c = x_2d.contiguous()
    V_c = V_mat.contiguous()
    U_c = U_mat.contiguous()

    if has_sparse:
        crow_c = crow_indices.contiguous().to(torch.int32)
        col_c = col_indices.contiguous().to(torch.int32)
        val_c = values.contiguous().to(x.dtype)
    else:
        crow_c = torch.empty(0, device=x.device, dtype=torch.int32)
        col_c = torch.empty(0, device=x.device, dtype=torch.int32)
        val_c = torch.empty(0, device=x.device, dtype=x.dtype)

    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_M"]),
        triton.cdiv(N, META["BLOCK_N"]),
    )

    _fused_rpca_gemm_kernel[grid](
        x_c,
        V_c,
        U_c,
        crow_c,
        col_c,
        val_c,
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
        HAS_SPARSE=has_sparse,
        HAS_BIAS=has_bias,
        BLOCK_R=BLOCK_R,
    )

    return y.view(*orig_shape[:-1], N)


rpca_sparse_gemm = triton_fused_lowrank_sparse_gemm
fused_lowrank_sparse_gemm = triton_fused_lowrank_sparse_gemm

__all__ = [
    "triton_fused_lowrank_sparse_gemm",
    "rpca_sparse_gemm",
    "fused_lowrank_sparse_gemm",
]

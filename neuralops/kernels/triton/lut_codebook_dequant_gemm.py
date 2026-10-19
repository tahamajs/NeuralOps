#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
LUT Codebook Dequantization GEMM

Dequantizes non-linear lookup-table compressed weights directly in register files before matrix multiplication.
"""

from __future__ import annotations

import math
from typing import Any, Optional, Tuple

import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 64, "BLOCK_K": 64}, num_stages=3, num_warps=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32}, num_stages=3, num_warps=4),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 32, "BLOCK_K": 64}, num_stages=2, num_warps=4),
    ],
    key=["M", "N", "K"],
)
@triton.jit
def _lut_codebook_dequant_gemm_kernel(
    x_ptr,
    indices_packed_ptr,
    codebook_ptr,
    b_ptr,
    y_ptr,
    M,
    N,
    K,
    group_size,
    stride_xm,
    stride_xk,
    stride_in,
    stride_ik,
    stride_cg,
    stride_cc,
    stride_ym,
    stride_yn,
    HAS_BIAS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    mask_m = offs_m < M
    mask_n = offs_n < N

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    num_k_blocks = tl.cdiv(K, BLOCK_K)
    for k_idx in range(0, num_k_blocks):
        offs_k_packed = (k_idx * (BLOCK_K // 2)) + tl.arange(0, BLOCK_K // 2)
        mask_k_packed = offs_k_packed < (K // 2)

        i_ptrs = indices_packed_ptr + offs_n[:, None] * stride_in + offs_k_packed[None, :] * stride_ik
        packed = tl.load(i_ptrs, mask=mask_n[:, None] & mask_k_packed[None, :], other=0)

        idx_0 = (packed & 0x0F).to(tl.int32)
        idx_1 = ((packed >> 4) & 0x0F).to(tl.int32)

        g_idx_0 = (2 * offs_k_packed) // group_size
        g_idx_1 = (2 * offs_k_packed + 1) // group_size

        c_ptrs_0 = codebook_ptr + g_idx_0[None, :] * stride_cg + idx_0 * stride_cc
        c_ptrs_1 = codebook_ptr + g_idx_1[None, :] * stride_cg + idx_1 * stride_cc

        w_0 = tl.load(c_ptrs_0, mask=mask_n[:, None] & mask_k_packed[None, :], other=0.0)
        w_1 = tl.load(c_ptrs_1, mask=mask_n[:, None] & mask_k_packed[None, :], other=0.0)

        offs_k_0 = 2 * offs_k_packed
        offs_k_1 = 2 * offs_k_packed + 1

        x_ptrs_0 = x_ptr + offs_m[:, None] * stride_xm + offs_k_0[None, :] * stride_xk
        x_ptrs_1 = x_ptr + offs_m[:, None] * stride_xm + offs_k_1[None, :] * stride_xk

        x_0 = tl.load(x_ptrs_0, mask=mask_m[:, None] & (offs_k_0 < K)[None, :], other=0.0)
        x_1 = tl.load(x_ptrs_1, mask=mask_m[:, None] & (offs_k_1 < K)[None, :], other=0.0)

        acc += tl.dot(x_0, tl.trans(w_0.to(x_0.dtype)), allow_tf32=True)
        acc += tl.dot(x_1, tl.trans(w_1.to(x_1.dtype)), allow_tf32=True)

    if HAS_BIAS:
        b_ptrs = b_ptr + offs_n
        b_tile = tl.load(b_ptrs, mask=mask_n, other=0.0)
        acc += b_tile[None, :].to(tl.float32)

    y_ptrs = y_ptr + offs_m[:, None] * stride_ym + offs_n[None, :] * stride_yn
    tl.store(y_ptrs, acc.to(x_ptr.dtype.element_ty), mask=mask_m[:, None] & mask_n[None, :])


def triton_lut_codebook_dequant_gemm(
    x: torch.Tensor,
    indices_packed: torch.Tensor,
    codebook: torch.Tensor,
    group_size: int = 64,
    bias: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    High-performance Triton GEMM with in-register codebook LUT dequantization.

    Args:
        x: Input activations (M, K) or (B, Seq, K).
        indices_packed: Packed 4-bit cluster indices (N, K // 2) as torch.uint8.
        codebook: Centroid codebook of shape (num_groups, 16) or (16,).
        group_size: Channels per codebook group (default: 64).
        bias: Optional bias vector of shape (N,).

    Returns:
        Tensor of shape (*x.shape[:-1], N).
    """
    orig_shape = x.shape
    x_2d = x.view(-1, orig_shape[-1])
    M, K = x_2d.shape
    N = indices_packed.shape[0]

    if codebook.dim() == 1:
        codebook_2d = codebook.unsqueeze(0).expand(K // group_size, 16)
    else:
        codebook_2d = codebook

    if not x.is_cuda:
        low = (indices_packed & 0x0F).to(torch.long)
        high = ((indices_packed >> 4) & 0x0F).to(torch.long)
        indices_unpacked = torch.empty((N, K), dtype=torch.long, device=x.device)
        indices_unpacked[:, 0::2] = low
        indices_unpacked[:, 1::2] = high

        w_dequant = torch.empty((N, K), dtype=x.dtype, device=x.device)
        num_groups = K // group_size
        for g in range(num_groups):
            st = g * group_size
            ed = min(K, (g + 1) * group_size)
            idx_g = indices_unpacked[:, st:ed]
            w_dequant[:, st:ed] = codebook_2d[g][idx_g].to(x.dtype)

        y = torch.matmul(x_2d, w_dequant.t())
        if bias is not None:
            y = y + bias.to(dtype=y.dtype, device=y.device)
        return y.view(*orig_shape[:-1], N)

    y = torch.empty((M, N), device=x.device, dtype=x.dtype)

    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_M"]),
        triton.cdiv(N, META["BLOCK_N"]),
    )

    has_bias = bias is not None
    b_t = bias.contiguous() if has_bias else torch.empty(0, device=x.device, dtype=x.dtype)
    x_c = x_2d.contiguous()
    idx_c = indices_packed.contiguous()
    cb_c = codebook_2d.contiguous().to(x.dtype)

    _lut_codebook_dequant_gemm_kernel[grid](
        x_c,
        idx_c,
        cb_c,
        b_t,
        y,
        M,
        N,
        K,
        group_size,
        x_c.stride(0),
        x_c.stride(1),
        idx_c.stride(0),
        idx_c.stride(1),
        cb_c.stride(0),
        cb_c.stride(1),
        y.stride(0),
        y.stride(1),
        HAS_BIAS=has_bias,
    )

    return y.view(*orig_shape[:-1], N)

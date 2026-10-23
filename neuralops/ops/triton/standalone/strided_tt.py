#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Strided Tensor-Train Contraction

Evaluates high-dimensional Tensor-Train matrix contractions via strided multi-index tensor products in SRAM.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
from neuralops.ops.triton._base import triton, tl, HAS_TRITON


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
    """
    Zero-copy TT core contraction into 2D weight matrix W:
        W[n, k] = sum_r core0[p0, q0, r] * core1[r, p1, q1]
    where n = q0 * q1_dim + q1, k = p0 * p1_dim + p1.
    Permutation (p0, q0, p1, q1) -> (q0, q1, p0, p1) is resolved entirely via index math.
    """
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


def triton_strided_tt_contraction(
    core0: torch.Tensor,
    core1: torch.Tensor,
    p_dims: Tuple[int, int],
    q_dims: Tuple[int, int],
) -> torch.Tensor:
    """
    Zero-copy reconstruction of weight matrix W from TT cores using Triton strided pointer arithmetic.

    Args:
        core0: First TT core of shape (1, p0, q0, r) or (p0, q0, r).
        core1: Second TT core of shape (r, p1, q1, 1) or (r, p1, q1).
        p_dims: (p0, p1) such that p0 * p1 = in_features.
        q_dims: (q0, q1) such that q0 * q1 = out_features.

    Returns:
        Reconstructed 2D weight matrix W of shape (out_features, in_features).
    """
    c0 = core0.squeeze(0) if core0.dim() == 4 and core0.shape[0] == 1 else core0
    c1 = core1.squeeze(-1) if core1.dim() == 4 and core1.shape[-1] == 1 else core1

    p0, p1 = p_dims
    q0, q1 = q_dims
    R = c0.shape[-1]
    K = p0 * p1
    N = q0 * q1

    if not c0.is_cuda:
        res = torch.tensordot(c0, c1, dims=([-1], [0]))
        res = res.permute(1, 3, 0, 2).contiguous()
        return res.view(N, K)

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
    """
    Zero-copy Tensor-Train linear layer forward execution:
        y = x @ W^T + bias
    where W is reconstructed without VRAM copies via Triton strided pointer arithmetic.

    Args:
        x: Input activations of shape (..., in_features).
        core0: TT core 0.
        core1: TT core 1.
        p_dims: Input dimension factors (p0, p1).
        q_dims: Output dimension factors (q0, q1).
        bias: Optional bias tensor of shape (out_features,).

    Returns:
        Output tensor of shape (..., out_features).
    """
    orig_shape = x.shape
    in_features = orig_shape[-1]
    x_2d = x.view(-1, in_features)

    W = triton_strided_tt_contraction(core0, core1, p_dims, q_dims)

    out = torch.matmul(x_2d, W.t())
    if bias is not None:
        out = out + bias.to(dtype=out.dtype, device=out.device)

    out_features = q_dims[0] * q_dims[1]
    return out.view(*orig_shape[:-1], out_features)


strided_tt = triton_strided_tt_contraction
strided_tt_contraction = triton_strided_tt_contraction
strided_tt_linear = triton_strided_tt_linear

__all__ = [
    "triton_strided_tt_contraction",
    "triton_strided_tt_linear",
    "strided_tt",
    "strided_tt_contraction",
    "strided_tt_linear",
]

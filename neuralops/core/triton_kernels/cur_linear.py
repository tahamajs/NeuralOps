#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
CUR Matrix Decomposition Linear Projection

Computes low-rank CUR matrix factorized projections with fused cross-factor contractions and analytical gradients.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch

try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False
    class _MockTriton:
        @staticmethod
        def jit(fn):
            return fn
        @staticmethod
        def autotune(*args, **kwargs):
            return lambda fn: fn
        @staticmethod
        def cdiv(a, b):
            return (a + b - 1) // b
        class Config:
            def __init__(self, *args, **kwargs):
                pass
    triton = _MockTriton()
    class _MockTl:
        constexpr = int
    tl = _MockTl()


if HAS_TRITON:
    @triton.autotune(
        configs=[
            triton.Config({"BLOCK_M": 32, "BLOCK_N": 32, "BLOCK_K": 32}, num_warps=4, num_stages=2),
            triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32}, num_warps=4, num_stages=3),
            triton.Config({"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 32}, num_warps=4, num_stages=3),
            triton.Config({"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 32}, num_warps=4, num_stages=3),
            triton.Config({"BLOCK_M": 32, "BLOCK_N": 64, "BLOCK_K": 64}, num_warps=4, num_stages=2),
        ],
        key=["M", "N", "K"],
    )
    @triton.jit
    def _gemm_bias_kernel(
        A_ptr, B_ptr, Bias_ptr, C_ptr,
        M, N, K,
        stride_am, stride_ak,
        stride_bk, stride_bn,
        stride_cm, stride_cn,
        HAS_BIAS: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, BLOCK_K)

        mask_m = offs_m < M
        mask_n = offs_n < N

        acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

        for k in range(0, K, BLOCK_K):
            k_idx = k + offs_k
            mask_k = k_idx < K

            a = tl.load(
                A_ptr + offs_m[:, None] * stride_am + k_idx[None, :] * stride_ak,
                mask=mask_m[:, None] & mask_k[None, :],
                other=0.0,
            )
            b = tl.load(
                B_ptr + k_idx[:, None] * stride_bk + offs_n[None, :] * stride_bn,
                mask=mask_k[:, None] & mask_n[None, :],
                other=0.0,
            )

            acc += tl.dot(a, b)

        if HAS_BIAS:
            bias = tl.load(Bias_ptr + offs_n, mask=mask_n, other=0.0)
            acc += bias[None, :]

        tl.store(
            C_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn,
            acc.to(A_ptr.dtype.element_ty),
            mask=mask_m[:, None] & mask_n[None, :],
        )
else:
    def _gemm_bias_kernel(*args, **kwargs):
        pass


def triton_cur_linear_pytorch_ref(
    x: torch.Tensor,
    R: torch.Tensor,
    U: torch.Tensor,
    C: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Pure PyTorch reference implementation for CUR matrix decomposition linear."""
    orig_shape = x.shape
    _M = x.reshape(-1, orig_shape[-1]).shape[0]
    K = orig_shape[-1]
    x_2d = x.reshape(-1, K)

    z1 = torch.matmul(x_2d.float(), R.float().t())
    z2 = torch.matmul(z1, U.float().t())
    out_2d = torch.matmul(z2, C.float().t())
    if bias is not None:
        out_2d = out_2d + bias.float()
    return out_2d.to(x.dtype).reshape(*orig_shape[:-1], C.shape[0])


class TritonCURLinearFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        R: torch.Tensor,
        U: torch.Tensor,
        C: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        orig_shape = x.shape
        K = orig_shape[-1]
        x_2d = x.reshape(-1, K)
        M = x_2d.shape[0]
        r = R.shape[0]
        N = C.shape[0]

        if not HAS_TRITON or not x.is_cuda:
            out_2d = triton_cur_linear_pytorch_ref(x, R, U, C, bias=bias)
            ctx.save_for_backward(x_2d, R, U, C, bias)
            ctx.orig_shape = orig_shape
            return out_2d

        x_c = x_2d.contiguous()
        R_c = R.contiguous()
        U_c = U.contiguous()
        C_c = C.contiguous()

        W_down = torch.matmul(U_c, R_c)
        W_down_t = W_down.t().contiguous()
        z = torch.empty((M, r), device=x.device, dtype=x.dtype)

        grid_1 = lambda META: (triton.cdiv(M, META["BLOCK_M"]), triton.cdiv(r, META["BLOCK_N"]))
        _gemm_bias_kernel[grid_1](
            x_c, W_down_t, None, z,
            M, r, K,
            x_c.stride(0), x_c.stride(1),
            W_down_t.stride(0), W_down_t.stride(1),
            z.stride(0), z.stride(1),
            HAS_BIAS=False,
        )

        C_t = C_c.t().contiguous()
        b_c = bias.contiguous() if bias is not None else None
        out_2d = torch.empty((M, N), device=x.device, dtype=x.dtype)
        grid_2 = lambda META: (triton.cdiv(M, META["BLOCK_M"]), triton.cdiv(N, META["BLOCK_N"]))
        _gemm_bias_kernel[grid_2](
            z, C_t, b_c, out_2d,
            M, N, r,
            z.stride(0), z.stride(1),
            C_t.stride(0), C_t.stride(1),
            out_2d.stride(0), out_2d.stride(1),
            HAS_BIAS=(bias is not None),
        )

        ctx.save_for_backward(x_c, R_c, U_c, C_c, bias)
        ctx.orig_shape = orig_shape
        return out_2d.reshape(*orig_shape[:-1], N)

    @staticmethod
    def backward(ctx, dout: torch.Tensor) -> Tuple[Optional[torch.Tensor], ...]:
        x_2d, R, U, C, bias = ctx.saved_tensors
        M, K = x_2d.shape
        N, r = C.shape
        dout_2d = dout.reshape(-1, N).float()

        z1 = torch.matmul(x_2d.float(), R.float().t())
        z2 = torch.matmul(z1, U.float().t())

        dbias = dout_2d.sum(dim=0).to(bias.dtype) if (bias is not None and ctx.needs_input_grad[4]) else None
        dC = torch.matmul(dout_2d.t(), z2).to(C.dtype) if ctx.needs_input_grad[3] else None

        dz2 = torch.matmul(dout_2d, C.float())
        dU = torch.matmul(dz2.t(), z1).to(U.dtype) if ctx.needs_input_grad[2] else None

        dz1 = torch.matmul(dz2, U.float())
        dR = torch.matmul(dz1.t(), x_2d.float()).to(R.dtype) if ctx.needs_input_grad[1] else None

        dx = torch.matmul(dz1, R.float()).to(x_2d.dtype) if ctx.needs_input_grad[0] else None
        dx_out = dx.reshape(ctx.orig_shape) if dx is not None else None

        return dx_out, dR, dU, dC, dbias


def triton_cur_linear(
    x: torch.Tensor,
    R: torch.Tensor,
    U: torch.Tensor,
    C: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Dispatched CUR Matrix Decomposition Linear with full Autograd backward support."""
    if torch.is_grad_enabled() and any(
        t.requires_grad for t in (x, R, U, C, bias) if isinstance(t, torch.Tensor)
    ):
        return TritonCURLinearFunction.apply(x, R, U, C, bias)

    orig_shape = x.shape
    K = orig_shape[-1]
    x_2d = x.reshape(-1, K)
    M = x_2d.shape[0]
    r = R.shape[0]
    N = C.shape[0]

    if not HAS_TRITON or not x.is_cuda:
        return triton_cur_linear_pytorch_ref(x, R, U, C, bias=bias)

    x_c = x_2d.contiguous()
    R_c = R.contiguous()
    U_c = U.contiguous()
    C_c = C.contiguous()

    W_down = torch.matmul(U_c, R_c)
    W_down_t = W_down.t().contiguous()
    z = torch.empty((M, r), device=x.device, dtype=x.dtype)

    grid_1 = lambda META: (triton.cdiv(M, META["BLOCK_M"]), triton.cdiv(r, META["BLOCK_N"]))
    _gemm_bias_kernel[grid_1](
        x_c, W_down_t, None, z,
        M, r, K,
        x_c.stride(0), x_c.stride(1),
        W_down_t.stride(0), W_down_t.stride(1),
        z.stride(0), z.stride(1),
        HAS_BIAS=False,
    )

    C_t = C_c.t().contiguous()
    b_c = bias.contiguous() if bias is not None else None
    out_2d = torch.empty((M, N), device=x.device, dtype=x.dtype)
    grid_2 = lambda META: (triton.cdiv(M, META["BLOCK_M"]), triton.cdiv(N, META["BLOCK_N"]))
    _gemm_bias_kernel[grid_2](
        z, C_t, b_c, out_2d,
        M, N, r,
        z.stride(0), z.stride(1),
        C_t.stride(0), C_t.stride(1),
        out_2d.stride(0), out_2d.stride(1),
        HAS_BIAS=(bias is not None),
    )

    return out_2d.reshape(*orig_shape[:-1], N)


__all__ = [
    "triton_cur_linear",
    "triton_cur_linear_pytorch_ref",
    "TritonCURLinearFunction",
]

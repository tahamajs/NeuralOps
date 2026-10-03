#!/usr/bin/env python3
"""
Triton Low-Rank Projection Kernel (Forward & Backward).

Computes low-rank bottleneck projections:
  y = (x @ V.T) @ U.T + bias

Used by:
  - Multi-Head Latent Attention (MLA)
  - SVD-LLM, ASVD, Direct SVD, DOBI SVD, Frandi
  - Triton SVD Linear Attention low-rank projections
"""

from __future__ import annotations

from typing import Any

import torch

from neuralops.core.triton_kernels._common import (
    HAS_TRITON,
    check_triton_suitability,
    low_rank_pytorch_fallback,
)

if HAS_TRITON:
    import triton
    import triton.language as tl

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


class _TritonLowRankAutogradFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        x: torch.Tensor,
        U: torch.Tensor,
        V: torch.Tensor,
        bias: torch.Tensor | None,
    ) -> torch.Tensor:
        orig_shape = x.shape
        D_in = orig_shape[-1]
        x_2d = x.contiguous().view(-1, D_in)
        M = x_2d.shape[0]

        # V_T has shape [D_in, Rank]
        V_mat = V.t() if V.shape[-1] == D_in else V
        Rank = V_mat.shape[1]

        # U_T has shape [Rank, D_out]
        U_mat = U.t() if U.shape[-1] == Rank else U
        D_out = U_mat.shape[1]

        # Pass 1: z = x @ V_mat
        z = torch.empty((M, Rank), dtype=x.dtype, device=x.device)
        grid_z = (
            triton.cdiv(M, 64),
            triton.cdiv(Rank, 64),
        )
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

        # Pass 2: y = z @ U_mat + bias
        y = torch.empty((M, D_out), dtype=x.dtype, device=x.device)
        grid_y = (
            triton.cdiv(M, 64),
            triton.cdiv(D_out, 64),
        )
        has_bias = bias is not None
        bias_t = bias.contiguous() if has_bias else z
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

        ctx.save_for_backward(x, U, V, bias)
        return y.view(*orig_shape[:-1], D_out)

    @staticmethod
    def backward(ctx: Any, grad_out: torch.Tensor) -> tuple[torch.Tensor | None, ...]:
        x, U, V, bias = ctx.saved_tensors
        orig_shape = x.shape
        D_in = orig_shape[-1]
        x_2d = x.contiguous().view(-1, D_in)
        x_2d.shape[0]

        # V_mat has shape [D_in, Rank]
        V_mat = V.t() if V.shape[-1] == D_in else V
        Rank = V_mat.shape[1]

        # U_mat has shape [Rank, D_out]
        U_mat = U.t() if U.shape[-1] == Rank else U
        D_out = U_mat.shape[1]

        grad_out_2d = grad_out.contiguous().view(-1, D_out)

        g_x = None
        g_U = None
        g_V = None
        g_bias = None

        if bias is not None and ctx.needs_input_grad[3]:
            g_bias = grad_out_2d.sum(dim=0).to(bias.dtype)

        # dz = grad_out @ U_mat.t() -> [M, Rank]
        dz = torch.matmul(grad_out_2d, U_mat.t())

        if ctx.needs_input_grad[0]:
            g_x_2d = torch.matmul(dz, V_mat.t())
            g_x = g_x_2d.view(*orig_shape).to(x.dtype)

        if ctx.needs_input_grad[1]:
            # z = x @ V_mat -> [M, Rank]
            z = torch.matmul(x_2d, V_mat)
            g_U_mat = torch.matmul(z.t(), grad_out_2d)  # [Rank, D_out]
            if U.shape[-1] == Rank:
                g_U = g_U_mat.t().to(U.dtype)
            else:
                g_U = g_U_mat.to(U.dtype)

        if ctx.needs_input_grad[2]:
            g_V_mat = torch.matmul(x_2d.t(), dz)  # [D_in, Rank]
            if V.shape[-1] == D_in:
                g_V = g_V_mat.t().to(V.dtype)
            else:
                g_V = g_V_mat.to(V.dtype)

        return g_x, g_U, g_V, g_bias


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
    **kwargs: Any,
) -> torch.Tensor:
    """Forward entrypoint for fused Triton low-rank projection: y = (x @ V.T) @ U.T + bias."""
    if not is_supported(x, U, V, bias=bias, **kwargs):
        return low_rank_pytorch_fallback(x, U, V, bias=bias, **kwargs)
    return _TritonLowRankAutogradFunction.apply(x, U, V, bias)


def bwd(
    x: torch.Tensor,
    U: torch.Tensor,
    V: torch.Tensor,
    grad_out: torch.Tensor,
    bias: torch.Tensor | None = None,
    **kwargs: Any,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Backward entrypoint computing gradients for x, U, V, and bias."""
    x_req = x.detach().requires_grad_(True)
    U_req = U.detach().requires_grad_(True)
    V_req = V.detach().requires_grad_(True)
    b_req = bias.detach().requires_grad_(True) if bias is not None else None
    with torch.enable_grad():
        out = fwd(x_req, U_req, V_req, bias=b_req, **kwargs)
        diff_vars = [x_req, U_req, V_req]
        if b_req is not None:
            diff_vars.append(b_req)
        grads = torch.autograd.grad(out, diff_vars, grad_out)
    return grads[0], grads[1], grads[2], (grads[3] if b_req is not None else None)

"""
Triton Fused Activation-Scaling Quantized GEMM Kernel (Phase 3 / Kernel 12).

Target Modules: smoothquant_real (smoothquant.py), faq_quantizer (w4a16)
Solves:
    SmoothQuant balances outlier difficulty by multiplying weights by diag(s) and dividing
    activations by diag(s): Y = (X * s_inv) @ (W * s)^T + bias.
    Materializing (X * s_inv) in PyTorch creates an extra full-sized tensor in HBM on every layer.

This kernel scales activations on-the-fly in SRAM/registers immediately prior to Tensor Core GEMM.
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
def _fused_act_scale_gemm_kernel(
    x_ptr,
    s_inv_ptr,
    w_ptr,
    b_ptr,
    y_ptr,
    M,
    N,
    K,
    stride_xm,
    stride_xk,
    stride_wn,
    stride_wk,
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
        offs_k = k_idx * BLOCK_K + tl.arange(0, BLOCK_K)
        mask_k = offs_k < K

        # Load X tile [BLOCK_M, BLOCK_K]
        x_ptrs = x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk
        x_tile = tl.load(x_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0)

        # Load s_inv tile [BLOCK_K]
        s_inv_tile = tl.load(s_inv_ptr + offs_k, mask=mask_k, other=1.0)

        # In-SRAM activation smoothing: x_smoothed = x_tile * s_inv
        x_smoothed = x_tile * s_inv_tile[None, :]

        # Load W tile [BLOCK_N, BLOCK_K]
        w_ptrs = w_ptr + offs_n[:, None] * stride_wn + offs_k[None, :] * stride_wk
        w_tile = tl.load(w_ptrs, mask=mask_n[:, None] & mask_k[None, :], other=0.0)

        acc += tl.dot(x_smoothed, tl.trans(w_tile), allow_tf32=True)

    if HAS_BIAS:
        b_ptrs = b_ptr + offs_n
        b_tile = tl.load(b_ptrs, mask=mask_n, other=0.0)
        acc += b_tile[None, :].to(tl.float32)

    y_ptrs = y_ptr + offs_m[:, None] * stride_ym + offs_n[None, :] * stride_yn
    tl.store(y_ptrs, acc.to(x_ptr.dtype.element_ty), mask=mask_m[:, None] & mask_n[None, :])


def triton_fused_act_scale_quant_gemm(
    x: torch.Tensor,
    s_inv: torch.Tensor,
    w: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    High-performance Triton fused activation-scaling GEMM:
        Y = (X * s_inv) @ W^T + bias

    Args:
        x: Input activations of shape (M, K) or (B, Seq, K).
        s_inv: Inverse activation scale vector of shape (K,).
        w: Weight matrix of shape (N, K).
        bias: Optional bias of shape (N,).

    Returns:
        Tensor of shape (*x.shape[:-1], N).
    """
    orig_shape = x.shape
    x_2d = x.view(-1, orig_shape[-1])
    M, K = x_2d.shape
    N = w.shape[0]

    if not x.is_cuda:
        # PyTorch reference
        x_smooth = x_2d.float() * s_inv.view(1, -1).to(dtype=torch.float32, device=x.device)
        y = torch.matmul(x_smooth.to(x.dtype), w.t())
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
    s_inv_c = s_inv.contiguous().to(x.dtype)
    w_c = w.contiguous()

    _fused_act_scale_gemm_kernel[grid](
        x_c,
        s_inv_c,
        w_c,
        b_t,
        y,
        M,
        N,
        K,
        x_c.stride(0),
        x_c.stride(1),
        w_c.stride(0),
        w_c.stride(1),
        y.stride(0),
        y.stride(1),
        HAS_BIAS=has_bias,
    )

    return y.view(*orig_shape[:-1], N)

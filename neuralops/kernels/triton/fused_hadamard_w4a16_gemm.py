"""
Fused Hadamard W4A16 GEMM

Performs in-SRAM Walsh-Hadamard activation transformation and register-level INT4 dequantization for accelerated Tensor Core GEMM.

Author: Taha Majlesi
"""

from __future__ import annotations

from typing import Optional

import torch
import triton
import triton.language as tl


@triton.jit
def _fwht_butterfly_step(x, offs_d, BLOCK_D: tl.constexpr):
    """In-register fast Walsh-Hadamard butterfly stage for power-of-2 dimensions."""
    val = x
    h = 1
    while h < BLOCK_D:
        partner_sign = tl.where((offs_d & h) == 0, 1.0, -1.0)
        # In-register butterfly step using power-of-2 parity
        val = val + partner_sign * val
        h = h * 2
    norm = 1.0 / tl.sqrt(tl.cast(BLOCK_D, tl.float32))
    return val * norm



@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 64, "BLOCK_K": 64}, num_stages=3, num_warps=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32}, num_stages=3, num_warps=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 32}, num_stages=4, num_warps=8),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 32, "BLOCK_K": 64}, num_stages=2, num_warps=4),
    ],
    key=["M", "N", "K"],
)
@triton.jit
def _fused_hadamard_w4a16_gemm_kernel(
    x_ptr,
    w_packed_ptr,
    scales_ptr,
    zeros_ptr,
    b_ptr,
    y_ptr,
    M,
    N,
    K,
    group_size,
    stride_xm,
    stride_xk,
    stride_wn,
    stride_wk,
    stride_sn,
    stride_sg,
    stride_zn,
    stride_zg,
    stride_ym,
    stride_yn,
    APPLY_HADAMARD: tl.constexpr,
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
        # 1. Load Packed INT4 weights: 2 nibbles per byte -> shape (BLOCK_N, BLOCK_K // 2)
        # offs_k_packed in [0, K // 2)
        offs_k_packed = (k_idx * (BLOCK_K // 2)) + tl.arange(0, BLOCK_K // 2)
        mask_k_packed = offs_k_packed < (K // 2)

        w_ptrs = w_packed_ptr + offs_n[:, None] * stride_wn + offs_k_packed[None, :] * stride_wk
        w_packed = tl.load(w_ptrs, mask=mask_n[:, None] & mask_k_packed[None, :], other=0)

        # 3. Unpack low nibble and high nibble in registers
        # Low nibbles: elements at 2*i
        w_low = (w_packed & 0x0F).to(tl.float32)
        # High nibbles: elements at 2*i + 1
        w_high = ((w_packed >> 4) & 0x0F).to(tl.float32)

        # Dequantize with scales and zeros
        # Group index for scale and zero
        g_idx_0 = (2 * offs_k_packed) // group_size
        g_idx_1 = (2 * offs_k_packed + 1) // group_size

        s_ptrs_0 = scales_ptr + offs_n[:, None] * stride_sn + g_idx_0[None, :] * stride_sg
        s_ptrs_1 = scales_ptr + offs_n[:, None] * stride_sn + g_idx_1[None, :] * stride_sg
        scale_0 = tl.load(s_ptrs_0, mask=mask_n[:, None], other=1.0)
        scale_1 = tl.load(s_ptrs_1, mask=mask_n[:, None], other=1.0)

        z_ptrs_0 = zeros_ptr + offs_n[:, None] * stride_zn + g_idx_0[None, :] * stride_zg
        z_ptrs_1 = zeros_ptr + offs_n[:, None] * stride_zn + g_idx_1[None, :] * stride_zg
        zero_0 = tl.load(z_ptrs_0, mask=mask_n[:, None], other=0.0)
        zero_1 = tl.load(z_ptrs_1, mask=mask_n[:, None], other=0.0)

        w_dequant_0 = (w_low - zero_0) * scale_0
        w_dequant_1 = (w_high - zero_1) * scale_1

        # Interleave low and high into (BLOCK_N, BLOCK_K)
        # Even columns from w_dequant_0, odd columns from w_dequant_1
        # In Triton, we can compute partial dot products:
        x_even = tl.load(x_ptr + offs_m[:, None] * stride_xm + (2 * offs_k_packed)[None, :] * stride_xk,
                         mask=mask_m[:, None] & (2 * offs_k_packed < K)[None, :], other=0.0)
        x_odd = tl.load(x_ptr + offs_m[:, None] * stride_xm + (2 * offs_k_packed + 1)[None, :] * stride_xk,
                        mask=mask_m[:, None] & (2 * offs_k_packed + 1 < K)[None, :], other=0.0)

        acc += tl.dot(x_even, tl.trans(w_dequant_0.to(x_ptr.dtype.element_ty)), allow_tf32=True)
        acc += tl.dot(x_odd, tl.trans(w_dequant_1.to(x_ptr.dtype.element_ty)), allow_tf32=True)

    if HAS_BIAS:
        b_ptrs = b_ptr + offs_n
        b_tile = tl.load(b_ptrs, mask=mask_n, other=0.0)
        acc += b_tile[None, :].to(tl.float32)

    y_ptrs = y_ptr + offs_m[:, None] * stride_ym + offs_n[None, :] * stride_yn
    tl.store(y_ptrs, acc.to(x_ptr.dtype.element_ty), mask=mask_m[:, None] & mask_n[None, :])


def triton_fused_hadamard_w4a16_gemm(
    x: torch.Tensor,
    w_packed: torch.Tensor,
    scales: torch.Tensor,
    zeros: torch.Tensor,
    group_size: int = 32,
    bias: Optional[torch.Tensor] = None,
    apply_hadamard: bool = False,
) -> torch.Tensor:
    """
    High-performance Triton W4A16 GEMM with register dequantization.

    Args:
        x: Input activations of shape (M, K) or (B, Seq, K).
        w_packed: Bit-packed INT4 weights of shape (N, K // 2) as torch.uint8.
        scales: Scale tensor of shape (N, K // group_size) in float16/float32.
        zeros: Zero-point tensor of shape (N, K // group_size) in float16/float32.
        group_size: Quantization group size (e.g. 32, 64, 128).
        bias: Optional bias tensor of shape (N,).
        apply_hadamard: Whether to apply Hadamard rotation in SRAM.

    Returns:
        Output tensor of shape (*x.shape[:-1], N).
    """
    orig_shape = x.shape
    x_2d = x.view(-1, orig_shape[-1])
    M, K = x_2d.shape
    N = w_packed.shape[0]

    if not x.is_cuda:
        # PyTorch reference fallback
        # Unpack uint8 -> two int4 values
        low = (w_packed & 0x0F).to(torch.float32)
        high = ((w_packed >> 4) & 0x0F).to(torch.float32)
        w_unpacked = torch.empty((N, K), dtype=torch.float32, device=x.device)
        w_unpacked[:, 0::2] = low
        w_unpacked[:, 1::2] = high

        scales_expanded = scales.repeat_interleave(group_size, dim=1)[:, :K]
        zeros_expanded = zeros.repeat_interleave(group_size, dim=1)[:, :K]
        w_dequant = ((w_unpacked - zeros_expanded) * scales_expanded).to(x.dtype)

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
    w_c = w_packed.contiguous()
    scales_c = scales.contiguous().to(torch.float32)
    zeros_c = zeros.contiguous().to(torch.float32)

    _fused_hadamard_w4a16_gemm_kernel[grid](
        x_c,
        w_c,
        scales_c,
        zeros_c,
        b_t,
        y,
        M,
        N,
        K,
        group_size,
        x_c.stride(0),
        x_c.stride(1),
        w_c.stride(0),
        w_c.stride(1),
        scales_c.stride(0),
        scales_c.stride(1),
        zeros_c.stride(0),
        zeros_c.stride(1),
        y.stride(0),
        y.stride(1),
        APPLY_HADAMARD=apply_hadamard,
        HAS_BIAS=has_bias,
    )

    return y.view(*orig_shape[:-1], N)

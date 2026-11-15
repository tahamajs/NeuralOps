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

if HAS_TRITON:

    @triton.jit
    def _dynamic_fp8_quant_kernel(
        X_ptr,
        Y_fp8_ptr,
        Scale_ptr,
        M,
        K,
        stride_xm,
        stride_xk,
        stride_ym,
        stride_yk,
        BLOCK_K: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        offs_k = tl.arange(0, BLOCK_K)
        mask = offs_k < K

        x_ptrs = X_ptr + pid_m * stride_xm + offs_k * stride_xk
        x = tl.load(x_ptrs, mask=mask, other=0.0).to(tl.float32)

        abs_x = tl.abs(x)
        amax = tl.max(abs_x, axis=0)
        scale = amax / 448.0 + 1e-8
        tl.store(Scale_ptr + pid_m, scale)

        scaled_x = x / scale
        clamped_x = tl.clamp(scaled_x, -448.0, 448.0)
        tl.store(
            Y_fp8_ptr + pid_m * stride_ym + offs_k * stride_yk, clamped_x.to(Y_fp8_ptr.dtype.element_ty), mask=mask
        )

    @triton.jit
    def _fp8_matmul_kernel(
        A_ptr,
        B_ptr,
        C_ptr,
        ScaleA_ptr,
        ScaleB_ptr,
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
    ):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)

        offs_am = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_bn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, BLOCK_K)

        mask_m = offs_am < M
        mask_n = offs_bn < N

        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        for k in range(0, K, BLOCK_K):
            mask_k = (k + offs_k) < K
            a_ptrs = A_ptr + offs_am[:, None] * stride_am + (k + offs_k)[None, :] * stride_ak
            b_ptrs = B_ptr + (k + offs_k)[:, None] * stride_bk + offs_bn[None, :] * stride_bn

            a = tl.load(a_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0)
            b = tl.load(b_ptrs, mask=mask_k[:, None] & mask_n[None, :], other=0.0)

            acc = tl.dot(a, b, acc, out_dtype=tl.float32)

        scale_a = tl.load(ScaleA_ptr + offs_am, mask=mask_m, other=1.0)
        scale_b = tl.load(ScaleB_ptr + offs_bn, mask=mask_n, other=1.0)
        acc = acc * scale_a[:, None] * scale_b[None, :]

        c_ptrs = C_ptr + offs_am[:, None] * stride_cm + offs_bn[None, :] * stride_cn
        tl.store(c_ptrs, acc.to(C_ptr.dtype.element_ty), mask=mask_m[:, None] & mask_n[None, :])


else:
    _dynamic_fp8_quant_kernel = None
    _fp8_matmul_kernel = None

def quantize_to_fp8_dynamic(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantizes a 2D tensor to torch.float8_e4m3fn with per-row scaling.

    Returns:
        (x_fp8, scale) where x = x_fp8.float() * scale[:, None]
    """
    orig_shape = x.shape
    M = x.numel() // x.shape[-1]
    K = x.shape[-1]
    x_2d = x.view(M, K).contiguous()

    has_native_fp8 = hasattr(torch, "float8_e4m3fn") and x.is_cuda
    if has_native_fp8:
        try:
            from neuralops.ops.triton.hardware_profile import get_device_hardware_profile

            prof = get_device_hardware_profile(x.device)
            if prof.sm_major >= 8 and prof.sm_minor >= 9:
                x_fp8 = torch.empty((M, K), device=x.device, dtype=torch.float8_e4m3fn)
                scale = torch.empty((M,), device=x.device, dtype=torch.float32)
                BLOCK_K = triton.next_power_of_2(K)
                grid = (M,)
                _dynamic_fp8_quant_kernel[grid](
                    x_2d,
                    x_fp8,
                    scale,
                    M,
                    K,
                    x_2d.stride(0),
                    x_2d.stride(1),
                    x_fp8.stride(0),
                    x_fp8.stride(1),
                    BLOCK_K=BLOCK_K,
                )
                return x_fp8.view(orig_shape), scale
        except Exception as e:
            logger.warning("Dynamic FP8 quantization kernel failed (%s); delegating to reference path.", e)

    amax = torch.max(torch.abs(x_2d.float()), dim=1, keepdim=True).values.clamp_min(1e-8)
    scale = (amax / 448.0).squeeze(-1)
    scaled = (x_2d.float() / amax * 448.0).clamp(-448.0, 448.0)
    if hasattr(torch, "float8_e4m3fn"):
        try:
            return scaled.to(torch.float8_e4m3fn).view(orig_shape), scale
        except Exception as e:
            logger.warning("Cast to float8_e4m3fn failed (%s); keeping original dtype.", e)
    return scaled.to(x.dtype).view(orig_shape), scale


dynamic_fp8_quant_triton = quantize_to_fp8_dynamic


def fp8_tensorcore_matmul_triton(
    a: torch.Tensor,
    b: torch.Tensor,
    scale_a: torch.Tensor | None = None,
    scale_b: torch.Tensor | None = None,
    out_dtype: torch.dtype = torch.float16,
) -> torch.Tensor:
    """Performs native hardware FP8 Tensor Core matrix multiplication."""
    if scale_a is None or scale_b is None:
        a_fp8, scale_a = quantize_to_fp8_dynamic(a)
        b_fp8, scale_b = quantize_to_fp8_dynamic(b)
    else:
        a_fp8 = a
        b_fp8 = b

    M = a_fp8.shape[0] if a_fp8.dim() == 2 else a_fp8.shape[0] * a_fp8.shape[1]
    K = a_fp8.shape[-1]
    N = b_fp8.shape[-1]

    if a_fp8.is_cuda and HAS_TRITON and hasattr(torch, "float8_e4m3fn") and a_fp8.dtype == torch.float8_e4m3fn:
        try:
            c = torch.empty((M, N), device=a_fp8.device, dtype=out_dtype)
            BLOCK_M = 32
            BLOCK_N = 32
            BLOCK_K = 32
            grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
            _fp8_matmul_kernel[grid](
                a_fp8.view(M, K),
                b_fp8.view(K, N),
                c,
                scale_a,
                scale_b,
                M,
                N,
                K,
                a_fp8.stride(0),
                a_fp8.stride(1),
                b_fp8.stride(0),
                b_fp8.stride(1),
                c.stride(0),
                c.stride(1),
                BLOCK_M=BLOCK_M,
                BLOCK_N=BLOCK_N,
                BLOCK_K=BLOCK_K,
            )
            return c
        except Exception as e:
            logger.warning("_fp8_matmul_kernel failed (%s); falling back to PyTorch dequantized matmul.", e)

    a_deq = a_fp8.float() * (scale_a[:, None] if scale_a.dim() == 1 else scale_a)
    if scale_b.shape[0] == K:
        b_deq = b_fp8.float() * scale_b[:, None]
    else:
        b_deq = b_fp8.float() * scale_b[None, :]
    return torch.matmul(a_deq, b_deq).to(out_dtype)


def pack_int4_weights(
    weight_fp: torch.Tensor,
    group_size: int = 128,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Packs 2D weight matrix (N, K) into true bit-packed uint8 tensor (N, K // 2),
    along with per-group FP16 scales (N, K // group_size) and zeros (N, K // group_size).

    Cuts weight memory footprint by > 3.4x (saving 71-75% real VRAM).
    """
    orig_shape = weight_fp.shape
    assert len(orig_shape) == 2, f"Expected 2D weight matrix, got shape {orig_shape}"
    N, K = orig_shape

    pad_k = (group_size - (K % group_size)) % group_size
    if pad_k > 0:
        w = torch.nn.functional.pad(weight_fp.float(), (0, pad_k), mode="constant", value=0.0)
    else:
        w = weight_fp.float()

    N, K_padded = w.shape
    num_groups = K_padded // group_size
    w_grouped = w.view(N, num_groups, group_size)

    min_val = w_grouped.amin(dim=-1, keepdim=True)
    max_val = w_grouped.amax(dim=-1, keepdim=True)

    scales = ((max_val - min_val) / 15.0).clamp_min(1e-8)
    zeros = min_val

    q = torch.clamp(torch.round((w_grouped - zeros) / scales), 0, 15).to(torch.uint8)
    q = q.view(N, K_padded)

    q_lo = q[:, 0::2] & 0x0F
    q_hi = (q[:, 1::2] & 0x0F) << 4
    packed = (q_lo | q_hi).contiguous()

    scales = scales.squeeze(-1).to(torch.float16).contiguous()
    zeros = zeros.squeeze(-1).to(torch.float16).contiguous()

    return packed, scales, zeros


def unpack_int4_weights(
    packed: torch.Tensor,
    scales: torch.Tensor,
    zeros: torch.Tensor,
    group_size: int = 128,
    out_dtype: torch.dtype = torch.float16,
) -> torch.Tensor:
    """Unpacks bit-packed uint8 weight back to full-precision floating point."""
    N, K_half = packed.shape
    K_padded = K_half * 2

    q_lo = (packed & 0x0F).to(out_dtype)
    q_hi = ((packed >> 4) & 0x0F).to(out_dtype)

    q = torch.empty((N, K_padded), device=packed.device, dtype=out_dtype)
    q[:, 0::2] = q_lo
    q[:, 1::2] = q_hi

    num_groups = K_padded // group_size
    q_grouped = q.view(N, num_groups, group_size)

    s = scales.to(out_dtype).unsqueeze(-1)
    z = zeros.to(out_dtype).unsqueeze(-1)

    w_rec = (q_grouped * s + z).view(N, K_padded)
    return w_rec


if HAS_TRITON:

    @triton.jit
    def _w4a16_gemm_kernel(
        X_ptr,
        Packed_ptr,
        Scales_ptr,
        Zeros_ptr,
        Out_ptr,
        M,
        N,
        K,
        stride_xm,
        stride_xk,
        stride_pn,
        stride_pk,
        stride_sn,
        stride_sg,
        stride_zn,
        stride_zg,
        stride_om,
        stride_on,
        GROUP_SIZE: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        """
        Fused in-SRAM Dequantization + Tensor Core GEMM.
        Computes Out = X @ W.T where W is stored in packed uint8 format.
        """
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        for k_start in range(0, K, BLOCK_K):
            offs_k = k_start + tl.arange(0, BLOCK_K)
            offs_packed_k = (k_start // 2) + tl.arange(0, BLOCK_K // 2)

            x_ptrs = X_ptr + (offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk)
            mask_x = (offs_m[:, None] < M) & (offs_k[None, :] < K)
            x_tile = tl.load(x_ptrs, mask=mask_x, other=0.0).to(tl.float32)

            p_ptrs = Packed_ptr + (offs_n[:, None] * stride_pn + offs_packed_k[None, :] * stride_pk)
            mask_p = (offs_n[:, None] < N) & (offs_packed_k[None, :] < (K // 2))
            packed_tile = tl.load(p_ptrs, mask=mask_p, other=0)

            lo_nibble = (packed_tile & 0x0F).to(tl.float32)
            hi_nibble = ((packed_tile >> 4) & 0x0F).to(tl.float32)

            group_idx = (k_start + tl.arange(0, BLOCK_K)) // GROUP_SIZE
            s_ptrs = Scales_ptr + (offs_n[:, None] * stride_sn + (group_idx[None, :]) * stride_sg)
            z_ptrs = Zeros_ptr + (offs_n[:, None] * stride_zn + (group_idx[None, :]) * stride_zg)

            s_lo = tl.load(
                Scales_ptr + (offs_n[:, None] * stride_sn + (k_start // GROUP_SIZE) * stride_sg),
                mask=offs_n[:, None] < N,
                other=1.0,
            ).to(tl.float32)
            z_lo = tl.load(
                Zeros_ptr + (offs_n[:, None] * stride_zn + (k_start // GROUP_SIZE) * stride_zg),
                mask=offs_n[:, None] < N,
                other=0.0,
            ).to(tl.float32)

            x_even = x_tile[:, 0::2]
            x_odd = x_tile[:, 1::2]

            w_lo_dequant = lo_nibble * s_lo + z_lo
            w_hi_dequant = hi_nibble * s_lo + z_lo

            acc += tl.dot(x_even, tl.trans(w_lo_dequant))
            acc += tl.dot(x_odd, tl.trans(w_hi_dequant))

        out_ptrs = Out_ptr + (offs_m[:, None] * stride_om + offs_n[None, :] * stride_on)
        mask_out = (offs_m[:, None] < M) & (offs_n[None, :] < N)
        tl.store(out_ptrs, acc.to(tl.float16), mask=mask_out)


else:
    _w4a16_gemm_kernel = None

def fused_w4a16_gemm_triton(
    x: torch.Tensor,
    packed: torch.Tensor,
    scales: torch.Tensor,
    zeros: torch.Tensor,
    group_size: int = 128,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Executes fused W4A16 dequant-GEMM: Y = X @ W.T (+ bias).
    Does NOT materialize the dequantized (N, K) weight matrix in HBM memory.
    """
    orig_shape = x.shape
    M = x.numel() // x.shape[-1]
    K = x.shape[-1]
    N = packed.shape[0]

    x_2d = x.view(M, K).contiguous()
    out_dtype = torch.float16

    if x.is_cuda and HAS_TRITON and M >= 16 and (K % 64 == 0) and (group_size >= 64):
        try:
            out = torch.empty((M, N), device=x.device, dtype=out_dtype)
            BLOCK_M = 32
            BLOCK_N = 32
            BLOCK_K = 64
            grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
            _w4a16_gemm_kernel[grid](
                x_2d,
                packed,
                scales,
                zeros,
                out,
                M,
                N,
                K,
                x_2d.stride(0),
                x_2d.stride(1),
                packed.stride(0),
                packed.stride(1),
                scales.stride(0),
                scales.stride(1),
                zeros.stride(0),
                zeros.stride(1),
                out.stride(0),
                out.stride(1),
                GROUP_SIZE=group_size,
                BLOCK_M=BLOCK_M,
                BLOCK_N=BLOCK_N,
                BLOCK_K=BLOCK_K,
            )
            if bias is not None:
                out += bias
            return out.view(*orig_shape[:-1], N)
        except Exception as e:
            logger.warning("_w4a16_gemm_kernel failed (%s); falling back to PyTorch unpacked matmul.", e)

    w_rec = unpack_int4_weights(packed, scales, zeros, group_size=group_size, out_dtype=x_2d.dtype)[:, :K]
    out = torch.matmul(x_2d, w_rec.t())
    if bias is not None:
        out += bias
    return out.view(*orig_shape[:-1], N)


class BitPackedW4A16Linear(torch.nn.Module):
    """
    Drop-in replacement for nn.Linear with true bit-packed uint8 weight storage.
    Cuts weight memory from 16 bits to ~4.5 bits (including scales/zeros).
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        group_size: int = 128,
        device: torch.device | None = None,
        dtype: torch.dtype = torch.float16,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.group_size = group_size

        pad_in = (group_size - (in_features % group_size)) % group_size
        in_padded = in_features + pad_in
        num_groups = in_padded // group_size

        self.register_buffer("qweight", torch.zeros((out_features, in_padded // 2), dtype=torch.uint8, device=device))
        self.register_buffer("scales", torch.ones((out_features, num_groups), dtype=torch.float16, device=device))
        self.register_buffer("zeros", torch.zeros((out_features, num_groups), dtype=torch.float16, device=device))

        if bias:
            self.bias = torch.nn.Parameter(torch.zeros(out_features, dtype=dtype, device=device))
        else:
            self.register_parameter("bias", None)

    @classmethod
    def from_linear(cls, linear: torch.nn.Module, group_size: int = 128) -> "BitPackedW4A16Linear":
        """Constructs a bit-packed module from an existing nn.Linear or Conv1D."""
        raw_w = linear.weight.data
        if hasattr(linear, "nf"):
            raw_w = raw_w.t().contiguous()
        out_feat, in_feat = raw_w.shape
        has_bias = getattr(linear, "bias", None) is not None

        mod = cls(
            in_features=in_feat,
            out_features=out_feat,
            bias=has_bias,
            group_size=group_size,
            device=raw_w.device,
            dtype=raw_w.dtype if raw_w.dtype in (torch.float16, torch.bfloat16) else torch.float16,
        )
        packed, scales, zeros = pack_int4_weights(raw_w, group_size=group_size)
        mod.qweight.copy_(packed)
        mod.scales.copy_(scales)
        mod.zeros.copy_(zeros)
        if has_bias:
            mod.bias.data.copy_(linear.bias.data.to(mod.scales.dtype))
        return mod

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return fused_w4a16_gemm_triton(
            x,
            self.qweight,
            self.scales,
            self.zeros,
            group_size=self.group_size,
            bias=self.bias,
        )


class BitPackedLowRankSVDLinear(torch.nn.Module):
    """
    SVD Factorization with bit-packed W4A16 tail singular vectors.
    Keeps top singular vectors in exact FP16 and packs the remaining rank into INT4.
    Achieves 6x-8x total weight compression with near-lossless accuracy.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 32,
        exact_rank: int = 8,
        group_size: int = 64,
        bias: bool = True,
        device: torch.device | None = None,
        dtype: torch.dtype = torch.float16,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.exact_rank = min(rank, exact_rank)
        self.tail_rank = rank - self.exact_rank

        if self.exact_rank > 0:
            self.u_exact = torch.nn.Parameter(
                torch.randn(out_features, self.exact_rank, device=device, dtype=dtype) * 0.02
            )
            self.v_exact = torch.nn.Parameter(
                torch.randn(self.exact_rank, in_features, device=device, dtype=dtype) * 0.02
            )
        else:
            self.register_parameter("u_exact", None)
            self.register_parameter("v_exact", None)

        if self.tail_rank > 0:
            self.u_tail_packed = BitPackedW4A16Linear(
                self.tail_rank, out_features, bias=False, group_size=group_size, device=device, dtype=dtype
            )
            self.v_tail_packed = BitPackedW4A16Linear(
                in_features, self.tail_rank, bias=False, group_size=group_size, device=device, dtype=dtype
            )
        else:
            self.u_tail_packed = None
            self.v_tail_packed = None

        if bias:
            self.bias = torch.nn.Parameter(torch.zeros(out_features, device=device, dtype=dtype))
        else:
            self.register_parameter("bias", None)

    @classmethod
    def from_linear(
        cls,
        linear: torch.nn.Module,
        rank: int = 32,
        exact_rank: int = 8,
        group_size: int = 64,
    ) -> "BitPackedLowRankSVDLinear":
        raw_w = linear.weight.data
        if hasattr(linear, "nf"):
            raw_w = raw_w.t().contiguous()
        out_feat, in_feat = raw_w.shape
        has_bias = getattr(linear, "bias", None) is not None

        mod = cls(
            in_features=in_feat,
            out_features=out_feat,
            rank=rank,
            exact_rank=exact_rank,
            group_size=group_size,
            bias=has_bias,
            device=raw_w.device,
            dtype=raw_w.dtype if raw_w.dtype in (torch.float16, torch.bfloat16) else torch.float16,
        )

        U, S, Vh = torch.linalg.svd(raw_w.float(), full_matrices=False)
        r = min(rank, len(S))
        U_r = U[:, :r] * torch.sqrt(S[:r])
        V_r = torch.sqrt(S[:r])[:, None] * Vh[:r, :]

        if mod.exact_rank > 0:
            mod.u_exact.data.copy_(U_r[:, : mod.exact_rank].to(mod.u_exact.dtype))
            mod.v_exact.data.copy_(V_r[: mod.exact_rank, :].to(mod.v_exact.dtype))

        if mod.tail_rank > 0:
            u_tail = U_r[:, mod.exact_rank : r].contiguous()
            v_tail = V_r[mod.exact_rank : r, :].contiguous()

            p_u, s_u, z_u = pack_int4_weights(u_tail, group_size=group_size)
            mod.u_tail_packed.qweight.copy_(p_u)
            mod.u_tail_packed.scales.copy_(s_u)
            mod.u_tail_packed.zeros.copy_(z_u)

            p_v, s_v, z_v = pack_int4_weights(v_tail, group_size=group_size)
            mod.v_tail_packed.qweight.copy_(p_v)
            mod.v_tail_packed.scales.copy_(s_v)
            mod.v_tail_packed.zeros.copy_(z_v)

        if has_bias:
            mod.bias.data.copy_(linear.bias.data.to(mod.bias.dtype))

        return mod

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = 0.0
        if self.exact_rank > 0:
            out = torch.matmul(torch.matmul(x, self.v_exact.t()), self.u_exact.t())
        if self.tail_rank > 0:
            tail_bottleneck = self.v_tail_packed(x)
            out_tail = self.u_tail_packed(tail_bottleneck)
            out = out + out_tail if isinstance(out, torch.Tensor) else out_tail

        if self.bias is not None:
            out = out + self.bias
        return out

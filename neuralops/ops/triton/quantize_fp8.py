#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""FP8 (E4M3 / E5M2) quantization Triton kernel with PyTorch fallback."""

from __future__ import annotations

import logging
import os

import torch

try:
    import triton
    import triton.language as tl
except ImportError:
    triton = None
    tl = None

from neuralops.ops.triton._types import HAS_TRITON, check_triton_suitability

logger = logging.getLogger(__name__)

_INTERPRET_MODE = os.environ.get("TRITON_INTERPRET") == "1"

if HAS_TRITON and triton is not None and tl is not None:
    @triton.jit
    def _fp8_quantize_kernel(
        W_ptr,
        Out_ptr,
        Scale_ptr,
        N: tl.constexpr,
        max_val: tl.constexpr,
        mantissa_step: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        pid = tl.program_id(0)
        offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offs < N
        w = tl.load(W_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        scale = tl.load(Scale_ptr + offs, mask=mask, other=1.0).to(tl.float32)

        w_scaled = tl.clamp(w / scale, -max_val, max_val)
        abs_w = tl.maximum(tl.abs(w_scaled), 1e-8)
        exp = tl.math.floor(tl.math.log2(abs_w))
        base = tl.math.exp2(exp)
        step = base * mantissa_step
        w_fp8 = tl.math.floor(w_scaled / step + 0.5) * step
        deq = w_fp8 * scale
        tl.store(Out_ptr + offs, deq, mask=mask)

    @triton.autotune(
        configs=[
            triton.Config({"BLOCK_SIZE": 256}, num_warps=4),
            triton.Config({"BLOCK_SIZE": 512}, num_warps=4),
            triton.Config({"BLOCK_SIZE": 1024}, num_warps=8),
        ],
        key=["N"],
    )
    @triton.jit
    def _fp8_quantize_kernel_autotuned(
        W_ptr,
        Out_ptr,
        Scale_ptr,
        N: tl.constexpr,
        max_val: tl.constexpr,
        mantissa_step: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        pid = tl.program_id(0)
        offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offs < N
        w = tl.load(W_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        scale = tl.load(Scale_ptr + offs, mask=mask, other=1.0).to(tl.float32)

        w_scaled = tl.clamp(w / scale, -max_val, max_val)
        abs_w = tl.maximum(tl.abs(w_scaled), 1e-8)
        exp = tl.math.floor(tl.math.log2(abs_w))
        base = tl.math.exp2(exp)
        step = base * mantissa_step
        w_fp8 = tl.math.floor(w_scaled / step + 0.5) * step
        deq = w_fp8 * scale
        tl.store(Out_ptr + offs, deq, mask=mask)


def _compute_fp8_scales(
    W: torch.Tensor,
    scale_type: str = "per_tensor",
    max_val: float = 448.0,
) -> torch.Tensor:
    """Computes expanded scale tensor matching FP8Compressor."""
    w_float = W.to(torch.float32)
    if scale_type == "per_channel":
        if w_float.dim() > 1:
            max_abs = w_float.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)
            scale = max_abs / max_val
            return scale.expand_as(W).contiguous()
        max_abs = w_float.abs().amax().clamp(min=1e-8)
        scale = max_abs / max_val
        return scale.expand_as(W).contiguous()

    max_abs = w_float.abs().amax().clamp(min=1e-8)
    scale = max_abs / max_val
    return scale.expand_as(W).contiguous()


def _fp8_pytorch(
    W: torch.Tensor,
    scale_type: str = "per_tensor",
    fp8_format: str = "e4m3",
) -> torch.Tensor:
    """Pure PyTorch reference matching FP8Compressor."""
    max_val = 448.0 if fp8_format == "e4m3" else 57344.0
    mantissa_bits = 3 if fp8_format == "e4m3" else 2
    step_mult = 2.0 ** (-mantissa_bits)

    scale = _compute_fp8_scales(W, scale_type=scale_type, max_val=max_val)
    w_float = W.to(torch.float32)
    w_scaled = (w_float / scale).clamp(-max_val, max_val)

    exp = torch.floor(torch.log2(w_scaled.abs().clamp(min=1e-8)))
    base = torch.pow(2.0, exp)
    step = base * step_mult
    w_fp8 = torch.round(w_scaled / step) * step
    w_dequant = (w_fp8 * scale).to(dtype=W.dtype)
    return w_dequant


def fp8_compressor_compress(
    W: torch.Tensor,
    scale_type: str = "per_tensor",
    fp8_format: str = "e4m3",
) -> torch.Tensor:
    """FP8 quantization (E4M3 / E5M2) with per-tensor or per-channel scaling."""
    if not _INTERPRET_MODE:
        is_suitable, reason = check_triton_suitability(W, "fp8_compressor", allow_fp32=True)
        if not is_suitable:
            logger.warning("fp8 Triton fallback: %s. Using PyTorch reference.", reason)
            return _fp8_pytorch(W, scale_type=scale_type, fp8_format=fp8_format)

    max_val = 448.0 if fp8_format == "e4m3" else 57344.0
    mantissa_step = 0.125 if fp8_format == "e4m3" else 0.25

    scale = _compute_fp8_scales(W, scale_type=scale_type, max_val=max_val)

    W_flat = W.contiguous().reshape(-1).to(torch.float32)
    scales_flat = scale.contiguous().reshape(-1).to(torch.float32)
    N = W_flat.numel()

    out = torch.empty_like(W_flat)

    if _INTERPRET_MODE:
        block_size = 256
        grid = (triton.cdiv(N, block_size),)
        _fp8_quantize_kernel[grid](
            W_flat,
            out,
            scales_flat,
            N,
            max_val,
            mantissa_step,
            BLOCK_SIZE=block_size,
        )
    else:
        grid = lambda meta: (triton.cdiv(N, meta["BLOCK_SIZE"]),)
        _fp8_quantize_kernel_autotuned[grid](
            W_flat,
            out,
            scales_flat,
            N,
            max_val,
            mantissa_step,
        )

    return out.reshape(W.shape).to(dtype=W.dtype)


fp8_compressor_compress_pytorch = _fp8_pytorch
fp8_compress = fp8_compressor_compress

__all__ = ["fp8_compress", "fp8_compressor_compress", "fp8_compressor_compress_pytorch"]

#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Flash-Decoding Attention

Accelerates autoregressive decoding via split-KV sequence chunking and multi-block parallel reduction.
"""

from __future__ import annotations
import math
import torch
import torch.nn.functional as F

from neuralops.core.triton_kernels._common import HAS_TRITON, triton, tl

__all__ = ["flash_decoding_attention", "flash_decoding_pytorch", "flash_decode_kernel"]


if HAS_TRITON and tl is not None:
    @triton.jit
    def flash_decoding_stage1_kernel(
        Q_ptr, K_ptr, V_ptr, MidOut_ptr, MidLse_ptr,
        stride_qb, stride_qh, stride_qd,
        stride_kb, stride_kh, stride_kn, stride_kd,
        stride_vb, stride_vh, stride_vn, stride_vd,
        stride_mb, stride_mh, stride_mc, stride_md,
        stride_lb, stride_lh, stride_lc,
        H, N_kv,
        CHUNK_SIZE: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        """Split-KV Flash Decoding Stage 1: Parallel online softmax over KV chunks in SRAM."""
        pid_c = tl.program_id(0)
        pid_bh = tl.program_id(1)
        i_b = pid_bh // H
        i_h = pid_bh % H

        offs_d = tl.arange(0, BLOCK_D)
        q_ptrs = Q_ptr + i_b * stride_qb + i_h * stride_qh + offs_d * stride_qd
        q = tl.load(q_ptrs).to(tl.float32)

        scale = 1.0 / tl.sqrt(tl.cast(BLOCK_D, tl.float32))

        start_n = pid_c * CHUNK_SIZE
        offs_n = start_n + tl.arange(0, CHUNK_SIZE)
        mask_n = offs_n < N_kv

        k_ptrs = K_ptr + i_b * stride_kb + i_h * stride_kh + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kd
        k = tl.load(k_ptrs, mask=mask_n[:, None], other=0.0).to(tl.float32)

        scores = tl.sum(q[None, :] * k, axis=1) * scale
        scores = tl.where(mask_n, scores, -1e4)

        m_c = tl.max(scores, axis=0)
        p = tl.exp(scores - m_c)
        l_c = tl.sum(p, axis=0)

        v_ptrs = V_ptr + i_b * stride_vb + i_h * stride_vh + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vd
        v = tl.load(v_ptrs, mask=mask_n[:, None], other=0.0).to(tl.float32)

        out_chunk = tl.sum(p[:, None] * v, axis=0) / tl.maximum(l_c, 1e-6)

        lse_c = m_c + tl.log(tl.maximum(l_c, 1e-6))

        out_ptrs = MidOut_ptr + i_b * stride_mb + i_h * stride_mh + pid_c * stride_mc + offs_d * stride_md
        tl.store(out_ptrs, out_chunk.to(MidOut_ptr.dtype.element_ty))

        lse_ptrs = MidLse_ptr + i_b * stride_lb + i_h * stride_lh + pid_c * stride_lc
        tl.store(lse_ptrs, lse_c)


def flash_decoding_pytorch(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    chunk_size: int = 128,
) -> torch.Tensor:
    """Split-KV Flash-Decoding reference implementation."""
    B, H, N_q, D = q.shape
    _, _, N_kv, _ = k.shape
    scale = 1.0 / math.sqrt(D)

    if N_kv <= chunk_size:
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale
        weights = F.softmax(scores, dim=-1)
        return torch.matmul(weights, v)

    num_chunks = (N_kv + chunk_size - 1) // chunk_size
    chunk_outputs = []
    chunk_logsumexps = []

    for i in range(num_chunks):
        start = i * chunk_size
        end = min((i + 1) * chunk_size, N_kv)
        k_chunk = k[:, :, start:end, :]
        v_chunk = v[:, :, start:end, :]

        scores = torch.matmul(q, k_chunk.transpose(-2, -1)) * scale
        lse = torch.logsumexp(scores, dim=-1, keepdim=True)
        weights = torch.exp(scores - lse)
        out_chunk = torch.matmul(weights, v_chunk)

        chunk_outputs.append(out_chunk)
        chunk_logsumexps.append(lse)

    all_lse = torch.cat(chunk_logsumexps, dim=-1)
    global_lse = torch.logsumexp(all_lse, dim=-1, keepdim=True)

    final_out = torch.zeros_like(q)
    for out_chunk, lse_chunk in zip(chunk_outputs, chunk_logsumexps):
        alpha = torch.exp(lse_chunk - global_lse)
        final_out += alpha * out_chunk

    return final_out


if HAS_TRITON and tl is not None:
    @triton.jit
    def flash_decoding_stage2_kernel(
        MidOut_ptr, MidLse_ptr, FinalOut_ptr,
        stride_mb, stride_mh, stride_mc, stride_md,
        stride_lb, stride_lh, stride_lc,
        stride_ob, stride_oh, stride_oq, stride_od,
        H, num_chunks,
        BLOCK_C: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        """Split-KV Flash Decoding Stage 2: Reduction over intermediate chunks in SRAM."""
        pid_bh = tl.program_id(0)
        i_b = pid_bh // H
        i_h = pid_bh % H

        offs_d = tl.arange(0, BLOCK_D)
        offs_c = tl.arange(0, BLOCK_C)
        mask_c = offs_c < num_chunks

        lse_ptrs = MidLse_ptr + i_b * stride_lb + i_h * stride_lh + offs_c * stride_lc
        lse_vals = tl.load(lse_ptrs, mask=mask_c, other=-1e4)

        max_lse = tl.max(lse_vals, axis=0)
        p = tl.where(mask_c, tl.exp(lse_vals - max_lse), 0.0)
        global_sum = tl.sum(p, axis=0)
        _alpha = p / tl.maximum(global_sum, 1e-6)

        acc = tl.zeros([BLOCK_D], dtype=tl.float32)
        for c in range(num_chunks):
            w_c = tl.load(MidLse_ptr + i_b * stride_lb + i_h * stride_lh + c * stride_lc)
            weight = tl.exp(w_c - max_lse) / tl.maximum(global_sum, 1e-6)

            mid_ptrs = MidOut_ptr + i_b * stride_mb + i_h * stride_mh + c * stride_mc + offs_d * stride_md
            out_c = tl.load(mid_ptrs).to(tl.float32)
            acc += weight * out_c

        out_ptrs = FinalOut_ptr + i_b * stride_ob + i_h * stride_oh + 0 * stride_oq + offs_d * stride_od
        tl.store(out_ptrs, acc.to(FinalOut_ptr.dtype.element_ty))
else:
    flash_decoding_stage1_kernel = None  # noqa: F811
    flash_decoding_stage2_kernel = None  # noqa: F811


def flash_decoding_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    chunk_size: int = 128,
) -> torch.Tensor:
    """Dispatches split-KV flash decoding with Triton hardware acceleration."""
    from neuralops.core.triton_kernels._types import HAS_TRITON, FORCE_PYTORCH_FALLBACK, check_triton_suitability
    B, H, N_q, D = q.shape
    N_kv = k.shape[-2]

    if HAS_TRITON and not FORCE_PYTORCH_FALLBACK and q.is_cuda and N_q == 1 and D in (16, 32, 64, 128, 256):
        try:
            num_chunks = triton.cdiv(N_kv, chunk_size)
            mid_out = torch.empty((B, H, num_chunks, D), dtype=q.dtype, device=q.device)
            mid_lse = torch.empty((B, H, num_chunks), dtype=torch.float32, device=q.device)

            grid1 = (num_chunks, B * H)
            flash_decoding_stage1_kernel[grid1](
                q.squeeze(2), k, v, mid_out, mid_lse,
                q.squeeze(2).stride(0), q.squeeze(2).stride(1), q.squeeze(2).stride(2),
                k.stride(0), k.stride(1), k.stride(2), k.stride(3),
                v.stride(0), v.stride(1), v.stride(2), v.stride(3),
                mid_out.stride(0), mid_out.stride(1), mid_out.stride(2), mid_out.stride(3),
                mid_lse.stride(0), mid_lse.stride(1), mid_lse.stride(2),
                H, N_kv,
                CHUNK_SIZE=chunk_size,
                BLOCK_D=D,
            )

            final_out = torch.empty((B, H, 1, D), dtype=q.dtype, device=q.device)
            block_c = triton.next_power_of_2(max(16, num_chunks))
            grid2 = (B * H,)
            flash_decoding_stage2_kernel[grid2](
                mid_out, mid_lse, final_out,
                mid_out.stride(0), mid_out.stride(1), mid_out.stride(2), mid_out.stride(3),
                mid_lse.stride(0), mid_lse.stride(1), mid_lse.stride(2),
                final_out.stride(0), final_out.stride(1), final_out.stride(2), final_out.stride(3),
                H, num_chunks,
                BLOCK_C=block_c,
                BLOCK_D=D,
            )
            return final_out
        except Exception:
            pass

    return flash_decoding_pytorch(q, k, v, chunk_size=chunk_size)


flash_decode_kernel = flash_decoding_attention

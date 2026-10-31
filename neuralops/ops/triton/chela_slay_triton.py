#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""CHELA-SLAY Spherical Linearized Attention Triton kernel with PyTorch fallback."""

from __future__ import annotations

import logging
import os
from typing import Any

import torch
import torch.nn.functional as F

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
    def _chela_slay_chunk_kernel(
        Q_ptr,
        K_ptr,
        V_ptr,
        Out_ptr,
        stride_qb,
        stride_qh,
        stride_ql,
        stride_qd,
        stride_kb,
        stride_kh,
        stride_kl,
        stride_kd,
        stride_vb,
        stride_vh,
        stride_vl,
        stride_vd,
        stride_ob,
        stride_oh,
        stride_ol,
        stride_od,
        B,
        H,
        L,
        D: tl.constexpr,
        BLOCK_C: tl.constexpr = 64,
    ):
        """Chunked causal linear attention kernel."""
        bh_idx = tl.program_id(0)
        b_idx = bh_idx // H
        h_idx = bh_idx % H

        q_base = Q_ptr + b_idx * stride_qb + h_idx * stride_qh
        k_base = K_ptr + b_idx * stride_kb + h_idx * stride_kh
        v_base = V_ptr + b_idx * stride_vb + h_idx * stride_vh
        o_base = Out_ptr + b_idx * stride_ob + h_idx * stride_oh

        offs_c = tl.arange(0, BLOCK_C)
        offs_d = tl.arange(0, D)

        mask_d = offs_d < D

        S = tl.zeros((D, D), dtype=tl.float32)
        Z = tl.zeros((D,), dtype=tl.float32)

        num_chunks = (L + BLOCK_C - 1) // BLOCK_C

        for c in range(num_chunks):
            start_l = c * BLOCK_C
            cur_offs_l = start_l + offs_c
            mask_l = cur_offs_l < L

            q_ptrs = q_base + cur_offs_l[:, None] * stride_ql + offs_d[None, :] * stride_qd
            k_ptrs = k_base + cur_offs_l[:, None] * stride_kl + offs_d[None, :] * stride_kd
            v_ptrs = v_base + cur_offs_l[:, None] * stride_vl + offs_d[None, :] * stride_vd

            q_tile = tl.load(q_ptrs, mask=(mask_l[:, None] & mask_d[None, :]), other=0.0).to(tl.float32)
            k_tile = tl.load(k_ptrs, mask=(mask_l[:, None] & mask_d[None, :]), other=0.0).to(tl.float32)
            v_tile = tl.load(v_ptrs, mask=(mask_l[:, None] & mask_d[None, :]), other=0.0).to(tl.float32)

            scores = tl.dot(q_tile, tl.trans(k_tile))
            causal_mask = offs_c[:, None] >= offs_c[None, :]
            valid_mask = mask_l[:, None] & mask_l[None, :]
            scores_masked = tl.where(causal_mask & valid_mask, scores, 0.0)

            num_intra = tl.dot(scores_masked, v_tile)
            denom_intra = tl.sum(scores_masked, axis=1)

            num_inter = tl.dot(q_tile, S)
            denom_inter = tl.sum(q_tile * Z[None, :], axis=1)

            num_tot = num_intra + num_inter
            denom_tot = denom_intra + denom_inter
            denom_clamped = tl.maximum(tl.abs(denom_tot), 1.0)

            out_tile = num_tot / denom_clamped[:, None]

            o_ptrs = o_base + cur_offs_l[:, None] * stride_ol + offs_d[None, :] * stride_od
            tl.store(o_ptrs, out_tile, mask=(mask_l[:, None] & mask_d[None, :]))

            S += tl.dot(tl.trans(k_tile), v_tile)
            Z += tl.sum(k_tile, axis=0)

    @triton.autotune(
        configs=[
            triton.Config({"BLOCK_C": 32}, num_warps=4),
            triton.Config({"BLOCK_C": 64}, num_warps=4),
            triton.Config({"BLOCK_C": 64}, num_warps=8),
        ],
        key=["L", "D"],
    )
    @triton.jit
    def _chela_slay_chunk_kernel_autotuned(
        Q_ptr,
        K_ptr,
        V_ptr,
        Out_ptr,
        stride_qb,
        stride_qh,
        stride_ql,
        stride_qd,
        stride_kb,
        stride_kh,
        stride_kl,
        stride_kd,
        stride_vb,
        stride_vh,
        stride_vl,
        stride_vd,
        stride_ob,
        stride_oh,
        stride_ol,
        stride_od,
        B,
        H,
        L,
        D: tl.constexpr,
        BLOCK_C: tl.constexpr = 64,
    ):
        bh_idx = tl.program_id(0)
        b_idx = bh_idx // H
        h_idx = bh_idx % H

        q_base = Q_ptr + b_idx * stride_qb + h_idx * stride_qh
        k_base = K_ptr + b_idx * stride_kb + h_idx * stride_kh
        v_base = V_ptr + b_idx * stride_vb + h_idx * stride_vh
        o_base = Out_ptr + b_idx * stride_ob + h_idx * stride_oh

        offs_c = tl.arange(0, BLOCK_C)
        offs_d = tl.arange(0, D)

        mask_d = offs_d < D

        S = tl.zeros((D, D), dtype=tl.float32)
        Z = tl.zeros((D,), dtype=tl.float32)

        num_chunks = (L + BLOCK_C - 1) // BLOCK_C

        for c in range(num_chunks):
            start_l = c * BLOCK_C
            cur_offs_l = start_l + offs_c
            mask_l = cur_offs_l < L

            q_ptrs = q_base + cur_offs_l[:, None] * stride_ql + offs_d[None, :] * stride_qd
            k_ptrs = k_base + cur_offs_l[:, None] * stride_kl + offs_d[None, :] * stride_kd
            v_ptrs = v_base + cur_offs_l[:, None] * stride_vl + offs_d[None, :] * stride_vd

            q_tile = tl.load(q_ptrs, mask=(mask_l[:, None] & mask_d[None, :]), other=0.0).to(tl.float32)
            k_tile = tl.load(k_ptrs, mask=(mask_l[:, None] & mask_d[None, :]), other=0.0).to(tl.float32)
            v_tile = tl.load(v_ptrs, mask=(mask_l[:, None] & mask_d[None, :]), other=0.0).to(tl.float32)

            scores = tl.dot(q_tile, tl.trans(k_tile))
            causal_mask = offs_c[:, None] >= offs_c[None, :]
            valid_mask = mask_l[:, None] & mask_l[None, :]
            scores_masked = tl.where(causal_mask & valid_mask, scores, 0.0)

            num_intra = tl.dot(scores_masked, v_tile)
            denom_intra = tl.sum(scores_masked, axis=1)

            num_inter = tl.dot(q_tile, S)
            denom_inter = tl.sum(q_tile * Z[None, :], axis=1)

            num_tot = num_intra + num_inter
            denom_tot = denom_intra + denom_inter
            denom_clamped = tl.maximum(tl.abs(denom_tot), 1.0)

            out_tile = num_tot / denom_clamped[:, None]

            o_ptrs = o_base + cur_offs_l[:, None] * stride_ol + offs_d[None, :] * stride_od
            tl.store(o_ptrs, out_tile, mask=(mask_l[:, None] & mask_d[None, :]))

            S += tl.dot(tl.trans(k_tile), v_tile)
            Z += tl.sum(k_tile, axis=0)


def _chela_slay_pytorch(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    chunk_size: int = 64,
    **kwargs: Any,
) -> torch.Tensor:
    """Vectorized chunkwise parallel scan matching CHELA-SLAY causal attention."""
    B, H, L, D = q.shape
    num_chunks = (L + chunk_size - 1) // chunk_size
    pad_len = num_chunks * chunk_size - L

    if pad_len > 0:
        q_pad = F.pad(q, (0, 0, 0, pad_len))
        k_pad = F.pad(k, (0, 0, 0, pad_len))
        v_pad = F.pad(v, (0, 0, 0, pad_len))
    else:
        q_pad, k_pad, v_pad = q, k, v

    q_c = q_pad.view(B, H, num_chunks, chunk_size, D).float()
    k_c = k_pad.view(B, H, num_chunks, chunk_size, D).float()
    v_c = v_pad.view(B, H, num_chunks, chunk_size, D).float()

    if pad_len > 0:
        valid_mask = torch.ones(num_chunks, chunk_size, device=q.device, dtype=torch.float32)
        valid_mask[-1, chunk_size - pad_len:] = 0.0
        v_m = valid_mask.view(1, 1, num_chunks, chunk_size, 1)
        q_c = q_c * v_m
        k_c = k_c * v_m
        v_c = v_c * v_m

    scores = torch.matmul(q_c, k_c.transpose(-2, -1))
    mask = torch.triu(torch.ones(chunk_size, chunk_size, device=q.device, dtype=torch.bool), diagonal=1)
    scores = scores.masked_fill(mask, 0.0)

    num_intra = torch.matmul(scores, v_c)
    denom_intra = scores.sum(dim=-1, keepdim=True)

    delta_s = torch.matmul(k_c.transpose(-2, -1), v_c)
    delta_z = k_c.sum(dim=-2)

    s_cumsum = torch.cumsum(delta_s, dim=2)
    s_past = torch.cat([torch.zeros_like(delta_s[:, :, :1]), s_cumsum[:, :, :-1]], dim=2)

    z_cumsum = torch.cumsum(delta_z, dim=2)
    z_past = torch.cat([torch.zeros_like(delta_z[:, :, :1]), z_cumsum[:, :, :-1]], dim=2)

    num_inter = torch.matmul(q_c, s_past)
    denom_inter = (q_c * z_past.unsqueeze(-2)).sum(dim=-1, keepdim=True)

    denom = (denom_intra + denom_inter).abs().clamp_min(1.0)
    out = (num_intra + num_inter) / denom
    out = out.view(B, H, num_chunks * chunk_size, D)[:, :, :L, :]
    return out.to(q.dtype)


def chela_slay_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    chunk_size: int = 64,
    **kwargs: Any,
) -> torch.Tensor:
    """CHELA-SLAY causal linear attention forward operator.

    Args:
        q: Query tensor of shape (B, H, L, D).
        k: Key tensor of shape (B, H, L, D).
        v: Value tensor of shape (B, H, L, D).
        chunk_size: Tile chunk size (default 64).

    Returns:
        Attention output tensor of shape (B, H, L, D).
    """
    if not _INTERPRET_MODE:
        is_suitable, reason = check_triton_suitability(q, "chela_slay", allow_fp32=True)
        if not is_suitable:
            logger.warning("chela_slay Triton fallback: %s. Using PyTorch reference.", reason)
            return _chela_slay_pytorch(q, k, v, chunk_size=chunk_size, **kwargs)

    B, H, L, D = q.shape
    device = q.device
    dtype = q.dtype

    Q_f = q.contiguous().float()
    K_f = k.contiguous().float()
    V_f = v.contiguous().float()
    out = torch.empty(B, H, L, D, device=device, dtype=dtype)

    grid = (B * H,)

    if _INTERPRET_MODE:
        _chela_slay_chunk_kernel[grid](
            Q_f,
            K_f,
            V_f,
            out,
            Q_f.stride(0),
            Q_f.stride(1),
            Q_f.stride(2),
            Q_f.stride(3),
            K_f.stride(0),
            K_f.stride(1),
            K_f.stride(2),
            K_f.stride(3),
            V_f.stride(0),
            V_f.stride(1),
            V_f.stride(2),
            V_f.stride(3),
            out.stride(0),
            out.stride(1),
            out.stride(2),
            out.stride(3),
            B=B,
            H=H,
            L=L,
            D=D,
            BLOCK_C=min(chunk_size, 64),
        )
    else:
        _chela_slay_chunk_kernel_autotuned[grid](
            Q_f,
            K_f,
            V_f,
            out,
            Q_f.stride(0),
            Q_f.stride(1),
            Q_f.stride(2),
            Q_f.stride(3),
            K_f.stride(0),
            K_f.stride(1),
            K_f.stride(2),
            K_f.stride(3),
            V_f.stride(0),
            V_f.stride(1),
            V_f.stride(2),
            V_f.stride(3),
            out.stride(0),
            out.stride(1),
            out.stride(2),
            out.stride(3),
            B=B,
            H=H,
            L=L,
            D=D,
        )

    return out


__all__ = ["chela_slay_forward", "_chela_slay_pytorch"]

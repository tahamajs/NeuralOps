#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""Locality-Sensitive Hashing (LSH) Attention Triton kernel with PyTorch fallback."""

from __future__ import annotations

import logging
import math
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
    def _lsh_attn_kernel(
        Q_ptr,
        K_ptr,
        V_ptr,
        Out_ptr,
        Bucket_q_ptr,
        Bucket_k_ptr,
        Mask_ptr,
        stride_qm,
        stride_qd,
        stride_kn,
        stride_kd,
        stride_vn,
        stride_vd,
        stride_om,
        stride_od,
        M_q: tl.constexpr,
        N_k: tl.constexpr,
        D: tl.constexpr,
        scale: tl.constexpr,
    ):
        """Triton kernel computing bucketed LSH attention for a single batch/head."""
        m_idx = tl.program_id(0)
        d_offs = tl.arange(0, D)

        q = tl.load(Q_ptr + m_idx * stride_qm + d_offs * stride_qd).to(tl.float32)
        b_q = tl.load(Bucket_q_ptr + m_idx).to(tl.int64)

        max_score = -1e9
        sum_exp = 0.0

        # First pass: find max score among matching bucket keys
        for n in range(N_k):
            b_k = tl.load(Bucket_k_ptr + n).to(tl.int64)
            match = b_q == b_k

            valid = 1.0
            if Mask_ptr is not None:
                valid = tl.load(Mask_ptr + n).to(tl.float32)

            k = tl.load(K_ptr + n * stride_kn + d_offs * stride_kd).to(tl.float32)
            dot = tl.sum(q * k) * scale

            if (match == 0) or (valid <= 0.0):
                dot = -1e4

            if dot > max_score:
                max_score = dot

        # Second pass: compute softmax weights and accumulate values
        acc = tl.zeros((D,), dtype=tl.float32)
        for n in range(N_k):
            b_k = tl.load(Bucket_k_ptr + n).to(tl.int64)
            match = b_q == b_k

            valid = 1.0
            if Mask_ptr is not None:
                valid = tl.load(Mask_ptr + n).to(tl.float32)

            k = tl.load(K_ptr + n * stride_kn + d_offs * stride_kd).to(tl.float32)
            dot = tl.sum(q * k) * scale

            if (match == 0) or (valid <= 0.0):
                dot = -1e4

            p = tl.math.exp2(1.44269504 * (dot - max_score))
            sum_exp += p
            v = tl.load(V_ptr + n * stride_vn + d_offs * stride_vd).to(tl.float32)
            acc += p * v

        if sum_exp > 0.0:
            out = acc / sum_exp
        else:
            out = tl.zeros((D,), dtype=tl.float32)

        tl.store(Out_ptr + m_idx * stride_om + d_offs * stride_od, out)

    @triton.autotune(
        configs=[
            triton.Config({}, num_warps=4),
            triton.Config({}, num_warps=8),
        ],
        key=["M_q", "N_k"],
    )
    @triton.jit
    def _lsh_attn_kernel_autotuned(
        Q_ptr,
        K_ptr,
        V_ptr,
        Out_ptr,
        Bucket_q_ptr,
        Bucket_k_ptr,
        Mask_ptr,
        stride_qm,
        stride_qd,
        stride_kn,
        stride_kd,
        stride_vn,
        stride_vd,
        stride_om,
        stride_od,
        M_q: tl.constexpr,
        N_k: tl.constexpr,
        D: tl.constexpr,
        scale: tl.constexpr,
    ):
        m_idx = tl.program_id(0)
        d_offs = tl.arange(0, D)

        q = tl.load(Q_ptr + m_idx * stride_qm + d_offs * stride_qd).to(tl.float32)
        b_q = tl.load(Bucket_q_ptr + m_idx).to(tl.int64)

        max_score = -1e9
        sum_exp = 0.0

        for n in range(N_k):
            b_k = tl.load(Bucket_k_ptr + n).to(tl.int64)
            match = b_q == b_k

            valid = 1.0
            if Mask_ptr is not None:
                valid = tl.load(Mask_ptr + n).to(tl.float32)

            k = tl.load(K_ptr + n * stride_kn + d_offs * stride_kd).to(tl.float32)
            dot = tl.sum(q * k) * scale

            if (match == 0) or (valid <= 0.0):
                dot = -1e4

            if dot > max_score:
                max_score = dot

        acc = tl.zeros((D,), dtype=tl.float32)
        for n in range(N_k):
            b_k = tl.load(Bucket_k_ptr + n).to(tl.int64)
            match = b_q == b_k

            valid = 1.0
            if Mask_ptr is not None:
                valid = tl.load(Mask_ptr + n).to(tl.float32)

            k = tl.load(K_ptr + n * stride_kn + d_offs * stride_kd).to(tl.float32)
            dot = tl.sum(q * k) * scale

            if (match == 0) or (valid <= 0.0):
                dot = -1e4

            p = tl.math.exp2(1.44269504 * (dot - max_score))
            sum_exp += p
            v = tl.load(V_ptr + n * stride_vn + d_offs * stride_vd).to(tl.float32)
            acc += p * v

        if sum_exp > 0.0:
            out = acc / sum_exp
        else:
            out = tl.zeros((D,), dtype=tl.float32)

        tl.store(Out_ptr + m_idx * stride_om + d_offs * stride_od, out)


def _compute_lsh_buckets(vectors: torch.Tensor, proj: torch.Tensor, num_buckets: int) -> torch.Tensor:
    """Computes integer bucket indices using random hyperplane projections."""
    norm = F.normalize(vectors.float(), p=2, dim=-1)
    dots = torch.matmul(norm, proj.float())
    signs = (dots >= 0).long()
    num_planes = signs.shape[-1]
    weights = (2 ** torch.arange(num_planes, device=signs.device)).long()
    bucket = (signs * weights).sum(dim=-1)
    return bucket % num_buckets


def _lsh_attention_pytorch(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: torch.Tensor | None = None,
    rounds: int = 4,
    num_buckets: int = 16,
    proj_matrices: torch.Tensor | None = None,
    **kwargs: Any,
) -> torch.Tensor:
    """Pure PyTorch reference implementation matching LSHAttention."""
    B, H, N_q, d = q.shape
    N_k = k.shape[2]
    device = q.device
    dtype = q.dtype
    scale = 1.0 / math.sqrt(int(d))

    num_planes = max(1, int(math.ceil(math.log2(max(2, num_buckets)))))
    if proj_matrices is None:
        generator = torch.Generator(device=device).manual_seed(0xCAFE)
        proj_matrices = torch.randn(rounds, d, num_planes, device=device, dtype=dtype, generator=generator)

    rounds_eff = min(rounds, proj_matrices.size(0))
    accum = torch.zeros_like(q)

    for r in range(rounds_eff):
        proj_r = proj_matrices[r].to(device=device, dtype=dtype)
        q_hash = _compute_lsh_buckets(q, proj_r, num_buckets)
        k_hash = _compute_lsh_buckets(k, proj_r, num_buckets)

        bucket_match = q_hash.unsqueeze(-1) == k_hash.unsqueeze(-2)

        if mask is not None:
            if mask.dim() == 2:
                kv_mask = mask[:, None, None, :]
            else:
                kv_mask = mask
            bucket_match = bucket_match & (kv_mask.bool())

        scores = torch.matmul(q.float(), k.float().transpose(-2, -1).contiguous()) * scale
        scores = scores.masked_fill(~bucket_match, -1e4)
        scores = torch.nan_to_num(scores, nan=-1e4, posinf=1e4, neginf=-1e4)

        attn = F.softmax(scores, dim=-1).to(dtype)
        attn = torch.nan_to_num(attn, nan=0.0)
        accum = accum + torch.matmul(attn, v)

    return accum / float(rounds_eff)


def lsh_attention_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: torch.Tensor | None = None,
    rounds: int = 4,
    num_buckets: int = 16,
    proj_matrices: torch.Tensor | None = None,
    **kwargs: Any,
) -> torch.Tensor:
    """LSH Attention forward operator using Triton kernel with PyTorch fallback.

    Args:
        q: Query tensor of shape (B, H, N_q, d).
        k: Key tensor of shape (B, H, N_k, d).
        v: Value tensor of shape (B, H, N_k, d).
        mask: Optional attention mask.
        rounds: Number of LSH hashing rounds.
        num_buckets: Number of hash buckets.
        proj_matrices: Optional pre-generated projection matrices (rounds, d, num_planes).

    Returns:
        Attention output tensor of shape (B, H, N_q, d).
    """
    if not _INTERPRET_MODE:
        is_suitable, reason = check_triton_suitability(q, "lsh_attention", allow_fp32=True)
        if not is_suitable:
            logger.warning("lsh_attention Triton fallback: %s. Using PyTorch reference.", reason)
            return _lsh_attention_pytorch(
                q, k, v, mask=mask, rounds=rounds, num_buckets=num_buckets, proj_matrices=proj_matrices, **kwargs
            )

    B, H, N_q, d = q.shape
    N_k = k.shape[2]
    device = q.device
    dtype = q.dtype
    scale = 1.0 / math.sqrt(int(d))

    num_planes = max(1, int(math.ceil(math.log2(max(2, num_buckets)))))
    if proj_matrices is None:
        generator = torch.Generator(device=device).manual_seed(0xCAFE)
        proj_matrices = torch.randn(rounds, d, num_planes, device=device, dtype=dtype, generator=generator)

    rounds_eff = min(rounds, proj_matrices.size(0))
    accum = torch.zeros_like(q)

    mask_f = mask.float() if mask is not None else None

    for r in range(rounds_eff):
        proj_r = proj_matrices[r].to(device=device, dtype=dtype)
        q_hash = _compute_lsh_buckets(q, proj_r, num_buckets)
        k_hash = _compute_lsh_buckets(k, proj_r, num_buckets)

        round_out = torch.empty_like(q)

        for b in range(B):
            for h in range(H):
                q_slice = q[b, h].contiguous().float()
                k_slice = k[b, h].contiguous().float()
                v_slice = v[b, h].contiguous().float()
                bq_slice = q_hash[b, h].contiguous().to(torch.int64)
                bk_slice = k_hash[b, h].contiguous().to(torch.int64)
                out_slice = round_out[b, h]
                m_slice = mask_f[b] if mask_f is not None else None

                if _INTERPRET_MODE:
                    grid = (N_q,)
                    _lsh_attn_kernel[grid](
                        q_slice,
                        k_slice,
                        v_slice,
                        out_slice,
                        bq_slice,
                        bk_slice,
                        m_slice,
                        q_slice.stride(0),
                        q_slice.stride(1),
                        k_slice.stride(0),
                        k_slice.stride(1),
                        v_slice.stride(0),
                        v_slice.stride(1),
                        out_slice.stride(0),
                        out_slice.stride(1),
                        M_q=N_q,
                        N_k=N_k,
                        D=d,
                        scale=scale,
                    )
                else:
                    grid = lambda meta: (N_q,)
                    _lsh_attn_kernel_autotuned[grid](
                        q_slice,
                        k_slice,
                        v_slice,
                        out_slice,
                        bq_slice,
                        bk_slice,
                        m_slice,
                        q_slice.stride(0),
                        q_slice.stride(1),
                        k_slice.stride(0),
                        k_slice.stride(1),
                        v_slice.stride(0),
                        v_slice.stride(1),
                        out_slice.stride(0),
                        out_slice.stride(1),
                        M_q=N_q,
                        N_k=N_k,
                        D=d,
                        scale=scale,
                    )

        accum = accum + round_out

    return (accum / float(rounds_eff)).to(dtype)


lsh_attention_forward_pytorch = _lsh_attention_pytorch

__all__ = ["lsh_attention_forward", "lsh_attention_forward_pytorch", "_lsh_attention_pytorch"]

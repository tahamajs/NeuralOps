#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""Routing Attention: Content-Based Clustering Sparse Attention Triton kernel with PyTorch fallback."""

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
    def _cluster_attn_kernel(
        Q_ptr,
        K_ptr,
        V_ptr,
        Out_ptr,
        Mask_ptr,
        stride_qm,
        stride_qd,
        stride_kn,
        stride_kd,
        stride_vn,
        stride_vd,
        stride_om,
        stride_od,
        stride_mm,
        stride_mn,
        M_q: tl.constexpr,
        M_k: tl.constexpr,
        D: tl.constexpr,
        scale: tl.constexpr,
    ):
        """Triton kernel computing attention within a cluster for a single batch/head."""
        m_idx = tl.program_id(0)
        d_offs = tl.arange(0, D)

        q = tl.load(Q_ptr + m_idx * stride_qm + d_offs * stride_qd).to(tl.float32)

        # Compute dot products with all keys in cluster
        max_score = -1e9
        sum_exp = 0.0

        for n in range(M_k):
            valid = 1.0
            if Mask_ptr is not None:
                valid = tl.load(Mask_ptr + m_idx * stride_mm + n * stride_mn).to(tl.float32)

            k = tl.load(K_ptr + n * stride_kn + d_offs * stride_kd).to(tl.float32)
            dot = tl.sum(q * k) * scale
            if valid <= 0.0:
                dot = -1e4

            if dot > max_score:
                max_score = dot

        # Second pass: compute softmax denominator and weighted sum of values
        acc = tl.zeros((D,), dtype=tl.float32)
        for n in range(M_k):
            valid = 1.0
            if Mask_ptr is not None:
                valid = tl.load(Mask_ptr + m_idx * stride_mm + n * stride_mn).to(tl.float32)

            k = tl.load(K_ptr + n * stride_kn + d_offs * stride_kd).to(tl.float32)
            dot = tl.sum(q * k) * scale
            if valid <= 0.0:
                dot = -1e4

            p = tl.exp(dot - max_score)
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
        key=["M_q", "M_k"],
    )
    @triton.jit
    def _cluster_attn_kernel_autotuned(
        Q_ptr,
        K_ptr,
        V_ptr,
        Out_ptr,
        Mask_ptr,
        stride_qm,
        stride_qd,
        stride_kn,
        stride_kd,
        stride_vn,
        stride_vd,
        stride_om,
        stride_od,
        stride_mm,
        stride_mn,
        M_q: tl.constexpr,
        M_k: tl.constexpr,
        D: tl.constexpr,
        scale: tl.constexpr,
    ):
        m_idx = tl.program_id(0)
        d_offs = tl.arange(0, D)

        q = tl.load(Q_ptr + m_idx * stride_qm + d_offs * stride_qd).to(tl.float32)

        max_score = -1e9
        sum_exp = 0.0

        for n in range(M_k):
            valid = 1.0
            if Mask_ptr is not None:
                valid = tl.load(Mask_ptr + m_idx * stride_mm + n * stride_mn).to(tl.float32)

            k = tl.load(K_ptr + n * stride_kn + d_offs * stride_kd).to(tl.float32)
            dot = tl.sum(q * k) * scale
            if valid <= 0.0:
                dot = -1e4

            if dot > max_score:
                max_score = dot

        acc = tl.zeros((D,), dtype=tl.float32)
        for n in range(M_k):
            valid = 1.0
            if Mask_ptr is not None:
                valid = tl.load(Mask_ptr + m_idx * stride_mm + n * stride_mn).to(tl.float32)

            k = tl.load(K_ptr + n * stride_kn + d_offs * stride_kd).to(tl.float32)
            dot = tl.sum(q * k) * scale
            if valid <= 0.0:
                dot = -1e4

            p = tl.exp(dot - max_score)
            sum_exp += p
            v = tl.load(V_ptr + n * stride_vn + d_offs * stride_vd).to(tl.float32)
            acc += p * v

        if sum_exp > 0.0:
            out = acc / sum_exp
        else:
            out = tl.zeros((D,), dtype=tl.float32)

        tl.store(Out_ptr + m_idx * stride_om + d_offs * stride_od, out)


def _kmeans_assign(feat: torch.Tensor, K: int, n_iters: int = 3) -> torch.Tensor:
    """Deterministic spherical k-means assignment on (B, H, L, d)."""
    B, H, L, d = feat.shape
    K = min(K, L)
    step = max(1, L // K)
    init_idx = torch.arange(0, L, step, device=feat.device)[:K]
    if init_idx.size(0) < K:
        extra = torch.arange(0, K - init_idx.size(0), device=feat.device)
        init_idx = torch.cat([init_idx, extra])

    feat_f = feat.float()
    centroids = feat_f[:, :, init_idx, :].clone()

    cluster_ids = torch.zeros(B, H, L, dtype=torch.long, device=feat.device)
    for _ in range(n_iters):
        x_norm = (feat_f**2).sum(dim=-1, keepdim=True)
        c_norm = (centroids**2).sum(dim=-1).unsqueeze(-2)
        cross = torch.matmul(feat_f, centroids.transpose(-2, -1))
        dist = x_norm - 2 * cross + c_norm
        cluster_ids = torch.argmin(dist, dim=-1)

        one_hot = F.one_hot(cluster_ids, num_classes=K).to(feat_f.dtype)
        counts = one_hot.sum(dim=2).clamp(min=1.0).unsqueeze(-1)
        sum_f = torch.einsum("bhlk,bhld->bhkd", one_hot, feat_f)
        centroids = sum_f / counts

    return cluster_ids


def _routing_attention_pytorch(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: torch.Tensor | None = None,
    num_clusters: int = 16,
    n_iters: int = 3,
    is_causal: bool = False,
    **kwargs: Any,
) -> torch.Tensor:
    """Pure PyTorch reference implementation matching RoutingAttention."""
    B, H, L, d = q.shape
    device = q.device
    dtype = q.dtype

    K_num = min(max(1, num_clusters), L)
    joint_feat = F.normalize(q + k, p=2, dim=-1)
    cluster_ids = _kmeans_assign(joint_feat, K_num, n_iters)

    scale = 1.0 / math.sqrt(int(d))
    outputs = torch.zeros_like(q)
    arange_L = torch.arange(L, device=device)

    for c in range(K_num):
        q_mask_c = cluster_ids == c
        k_mask_c = q_mask_c | (arange_L.view(1, 1, -1) == 0)

        max_q = int(q_mask_c.sum(dim=-1).max().item())
        max_k = int(k_mask_c.sum(dim=-1).max().item())

        if max_q == 0 or max_k == 0:
            continue

        q_score = q_mask_c.to(torch.int64) * (2 * L) - arange_L.view(1, 1, L)
        k_score = k_mask_c.to(torch.int64) * (2 * L) - arange_L.view(1, 1, L)

        q_idx = q_score.argsort(dim=-1, descending=True)[:, :, :max_q]
        k_idx = k_score.argsort(dim=-1, descending=True)[:, :, :max_k]

        valid_q = torch.gather(q_mask_c, 2, q_idx)
        valid_k = torch.gather(k_mask_c, 2, k_idx)

        q_idx_exp = q_idx.unsqueeze(-1).expand(-1, -1, -1, d)
        k_idx_exp = k_idx.unsqueeze(-1).expand(-1, -1, -1, d)

        Q_k = torch.gather(q, 2, q_idx_exp)
        K_k = torch.gather(k, 2, k_idx_exp)
        V_k = torch.gather(v, 2, k_idx_exp)

        scores_k = torch.matmul(Q_k.float(), K_k.float().transpose(-2, -1).contiguous()) * scale
        pair_mask = valid_q.unsqueeze(-1) & valid_k.unsqueeze(-2)

        if is_causal:
            abs_pos_q = q_idx.unsqueeze(-1)
            abs_pos_k = k_idx.unsqueeze(-2)
            causal_mask = abs_pos_q >= abs_pos_k
            pair_mask = pair_mask & causal_mask

        scores_k = scores_k.masked_fill(~pair_mask, -1e4)
        scores_k = torch.nan_to_num(scores_k, nan=-1e4, posinf=1e4, neginf=-1e4)

        attn_k = F.softmax(scores_k, dim=-1).to(dtype)
        attn_k = torch.nan_to_num(attn_k, nan=0.0)

        out_k = torch.matmul(attn_k, V_k)
        out_k = out_k * valid_q.unsqueeze(-1).to(out_k.dtype)
        outputs.scatter_add_(2, q_idx_exp, out_k.to(outputs.dtype))

    return outputs


def routing_attention_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: torch.Tensor | None = None,
    num_clusters: int = 16,
    n_iters: int = 3,
    is_causal: bool = False,
    **kwargs: Any,
) -> torch.Tensor:
    """Routing Attention forward operator using Triton kernel with PyTorch fallback.

    Args:
        q: Query tensor of shape (B, H, L, d).
        k: Key tensor of shape (B, H, L, d).
        v: Value tensor of shape (B, H, L, d).
        mask: Optional attention mask.
        num_clusters: Number of spherical k-means clusters.
        n_iters: Number of k-means clustering iterations.
        is_causal: Whether to apply causal autoregressive masking.

    Returns:
        Attention output tensor of shape (B, H, L, d).
    """
    if not _INTERPRET_MODE:
        is_suitable, reason = check_triton_suitability(q, "routing_attention", allow_fp32=True)
        if not is_suitable:
            logger.warning("routing_attention Triton fallback: %s. Using PyTorch reference.", reason)
            return _routing_attention_pytorch(
                q, k, v, mask=mask, num_clusters=num_clusters, n_iters=n_iters, is_causal=is_causal, **kwargs
            )

    B, H, L, d = q.shape
    device = q.device
    dtype = q.dtype

    K_num = min(max(1, num_clusters), L)
    joint_feat = F.normalize(q + k, p=2, dim=-1)
    cluster_ids = _kmeans_assign(joint_feat, K_num, n_iters)

    scale = 1.0 / math.sqrt(int(d))
    outputs = torch.zeros_like(q)
    arange_L = torch.arange(L, device=device)

    for c in range(K_num):
        q_mask_c = cluster_ids == c
        k_mask_c = q_mask_c | (arange_L.view(1, 1, -1) == 0)

        max_q = int(q_mask_c.sum(dim=-1).max().item())
        max_k = int(k_mask_c.sum(dim=-1).max().item())

        if max_q == 0 or max_k == 0:
            continue

        q_score = q_mask_c.to(torch.int64) * (2 * L) - arange_L.view(1, 1, L)
        k_score = k_mask_c.to(torch.int64) * (2 * L) - arange_L.view(1, 1, L)

        q_idx = q_score.argsort(dim=-1, descending=True)[:, :, :max_q]
        k_idx = k_score.argsort(dim=-1, descending=True)[:, :, :max_k]

        valid_q = torch.gather(q_mask_c, 2, q_idx)
        valid_k = torch.gather(k_mask_c, 2, k_idx)

        q_idx_exp = q_idx.unsqueeze(-1).expand(-1, -1, -1, d)
        k_idx_exp = k_idx.unsqueeze(-1).expand(-1, -1, -1, d)

        Q_k = torch.gather(q, 2, q_idx_exp).contiguous().float()
        K_k = torch.gather(k, 2, k_idx_exp).contiguous().float()
        V_k = torch.gather(v, 2, k_idx_exp).contiguous().float()

        pair_mask = valid_q.unsqueeze(-1) & valid_k.unsqueeze(-2)
        if is_causal:
            abs_pos_q = q_idx.unsqueeze(-1)
            abs_pos_k = k_idx.unsqueeze(-2)
            causal_mask = abs_pos_q >= abs_pos_k
            pair_mask = pair_mask & causal_mask

        pair_mask_f = pair_mask.contiguous().float()
        out_k = torch.empty_like(Q_k)

        # Dispatch per (batch, head)
        for b in range(B):
            for h in range(H):
                q_slice = Q_k[b, h]
                k_slice = K_k[b, h]
                v_slice = V_k[b, h]
                mask_slice = pair_mask_f[b, h]
                out_slice = out_k[b, h]

                if _INTERPRET_MODE:
                    grid = (max_q,)
                    _cluster_attn_kernel[grid](
                        q_slice,
                        k_slice,
                        v_slice,
                        out_slice,
                        mask_slice,
                        q_slice.stride(0),
                        q_slice.stride(1),
                        k_slice.stride(0),
                        k_slice.stride(1),
                        v_slice.stride(0),
                        v_slice.stride(1),
                        out_slice.stride(0),
                        out_slice.stride(1),
                        mask_slice.stride(0),
                        mask_slice.stride(1),
                        M_q=max_q,
                        M_k=max_k,
                        D=d,
                        scale=scale,
                    )
                else:
                    grid = lambda meta: (max_q,)
                    _cluster_attn_kernel_autotuned[grid](
                        q_slice,
                        k_slice,
                        v_slice,
                        out_slice,
                        mask_slice,
                        q_slice.stride(0),
                        q_slice.stride(1),
                        k_slice.stride(0),
                        k_slice.stride(1),
                        v_slice.stride(0),
                        v_slice.stride(1),
                        out_slice.stride(0),
                        out_slice.stride(1),
                        mask_slice.stride(0),
                        mask_slice.stride(1),
                        M_q=max_q,
                        M_k=max_k,
                        D=d,
                        scale=scale,
                    )

        out_k = out_k.to(dtype) * valid_q.unsqueeze(-1).to(dtype)
        outputs.scatter_add_(2, q_idx_exp, out_k)

    return outputs


routing_attention_forward_pytorch = _routing_attention_pytorch

__all__ = ["routing_attention_forward", "routing_attention_forward_pytorch", "_routing_attention_pytorch"]

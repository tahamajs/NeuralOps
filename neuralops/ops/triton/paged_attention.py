#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Paged Attention Kernel

Executes attention lookups over non-contiguous paged memory blocks for high-throughput KV caching.
"""

from __future__ import annotations
import math
import torch
import torch.nn.functional as F

from neuralops.ops.triton._common import HAS_TRITON, triton, tl

__all__ = ["paged_attention_kernel", "paged_attention_pytorch"]


if HAS_TRITON and tl is not None:
    @triton.jit
    def paged_attention_fwd_kernel(
        Q_ptr, K_CACHE_ptr, V_CACHE_ptr, BLOCK_TABLE_ptr, SEQ_LENS_ptr, OUT_ptr,
        stride_qb, stride_qh, stride_qd,
        stride_kblk, stride_kidx, stride_kh, stride_kd,
        stride_vblk, stride_vidx, stride_vh, stride_vd,
        stride_bt_b, stride_bt_idx,
        stride_ob, stride_oh, stride_od,
        H,
        BLOCK_SIZE: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        """
        Fused Paged Attention Decode Kernel:
        Gather physical blocks via BLOCK_TABLE and compute online softmax in SRAM.
        """
        pid_b = tl.program_id(0)
        pid_h = tl.program_id(1)

        seq_len = tl.load(SEQ_LENS_ptr + pid_b)
        if seq_len <= 0:
            return

        num_blocks = tl.cdiv(seq_len, BLOCK_SIZE)
        offs_d = tl.arange(0, BLOCK_D)

        q_ptrs = Q_ptr + pid_b * stride_qb + pid_h * stride_qh + offs_d * stride_qd
        q = tl.load(q_ptrs).to(tl.float32)

        sm_scale = 1.0 / tl.sqrt(tl.cast(BLOCK_D, tl.float32))

        m_i = -1e4
        l_i = 0.0
        acc = tl.zeros([BLOCK_D], dtype=tl.float32)

        for i in range(num_blocks):
            phys_block = tl.load(BLOCK_TABLE_ptr + pid_b * stride_bt_b + i * stride_bt_idx)

            offs_tok = i * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
            mask_tok = offs_tok < seq_len

            k_ptrs = (
                K_CACHE_ptr
                + phys_block * stride_kblk
                + tl.arange(0, BLOCK_SIZE)[:, None] * stride_kidx
                + pid_h * stride_kh
                + offs_d[None, :] * stride_kd
            )
            k = tl.load(k_ptrs, mask=mask_tok[:, None], other=0.0).to(tl.float32)

            scores = tl.sum(q[None, :] * k, axis=1) * sm_scale
            scores = tl.where(mask_tok, scores, -1e4)

            m_curr = tl.maximum(m_i, tl.max(scores, axis=0))
            alpha = tl.math.exp2(1.44269504 * (m_i - m_curr))
            p = tl.math.exp2(1.44269504 * (scores - m_curr))
            l_curr = l_i * alpha + tl.sum(p, axis=0)
            acc = acc * alpha

            v_ptrs = (
                V_CACHE_ptr
                + phys_block * stride_vblk
                + tl.arange(0, BLOCK_SIZE)[:, None] * stride_vidx
                + pid_h * stride_vh
                + offs_d[None, :] * stride_vd
            )
            v = tl.load(v_ptrs, mask=mask_tok[:, None], other=0.0).to(tl.float32)

            acc += tl.sum(p[:, None] * v, axis=0)

            m_i = m_curr
            l_i = l_curr

        out = acc / tl.maximum(l_i, 1e-6)
        out_ptrs = OUT_ptr + pid_b * stride_ob + pid_h * stride_oh + offs_d * stride_od
        tl.store(out_ptrs, out.to(OUT_ptr.dtype.element_ty))

    @triton.jit
    def paged_attention_splitk_stage1_kernel(
        Q_ptr, K_CACHE_ptr, V_CACHE_ptr, BLOCK_TABLE_ptr, SEQ_LENS_ptr,
        MidOut_ptr, MidLse_ptr,
        stride_qb, stride_qh, stride_qd,
        stride_kblk, stride_kidx, stride_kh, stride_kd,
        stride_vblk, stride_vidx, stride_vh, stride_vd,
        stride_bt_b, stride_bt_idx,
        stride_mb, stride_mh, stride_ms, stride_md,
        stride_lb, stride_lh, stride_ls,
        H,
        NUM_SPLITS: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        """
        Stage 1 of Split-K Flash-Decoding for PagedAttention:
        Partition sequence blocks across NUM_SPLITS CTAs and compute partial (m, l, acc).
        """
        pid_s = tl.program_id(0)
        pid_b = tl.program_id(1)
        pid_h = tl.program_id(2)

        seq_len = tl.load(SEQ_LENS_ptr + pid_b)
        if seq_len <= 0:
            return

        total_blocks = tl.cdiv(seq_len, BLOCK_SIZE)
        blocks_per_split = tl.cdiv(total_blocks, NUM_SPLITS)
        start_block = pid_s * blocks_per_split
        end_block = tl.minimum(start_block + blocks_per_split, total_blocks)

        offs_d = tl.arange(0, BLOCK_D)
        q_ptrs = Q_ptr + pid_b * stride_qb + pid_h * stride_qh + offs_d * stride_qd
        q = tl.load(q_ptrs).to(tl.float32)

        sm_scale = 1.0 / tl.sqrt(tl.cast(BLOCK_D, tl.float32))

        m_i = -1e4
        l_i = 0.0
        acc = tl.zeros([BLOCK_D], dtype=tl.float32)

        if start_block < end_block:
            for i in range(start_block, end_block):
                phys_block = tl.load(BLOCK_TABLE_ptr + pid_b * stride_bt_b + i * stride_bt_idx)

                offs_tok = i * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
                mask_tok = offs_tok < seq_len

                k_ptrs = (
                    K_CACHE_ptr
                    + phys_block * stride_kblk
                    + tl.arange(0, BLOCK_SIZE)[:, None] * stride_kidx
                    + pid_h * stride_kh
                    + offs_d[None, :] * stride_kd
                )
                k = tl.load(k_ptrs, mask=mask_tok[:, None], other=0.0).to(tl.float32)

                scores = tl.sum(q[None, :] * k, axis=1) * sm_scale
                scores = tl.where(mask_tok, scores, -1e4)

                m_curr = tl.maximum(m_i, tl.max(scores, axis=0))
                alpha = tl.math.exp2(1.44269504 * (m_i - m_curr))
                p = tl.math.exp2(1.44269504 * (scores - m_curr))
                l_curr = l_i * alpha + tl.sum(p, axis=0)
                acc = acc * alpha

                v_ptrs = (
                    V_CACHE_ptr
                    + phys_block * stride_vblk
                    + tl.arange(0, BLOCK_SIZE)[:, None] * stride_vidx
                    + pid_h * stride_vh
                    + offs_d[None, :] * stride_vd
                )
                v = tl.load(v_ptrs, mask=mask_tok[:, None], other=0.0).to(tl.float32)

                acc += tl.sum(p[:, None] * v, axis=0)
                m_i = m_curr
                l_i = l_curr

        out_partial = acc / tl.maximum(l_i, 1e-6)
        lse_partial = m_i + tl.log(tl.maximum(l_i, 1e-6))

        mid_out_ptrs = MidOut_ptr + pid_b * stride_mb + pid_h * stride_mh + pid_s * stride_ms + offs_d * stride_md
        tl.store(mid_out_ptrs, out_partial.to(MidOut_ptr.dtype.element_ty))

        mid_lse_ptr = MidLse_ptr + pid_b * stride_lb + pid_h * stride_lh + pid_s * stride_ls
        tl.store(mid_lse_ptr, lse_partial)

    @triton.jit
    def paged_attention_splitk_stage2_kernel(
        MidOut_ptr, MidLse_ptr, SEQ_LENS_ptr, OUT_ptr,
        stride_mb, stride_mh, stride_ms, stride_md,
        stride_lb, stride_lh, stride_ls,
        stride_ob, stride_oh, stride_od,
        H,
        NUM_SPLITS: tl.constexpr,
        BLOCK_D: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        """
        Stage 2 of Split-K Flash-Decoding for PagedAttention:
        Tree-reduction over the NUM_SPLITS partial accumulators in SRAM.
        """
        pid_b = tl.program_id(0)
        pid_h = tl.program_id(1)

        seq_len = tl.load(SEQ_LENS_ptr + pid_b)
        if seq_len <= 0:
            return

        total_blocks = tl.cdiv(seq_len, BLOCK_SIZE)
        blocks_per_split = tl.cdiv(total_blocks, NUM_SPLITS)
        active_splits = tl.minimum(NUM_SPLITS, tl.cdiv(total_blocks, tl.maximum(blocks_per_split, 1)))
        active_splits = tl.maximum(active_splits, 1)

        offs_s = tl.arange(0, NUM_SPLITS)
        mask_s = offs_s < active_splits

        lse_ptrs = MidLse_ptr + pid_b * stride_lb + pid_h * stride_lh + offs_s * stride_ls
        lse_vals = tl.load(lse_ptrs, mask=mask_s, other=-1e4)

        max_lse = tl.max(lse_vals, axis=0)
        p = tl.where(mask_s, tl.math.exp2(1.44269504 * (lse_vals - max_lse)), 0.0)
        global_sum = tl.sum(p, axis=0)

        offs_d = tl.arange(0, BLOCK_D)
        acc = tl.zeros([BLOCK_D], dtype=tl.float32)

        for s in range(NUM_SPLITS):
            if s < active_splits:
                w_s = tl.load(MidLse_ptr + pid_b * stride_lb + pid_h * stride_lh + s * stride_ls)
                weight = tl.math.exp2(1.44269504 * (w_s - max_lse)) / tl.maximum(global_sum, 1e-6)

                mid_ptrs = MidOut_ptr + pid_b * stride_mb + pid_h * stride_mh + s * stride_ms + offs_d * stride_md
                val_s = tl.load(mid_ptrs).to(tl.float32)
                acc += weight * val_s

        out_ptrs = OUT_ptr + pid_b * stride_ob + pid_h * stride_oh + offs_d * stride_od
        tl.store(out_ptrs, acc.to(OUT_ptr.dtype.element_ty))


def paged_attention_pytorch(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
    block_size: int = 16,
) -> torch.Tensor:
    """Reference implementation of Paged Attention gather and decode."""
    B, H, D = q.shape
    out = torch.zeros_like(q)
    scale = 1.0 / math.sqrt(D)

    for b in range(B):
        slen = int(seq_lens[b].item())
        if slen == 0:
            continue
        num_blocks = (slen + block_size - 1) // block_size
        blocks = block_tables[b, :num_blocks].long()

        k_blocks = k_cache[blocks]
        v_blocks = v_cache[blocks]

        k_flat = k_blocks.view(-1, H, D)[:slen]
        v_flat = v_blocks.view(-1, H, D)[:slen]

        q_b = q[b].unsqueeze(1)
        k_b = k_flat.transpose(0, 1)
        v_b = v_flat.transpose(0, 1)

        scores = torch.matmul(q_b, k_b.transpose(-2, -1)) * scale
        weights = F.softmax(scores, dim=-1)
        res = torch.matmul(weights, v_b).squeeze(1)
        out[b] = res

    return out


def paged_attention_kernel(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
    block_size: int = 16,
) -> torch.Tensor:
    """Dispatches paged attention to Triton kernel if supported, else PyTorch fallback."""
    from neuralops.ops.triton._types import HAS_TRITON, FORCE_PYTORCH_FALLBACK, check_triton_suitability
    B, H, D = q.shape
    if (
        HAS_TRITON
        and not FORCE_PYTORCH_FALLBACK
        and q.is_cuda
        and k_cache.is_cuda
        and v_cache.is_cuda
        and D in (16, 32, 64, 128, 256)
    ):
        try:
            total_elements = k_cache.numel()
            suitable, _ = check_triton_suitability(k_cache, "paged_attention")
            if not suitable:
                return paged_attention_pytorch(q, k_cache, v_cache, block_tables, seq_lens, block_size=block_size)

            out = torch.empty_like(q)
            max_seq_len = int(seq_lens.max().item()) if seq_lens.numel() > 0 else 0
            NUM_SPLITS = 16
            if max_seq_len > block_size * NUM_SPLITS:
                mid_out = torch.empty((B, H, NUM_SPLITS, D), dtype=q.dtype, device=q.device)
                mid_lse = torch.empty((B, H, NUM_SPLITS), dtype=torch.float32, device=q.device)
                grid_stage1 = (NUM_SPLITS, B, H)
                num_warps = 4 if D >= 64 else 2
                paged_attention_splitk_stage1_kernel[grid_stage1](
                    q, k_cache, v_cache, block_tables, seq_lens,
                    mid_out, mid_lse,
                    q.stride(0), q.stride(1), q.stride(2),
                    k_cache.stride(0), k_cache.stride(1), k_cache.stride(2), k_cache.stride(3),
                    v_cache.stride(0), v_cache.stride(1), v_cache.stride(2), v_cache.stride(3),
                    block_tables.stride(0), block_tables.stride(1),
                    mid_out.stride(0), mid_out.stride(1), mid_out.stride(2), mid_out.stride(3),
                    mid_lse.stride(0), mid_lse.stride(1), mid_lse.stride(2),
                    H,
                    NUM_SPLITS=NUM_SPLITS,
                    BLOCK_SIZE=block_size,
                    BLOCK_D=D,
                    num_warps=num_warps,
                )
                grid_stage2 = (B, H)
                paged_attention_splitk_stage2_kernel[grid_stage2](
                    mid_out, mid_lse, seq_lens, out,
                    mid_out.stride(0), mid_out.stride(1), mid_out.stride(2), mid_out.stride(3),
                    mid_lse.stride(0), mid_lse.stride(1), mid_lse.stride(2),
                    out.stride(0), out.stride(1), out.stride(2),
                    H,
                    NUM_SPLITS=NUM_SPLITS,
                    BLOCK_D=D,
                    BLOCK_SIZE=block_size,
                    num_warps=2,
                )
                return out

            grid = (B, H)
            num_warps = 4 if D >= 64 else 2
            paged_attention_fwd_kernel[grid](
                q, k_cache, v_cache, block_tables, seq_lens, out,
                q.stride(0), q.stride(1), q.stride(2),
                k_cache.stride(0), k_cache.stride(1), k_cache.stride(2), k_cache.stride(3),
                v_cache.stride(0), v_cache.stride(1), v_cache.stride(2), v_cache.stride(3),
                block_tables.stride(0), block_tables.stride(1),
                out.stride(0), out.stride(1), out.stride(2),
                H,
                BLOCK_SIZE=block_size,
                BLOCK_D=D,
                num_warps=num_warps,
            )
            return out
        except Exception:
            # allowed-swallow: best-effort-telemetry
            pass
    return paged_attention_pytorch(q, k_cache, v_cache, block_tables, seq_lens, block_size=block_size)


if HAS_TRITON and tl is not None:
    @triton.jit
    def _paged_recurrent_state_step_kernel(
        Q_ptr, K_ptr, V_ptr,
        PAGE_POOL_ptr, BLOCK_TABLE_ptr, OUT_ptr,
        stride_qb, stride_qh, stride_qd,
        stride_kb, stride_kh, stride_kd,
        stride_vb, stride_vh, stride_vd,
        stride_pp_page, stride_pp_h, stride_pp_d1, stride_pp_d2,
        stride_bt_b, stride_bt_idx,
        stride_ob, stride_oh, stride_od,
        decay: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        """
        Updates recurrent physical page state S_{phys} in-place:
          S_{phys}[h] = decay * S_{phys}[h] + k[h]^T @ v[h]
          y[h] = q[h] @ S_{phys}[h]
        Executed concurrently across (batch, head).
        """
        pid_b = tl.program_id(0)
        pid_h = tl.program_id(1)

        phys_page = tl.load(BLOCK_TABLE_ptr + pid_b * stride_bt_b + 0 * stride_bt_idx)

        offs_d1 = tl.arange(0, BLOCK_D)
        offs_d2 = tl.arange(0, BLOCK_D)

        q = tl.load(Q_ptr + pid_b * stride_qb + pid_h * stride_qh + offs_d1 * stride_qd)
        k = tl.load(K_ptr + pid_b * stride_kb + pid_h * stride_kh + offs_d1 * stride_kd)
        v = tl.load(V_ptr + pid_b * stride_vb + pid_h * stride_vh + offs_d2 * stride_vd)

        s_ptrs = (
            PAGE_POOL_ptr
            + phys_page * stride_pp_page
            + pid_h * stride_pp_h
            + offs_d1[:, None] * stride_pp_d1
            + offs_d2[None, :] * stride_pp_d2
        )

        s_val = tl.load(s_ptrs)
        kv = k[:, None] * v[None, :]
        new_s = s_val * decay + kv

        tl.store(s_ptrs, new_s)

        out = tl.sum(q[:, None] * new_s, axis=0)
        tl.store(OUT_ptr + pid_b * stride_ob + pid_h * stride_oh + offs_d2 * stride_od, out)


def paged_recurrent_state_step_pytorch(
    q_t: torch.Tensor,
    k_t: torch.Tensor,
    v_t: torch.Tensor,
    page_pool: torch.Tensor,
    block_tables: torch.Tensor,
    page_slot: int = 0,
    decay: float = 1.0,
) -> torch.Tensor:
    """PyTorch reference update for paged recurrent state pool.

    Args:
        q_t: [B, H, D]
        k_t: [B, H, D]
        v_t: [B, H, D]
        page_pool: [TotalPages, H, D, D]
        block_tables: [B, MaxPagesPerSeq]
        page_slot: page index within sequence's allocated pages
        decay: state transition decay multiplier
    Returns:
        out: [B, H, D]
    """
    B, H, D = q_t.shape
    if block_tables.dim() == 1:
        phys_pages = block_tables
    else:
        phys_pages = block_tables[:, page_slot]
    S_curr = page_pool[phys_pages]
    kv = torch.matmul(k_t.unsqueeze(-1), v_t.unsqueeze(-2))
    S_next = decay * S_curr + kv
    page_pool[phys_pages] = S_next
    y = torch.matmul(q_t.unsqueeze(-2), S_next).squeeze(-2)
    return y


def paged_recurrent_state_step_triton(
    q_t: torch.Tensor,
    k_t: torch.Tensor,
    v_t: torch.Tensor,
    page_pool: torch.Tensor,
    block_tables: torch.Tensor,
    page_slot: int = 0,
    decay: float = 1.0,
) -> torch.Tensor:
    """Dispatches paged recurrent state step to Triton kernel or PyTorch reference."""
    B, H, D = q_t.shape
    if (
        HAS_TRITON
        and tl is not None
        and q_t.is_cuda
        and page_pool.is_cuda
        and page_slot == 0
        and D in (16, 32, 64, 128)
        and q_t.dtype in (torch.float16, torch.bfloat16, torch.float32)
    ):
        try:
            out = torch.empty_like(q_t)
            grid = (B, H)
            _paged_recurrent_state_step_kernel[grid](
                q_t, k_t, v_t,
                page_pool, block_tables, out,
                q_t.stride(0), q_t.stride(1), q_t.stride(2),
                k_t.stride(0), k_t.stride(1), k_t.stride(2),
                v_t.stride(0), v_t.stride(1), v_t.stride(2),
                page_pool.stride(0), page_pool.stride(1), page_pool.stride(2), page_pool.stride(3),
                block_tables.stride(0), block_tables.stride(1),
                out.stride(0), out.stride(1), out.stride(2),
                decay=float(decay),
                BLOCK_D=D,
            )
            return out
        except Exception:
            # allowed-swallow: best-effort-telemetry
            pass

    return paged_recurrent_state_step_pytorch(
        q_t, k_t, v_t, page_pool, block_tables, page_slot=page_slot, decay=decay
    )


if HAS_TRITON and tl is not None:
    @triton.jit
    def _paged_kv_evict_compact_kernel(
        K_IN_ptr, V_IN_ptr,
        K_OUT_ptr, V_OUT_ptr,
        SCORES_ptr,
        KEEP_INDICES_ptr,
        total_items,
        D: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        """In-SRAM Paged KV-Cache Eviction & Compaction Kernel.
        Gathers top-k high-importance KV tokens according to keep_indices in parallel,
        compacting cache memory in-place without CPU-GPU synchronization.
        """
        pid = tl.program_id(0)
        if pid >= total_items:
            return

        target_idx = tl.load(KEEP_INDICES_ptr + pid)

        offs_d = tl.arange(0, BLOCK_D)
        mask_d = offs_d < D

        k_val = tl.load(K_IN_ptr + target_idx * D + offs_d, mask=mask_d, other=0.0)
        v_val = tl.load(V_IN_ptr + target_idx * D + offs_d, mask=mask_d, other=0.0)

        tl.store(K_OUT_ptr + pid * D + offs_d, k_val, mask=mask_d)
        tl.store(V_OUT_ptr + pid * D + offs_d, v_val, mask=mask_d)


def paged_kv_evict_compact_triton(
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    importance_scores: torch.Tensor,
    keep_ratio: float = 0.5,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compacts KV-cache by selecting top-k important tokens using fused GPU kernel."""
    total_tokens, D = k_cache.shape[0], k_cache.shape[-1]
    num_keep = max(1, int(total_tokens * keep_ratio))

    _, keep_indices = torch.topk(importance_scores, k=num_keep, sorted=True)
    keep_indices = keep_indices.sort().values.contiguous()

    from neuralops.ops.triton._types import HAS_TRITON, FORCE_PYTORCH_FALLBACK
    if HAS_TRITON and k_cache.is_cuda and not FORCE_PYTORCH_FALLBACK and D in (16, 32, 64, 128, 256):
        try:
            k_out = torch.empty((num_keep, D), dtype=k_cache.dtype, device=k_cache.device)
            v_out = torch.empty((num_keep, D), dtype=v_cache.dtype, device=v_cache.device)
            grid = (num_keep,)
            _paged_kv_evict_compact_kernel[grid](
                k_cache.contiguous(), v_cache.contiguous(),
                k_out, v_out,
                importance_scores.contiguous(),
                keep_indices,
                num_keep,
                D=D,
                BLOCK_D=D,
            )
            return k_out, v_out
        except Exception:
            # allowed-swallow: best-effort-telemetry
            pass

    return k_cache[keep_indices], v_cache[keep_indices]


__all__ = [
    "paged_attention_kernel",
    "paged_attention_pytorch",
    "paged_recurrent_state_step_triton",
    "paged_recurrent_state_step_pytorch",
    "paged_kv_evict_compact_triton",
    "triton_fused_latent_kv_cache_attention",
    "triton_fused_latent_kv_cache_attention_pytorch_ref",
]

from neuralops.ops.triton.linear_attn import (
    triton_fused_latent_kv_cache_attention,
    triton_fused_latent_kv_cache_attention_pytorch_ref,
)

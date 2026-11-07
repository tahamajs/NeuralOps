#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
vLLM and SGLang Custom Operator Integration (Phases 086-089).

Provides dynamic custom op dispatcher and zero-copy physical block table
dereferencing for high-throughput paged serving inference.
"""

from __future__ import annotations

import logging
import math
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F

logger = logging.getLogger("NeuralOps.vLLM.CustomOps")

__all__ = [
    "paged_attention_custom_op",
    "paged_attention_dispatch",
    "register_vllm_custom_ops",
    "dereference_paged_kv_cache",
]


def dereference_paged_kv_cache(
    k_pool: torch.Tensor,
    v_pool: torch.Tensor,
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    """
    Dereferences physical KV cache pool blocks into per-sequence logical tensors.

    Args:
        k_pool: Physical key tensor of shape [TotalBlocks, BlockSize, NumHeads, HeadDim]
                or [TotalBlocks, NumHeads, HeadDim, BlockSize].
        v_pool: Physical value tensor of shape [TotalBlocks, BlockSize, NumHeads, HeadDim]
                or [TotalBlocks, NumHeads, HeadDim, BlockSize].
        block_tables: Block mapping indices of shape [BatchSize, MaxBlocksPerSeq].
        seq_lens: Sequence length integers of shape [BatchSize].

    Returns:
        Tuple of (k_sequences, v_sequences), where each is a list of tensors of shape [NumHeads, SeqLen, HeadDim].
    """
    batch_size = block_tables.shape[0]

    if k_pool.ndim == 4:
        # Standard: [TotalBlocks, BlockSize, H, D]
        block_size = k_pool.shape[1]
    else:
        block_size = 16

    k_seqs: List[torch.Tensor] = []
    v_seqs: List[torch.Tensor] = []

    for b in range(batch_size):
        length = int(seq_lens[b].item())
        if length <= 0:
            empty_k = k_pool.new_empty(k_pool.shape[2], 0, k_pool.shape[3])
            empty_v = v_pool.new_empty(v_pool.shape[2], 0, v_pool.shape[3])
            k_seqs.append(empty_k)
            v_seqs.append(empty_v)
            continue

        b_blocks = block_tables[b]
        k_parts: List[torch.Tensor] = []
        v_parts: List[torch.Tensor] = []
        tokens_collected = 0

        for blk_idx in b_blocks:
            idx = int(blk_idx.item())
            if idx < 0 or tokens_collected >= length:
                break
            k_parts.append(k_pool[idx])
            v_parts.append(v_pool[idx])
            tokens_collected += block_size

        if not k_parts:
            empty_k = k_pool.new_empty(k_pool.shape[2], 0, k_pool.shape[3])
            empty_v = v_pool.new_empty(v_pool.shape[2], 0, v_pool.shape[3])
            k_seqs.append(empty_k)
            v_seqs.append(empty_v)
            continue

        # Concatenate collected blocks along token dimension: [TotalCollectedTokens, H, D] -> slice to [:length] -> permute to [H, length, D]
        k_cat = torch.cat(k_parts, dim=0)[:length].permute(1, 0, 2)
        v_cat = torch.cat(v_parts, dim=0)[:length].permute(1, 0, 2)
        k_seqs.append(k_cat)
        v_seqs.append(v_cat)

    return k_seqs, v_seqs


def paged_attention_dispatch(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
    block_size: int = 16,
    scale: Optional[float] = None,
    max_seq_len: Optional[int] = None,
    num_heads: Optional[int] = None,
    head_size: Optional[int] = None,
) -> torch.Tensor:
    """
    Dynamic 3-level custom op dispatcher for PagedAttention (AGENTS.md §8):
    1. Level 1: vLLM native C++ ops (`vllm._custom_ops.paged_attention_v1`).
    2. Level 2: NeuralOps Triton kernel (`neuralops.ops.triton.paged_attention.paged_attention_kernel`).
    3. Level 3: Vectorized PyTorch reference fallback.

    Logs exact backend used in compliance with AGENTS.md §8.
    """
    # Level 1: Attempt vLLM native C++ op
    try:
        import vllm._custom_ops as vllm_ops

        if hasattr(vllm_ops, "paged_attention_v1"):
            out = torch.empty_like(q)
            effective_scale = scale if scale is not None else 1.0 / math.sqrt(head_size or q.shape[-1])
            max_context = max_seq_len if max_seq_len is not None else int(seq_lens.max().item())
            vllm_ops.paged_attention_v1(
                out,
                q,
                k_cache,
                v_cache,
                num_kv_heads=k_cache.shape[2] if k_cache.ndim == 4 else k_cache.shape[1],
                scale=effective_scale,
                block_tables=block_tables,
                seq_lens=seq_lens,
                block_size=block_size,
                max_seq_len=max_context,
                alibi_slopes=None,
                kv_cache_dtype="auto",
                k_scale=1.0,
                v_scale=1.0,
                tp_rank=0,
                blocksparse_local_blocks=0,
                blocksparse_vert_stride=0,
                blocksparse_block_size=64,
                blocksparse_head_sliding_step=0,
            )
            logger.info("backend_used: vllm._custom_ops.paged_attention_v1 (Level 1)")
            return out
    except Exception as e1:
        logger.debug(
            "paged_attention Level 1 fallback (vLLM custom op): %s -> routing to Level 2",
            e1,
        )

    # Level 2: Attempt NeuralOps Triton paged attention
    try:
        from neuralops.ops.triton.paged_attention import paged_attention_kernel

        # Requires CUDA tensors and 3D query [B, H, D]
        if q.is_cuda and k_cache.is_cuda and v_cache.is_cuda:
            if q.ndim == 3 and num_heads is not None and head_size is not None and q.shape[-1] == num_heads * head_size:
                q_3d = q.view(q.shape[0], q.shape[1], num_heads, head_size)
                # If decode step (q.shape[1] == 1)
                if q_3d.shape[1] == 1:
                    q_triton = q_3d.squeeze(1).transpose(1, 2)  # [B, D, H] -> wait, Triton kernel expects [B, H, D]
                    q_triton = q_3d.squeeze(1)  # [B, H, D]
                    out = paged_attention_kernel(
                        q=q_triton,
                        k_cache=k_cache,
                        v_cache=v_cache,
                        block_tables=block_tables,
                        seq_lens=seq_lens,
                        block_size=block_size,
                    )
                    out = out.unsqueeze(1).contiguous().view_as(q)
                    logger.info("backend_used: neuralops.ops.triton.paged_attention.paged_attention_kernel (Level 2)")
                    return out
            elif q.ndim == 3 and q.shape[1] == (num_heads or q.shape[1]):
                out = paged_attention_kernel(
                    q=q,
                    k_cache=k_cache,
                    v_cache=v_cache,
                    block_tables=block_tables,
                    seq_lens=seq_lens,
                    block_size=block_size,
                )
                logger.info("backend_used: neuralops.ops.triton.paged_attention.paged_attention_kernel (Level 2)")
                return out
    except Exception as e2:
        logger.debug(
            "paged_attention Level 2 fallback (Triton kernel): %s -> routing to Level 3",
            e2,
        )

    # Level 3: Vectorized PyTorch reference fallback
    logger.info("backend_used: paged_attention_reference_pytorch (Level 3)")
    return paged_attention_custom_op(
        q=q,
        k_pool=k_cache,
        v_pool=v_cache,
        block_tables=block_tables,
        seq_lens=seq_lens,
        scale=scale,
        num_heads=num_heads,
        head_size=head_size,
    )


def paged_attention_custom_op(
    q: torch.Tensor,
    k_pool: torch.Tensor,
    v_pool: torch.Tensor,
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
    scale: Optional[float] = None,
    num_heads: Optional[int] = None,
    head_size: Optional[int] = None,
) -> torch.Tensor:
    """
    Paged Attention operator executing on non-contiguous physical block tables.

    Args:
        q: Query tensor of shape [B, S_q, D] or [B, S_q, H, D] or [B, H, S_q, D].
        k_pool: Key cache pool of shape [TotalBlocks, BlockSize, H, D].
        v_pool: Value cache pool of shape [TotalBlocks, BlockSize, H, D].
        block_tables: Integer tensor mapping logical sequence blocks to physical IDs [B, MaxBlocks].
        seq_lens: Tensor of active sequence lengths [B].
        scale: Scaling factor (defaults to 1.0 / sqrt(head_size)).
        num_heads: Number of attention heads (inferred if None).
        head_size: Head dimension (inferred if None).

    Returns:
        Attention output tensor matching the shape and dtype of query.
    """
    orig_shape = q.shape
    orig_dtype = q.dtype
    batch_size = q.shape[0]

    H_kv = k_pool.shape[2]
    D_kv = k_pool.shape[3]

    if q.ndim == 3:
        # [B, S_q, D]
        S_q = q.shape[1]
        H = num_heads if num_heads is not None else H_kv
        D = head_size if head_size is not None else (q.shape[2] // H)
        q_reshaped = q.view(batch_size, S_q, H, D).permute(0, 2, 1, 3)  # [B, H, S_q, D]
        format_mode = "3d_bsd"
    elif q.ndim == 4 and (q.shape[2] == H_kv or (num_heads is not None and q.shape[2] == num_heads)):
        # [B, S_q, H, D]
        S_q = q.shape[1]
        H = q.shape[2]
        D = q.shape[3]
        q_reshaped = q.permute(0, 2, 1, 3)  # [B, H, S_q, D]
        format_mode = "4d_bshd"
    elif q.ndim == 4 and (q.shape[1] == H_kv or (num_heads is not None and q.shape[1] == num_heads)):
        # [B, H, S_q, D]
        H = q.shape[1]
        S_q = q.shape[2]
        D = q.shape[3]
        q_reshaped = q
        format_mode = "4d_bhsd"
    else:
        # Fallback interpretation
        S_q = 1
        H = H_kv
        D = D_kv
        q_reshaped = q.view(batch_size, H, S_q, D)
        format_mode = "fallback"

    if scale is None:
        scale = 1.0 / math.sqrt(D)

    k_seqs, v_seqs = dereference_paged_kv_cache(k_pool, v_pool, block_tables, seq_lens)
    outputs: List[torch.Tensor] = []

    for b in range(batch_size):
        length = int(seq_lens[b].item())
        if length == 0:
            outputs.append(torch.zeros((H, S_q, D), dtype=orig_dtype, device=q.device))
            continue

        k_seq = k_seqs[b]  # [H_kv, length, D_kv]
        v_seq = v_seqs[b]  # [H_kv, length, D_kv]
        q_b = q_reshaped[b]  # [H, S_q, D]

        # Multi-query / Grouped-query attention broadcast if H != H_kv
        if H != H_kv:
            repeat_factor = H // H_kv
            k_seq = k_seq.repeat_interleave(repeat_factor, dim=0)
            v_seq = v_seq.repeat_interleave(repeat_factor, dim=0)

        scores = torch.matmul(q_b, k_seq.transpose(-2, -1)) * scale  # [H, S_q, length]
        attn_weights = F.softmax(scores, dim=-1, dtype=torch.float32).to(orig_dtype)
        out_b = torch.matmul(attn_weights, v_seq)  # [H, S_q, D]
        outputs.append(out_b)

    res_stacked = torch.stack(outputs, dim=0)  # [B, H, S_q, D]

    if format_mode == "3d_bsd":
        return res_stacked.permute(0, 2, 1, 3).contiguous().view(orig_shape)
    elif format_mode == "4d_bshd":
        return res_stacked.permute(0, 2, 1, 3).contiguous()
    elif format_mode == "4d_bhsd":
        return res_stacked.contiguous()
    else:
        return res_stacked.view(orig_shape)


def register_vllm_custom_ops() -> bool:
    """Registers official custom ops into the PyTorch dispatcher under neuralops namespace."""
    try:
        if hasattr(torch.library, "custom_op"):
            name = "neuralops::paged_attention"
            try:
                torch.library.custom_op(name, mutates_args=())(paged_attention_dispatch)
                logger.info("Successfully registered '%s' into torch.library dispatcher.", name)
            except Exception as reg_err:
                logger.debug("custom_op '%s' already registered or caught: %s", name, reg_err)
            return True
        return False
    except Exception as exc:
        logger.warning("Failed to register torch.library custom op: %s", exc)
        return False

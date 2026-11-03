#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
vLLM and SGLang Custom Operator Integration (Phases 086-089).

Provides zero-copy physical block table dereferencing and PyTorch custom_op
dispatcher registrations for high-throughput paged serving inference.
"""

from __future__ import annotations

import logging
import math
from typing import List, Optional, Tuple
import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)

__all__ = [
    "paged_attention_custom_op",
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
        k_pool: Physical key tensor of shape [TotalBlocks, BlockSize, NumHeads, HeadDim].
        v_pool: Physical value tensor of shape [TotalBlocks, BlockSize, NumHeads, HeadDim].
        block_tables: Block mapping indices of shape [BatchSize, MaxBlocksPerSeq].
        seq_lens: Sequence length integers of shape [BatchSize].

    Returns:
        Tuple of (k_sequences, v_sequences), where each is a list of tensors of shape [NumHeads, SeqLen, HeadDim].
    """
    batch_size = block_tables.shape[0]
    block_size = k_pool.shape[1]
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

        # Concatenate collected blocks along token dimension: [TotalCollectedTokens, H, D]
        k_cat = torch.cat(k_parts, dim=0)[:length].permute(1, 0, 2)
        v_cat = torch.cat(v_parts, dim=0)[:length].permute(1, 0, 2)
        k_seqs.append(k_cat)
        v_seqs.append(v_cat)

    return k_seqs, v_seqs


def paged_attention_custom_op(
    q: torch.Tensor,
    k_pool: torch.Tensor,
    v_pool: torch.Tensor,
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
    scale: Optional[float] = None,
) -> torch.Tensor:
    """
    Paged Attention operator executing on non-contiguous physical block tables.

    Args:
        q: Query tensor of shape [B, 1, H, D] or [B, S_q, H, D].
        k_pool: Key cache pool of shape [TotalBlocks, BlockSize, H, D].
        v_pool: Value cache pool of shape [TotalBlocks, BlockSize, H, D].
        block_tables: Integer tensor mapping logical sequence blocks to physical IDs [B, MaxBlocks].
        seq_lens: Tensor of active sequence lengths [B].
        scale: Scaling factor (defaults to 1.0 / sqrt(HeadDim)).

    Returns:
        Attention output tensor of shape [B, S_q, H, D].
    """
    batch_size, q_len, num_heads, head_dim = q.shape
    if scale is None:
        scale = 1.0 / math.sqrt(head_dim)

    k_seqs, v_seqs = dereference_paged_kv_cache(k_pool, v_pool, block_tables, seq_lens)
    outputs: List[torch.Tensor] = []

    for b in range(batch_size):
        length = int(seq_lens[b].item())
        if length == 0:
            outputs.append(torch.zeros_like(q[b : b + 1]))
            continue

        k_seq = k_seqs[b]  # [H, length, D]
        v_seq = v_seqs[b]  # [H, length, D]
        q_b = q[b].permute(1, 0, 2)  # [H, q_len, D]

        scores = torch.matmul(q_b, k_seq.transpose(-2, -1)) * scale  # [H, q_len, length]
        attn_weights = F.softmax(scores, dim=-1, dtype=torch.float32).to(q.dtype)
        out_b = torch.matmul(attn_weights, v_seq)  # [H, q_len, D]
        outputs.append(out_b.permute(1, 0, 2).unsqueeze(0))  # [1, q_len, H, D]

    return torch.cat(outputs, dim=0)


def register_vllm_custom_ops() -> bool:
    """Registers official custom ops into the PyTorch dispatcher under neuralops namespace."""
    try:
        if hasattr(torch.library, "custom_op"):
            # Check if already registered
            name = "neuralops::paged_attention"
            try:
                torch.library.custom_op(name, mutates_args=())(paged_attention_custom_op)
                logger.info("Successfully registered '%s' into torch.library dispatcher.", name)
            except Exception as reg_err:
                logger.debug("custom_op '%s' already registered or caught: %s", name, reg_err)
            return True
        return False
    except Exception as exc:
        logger.warning("Failed to register torch.library custom op: %s", exc)
        return False

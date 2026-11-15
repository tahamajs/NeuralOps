#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""Paged Attention Layer.

Executes attention lookups over non-contiguous paged memory blocks for high-throughput
KV caching (vLLM style), with strict validation guarding against out-of-bounds page IDs.
"""

from __future__ import annotations

import logging
import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from neuralops.core.attention_utils import LoRALinear
from neuralops.core.base import BaseAttention
from neuralops.core.common import _cfg_get
from neuralops.core.registry import ATTENTION_REGISTRY

logger = logging.getLogger(__name__)


def validate_block_tables(
    block_tables: torch.Tensor,
    num_cache_blocks: int,
    seq_lens: torch.Tensor | None = None,
    block_size: int = 16,
) -> None:
    """Validates that block tables contain strictly valid, in-bounds physical page IDs.

    Args:
        block_tables: Integer tensor of shape (batch_size, max_num_blocks_per_seq).
        num_cache_blocks: Total number of allocated physical blocks in the cache.
        seq_lens: Optional tensor of sequence lengths per batch element.
        block_size: Number of tokens stored per physical block.

    Raises:
        ValueError: If any active block ID is negative or >= num_cache_blocks.
    """
    if block_tables.numel() == 0:
        return

    # Check for active blocks per sequence if seq_lens is provided
    if seq_lens is not None:
        batch_size = block_tables.shape[0]
        for b in range(batch_size):
            slen = int(seq_lens[b].item())
            if slen <= 0:
                continue
            num_blocks = (slen + block_size - 1) // block_size
            active_blocks = block_tables[b, :num_blocks]

            min_id = int(active_blocks.min().item())
            max_id = int(active_blocks.max().item())

            if min_id < 0:
                raise ValueError(
                    f"Out-of-bounds page ID detected in block table for batch {b}: "
                    f"negative block index {min_id} is illegal."
                )
            if max_id >= num_cache_blocks:
                raise ValueError(
                    f"Out-of-bounds page ID detected in block table for batch {b}: "
                    f"block index {max_id} exceeds total physical cache blocks ({num_cache_blocks})."
                )
    else:
        # Global check across all table entries
        min_id = int(block_tables.min().item())
        max_id = int(block_tables.max().item())
        if min_id < 0:
            raise ValueError(f"Out-of-bounds page ID: negative block index {min_id} is illegal.")
        if max_id >= num_cache_blocks:
            raise ValueError(
                f"Out-of-bounds page ID: block index {max_id} exceeds total physical cache blocks ({num_cache_blocks})."
            )


def paged_attention_decode(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
    block_size: int = 16,
) -> torch.Tensor:
    """Executes paged attention decode step with bounds verification and hardware fallback.

    Args:
        q: Query tensor (batch, heads, head_dim) or (batch, heads, 1, head_dim).
        k_cache: Physical key cache (num_blocks, block_size, heads, head_dim).
        v_cache: Physical value cache (num_blocks, block_size, heads, head_dim).
        block_tables: Block tables (batch, max_blocks).
        seq_lens: Sequence lengths per batch sequence (batch,).
        block_size: Tokens per block.

    Returns:
        Decoded context tensor (batch, heads, head_dim).
    """
    while q.dim() > 3 and q.shape[-2] == 1:
        q = q.squeeze(-2)

    num_blocks = k_cache.shape[0]
    validate_block_tables(block_tables, num_blocks, seq_lens, block_size=block_size)

    # Try Triton kernel when on CUDA
    if q.is_cuda and k_cache.is_cuda:
        try:
            from neuralops.ops.triton.paged_attention import paged_attention_kernel

            return paged_attention_kernel(q, k_cache, v_cache, block_tables, seq_lens, block_size=block_size)
        except Exception as e_triton:
            logger.debug("Paged attention Triton kernel fallback: %s", e_triton)

    # Pure PyTorch fallback
    b, h, d = q.shape
    out = torch.zeros_like(q)
    scale = 1.0 / math.sqrt(max(d, 1))

    for i in range(b):
        slen = int(seq_lens[i].item())
        if slen <= 0:
            continue
        req_blocks = (slen + block_size - 1) // block_size
        blocks = block_tables[i, :req_blocks].long()

        k_blks = k_cache[blocks]
        v_blks = v_cache[blocks]

        k_flat = k_blks.view(-1, h, d)[:slen]
        v_flat = v_blks.view(-1, h, d)[:slen]

        q_i = q[i].unsqueeze(1)  # (h, 1, d)
        k_i = k_flat.transpose(0, 1)  # (h, slen, d)
        v_i = v_flat.transpose(0, 1)  # (h, slen, d)

        scores = torch.matmul(q_i.float(), k_i.float().transpose(-2, -1)) * scale
        weights = F.softmax(scores, dim=-1).to(q.dtype)
        res = torch.matmul(weights, v_i).squeeze(1)
        out[i] = res

    return out


@ATTENTION_REGISTRY.register(
    "paged_attention",
    supports_causal=True,
    supports_bidirectional=False,
    supports_gqa=True,
)
class PagedAttention(BaseAttention):
    """Paged Attention layer managing non-contiguous block memory lookups.

    Attributes:
        supports_causal: True.
        supports_bidirectional: False.
        supports_gqa: True.
    """

    FAMILY: str = "sparse"
    supports_causal: bool = True
    supports_bidirectional: bool = False
    supports_gqa: bool = True
    is_custom_attention: bool = True

    def __init__(
        self,
        config: Any | None = None,  # reason: HuggingFace config or dict
        dim: int | None = None,
        num_heads: int | None = None,
        head_dim: int | None = None,
        block_size: int = 16,
        bias: bool = True,
        **kwargs: Any,  # reason: arbitrary hyperparameters
    ) -> None:
        """Initializes PagedAttention."""
        super().__init__(config=config)

        resolved_dim = (
            dim
            or kwargs.get("hidden_dim")
            or kwargs.get("hidden_size")
            or kwargs.get("d_model")
            or _cfg_get(config, "hidden_dim", None)
            or _cfg_get(config, "hidden_size", None)
            or _cfg_get(config, "dim", 768)
        )
        self.dim: int = int(resolved_dim)
        self.hidden_size: int = self.dim

        resolved_heads = (
            num_heads
            or kwargs.get("num_attention_heads")
            or _cfg_get(config, "num_attention_heads", None)
            or _cfg_get(config, "num_heads", 8)
        )
        self.num_heads: int = max(1, int(resolved_heads))

        resolved_head_dim = head_dim or (self.dim // self.num_heads)
        self.head_dim: int = int(resolved_head_dim)
        self.all_head_size: int = self.num_heads * self.head_dim
        self.block_size: int = int(block_size)

        self.q_proj = LoRALinear(self.dim, self.all_head_size, bias=bias, freeze_base=True)
        self.out_proj = LoRALinear(self.all_head_size, self.dim, bias=bias, freeze_base=True)

    def forward(
        self,
        hidden_states: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        block_tables: torch.Tensor,
        seq_lens: torch.Tensor,
        *args: Any,  # reason: absorbs unused positional args
        **kwargs: Any,  # reason: absorbs unused kwargs
    ) -> tuple[torch.Tensor, None]:
        """Forward pass for decode token against paged KV cache."""
        b = hidden_states.shape[0]
        h, d = self.num_heads, self.head_dim

        q = self.q_proj(hidden_states).view(b, -1, h, d).squeeze(1).transpose(1, 2)
        if q.dim() == 4:
            q = q.squeeze(-2)

        context = paged_attention_decode(
            q,
            k_cache,
            v_cache,
            block_tables,
            seq_lens,
            block_size=self.block_size,
        )

        out = self.out_proj(context.transpose(1, 2).contiguous().view(b, 1, self.all_head_size))
        return out, None


PagedAttentionLayer = PagedAttention
paged_attention = PagedAttention

__all__ = [
    "PagedAttention",
    "PagedAttentionLayer",
    "paged_attention",
    "paged_attention_decode",
    "validate_block_tables",
]

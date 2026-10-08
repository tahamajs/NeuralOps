"""Paged KV Cache for high-throughput memory-efficient LLM decoding."""
from __future__ import annotations
from typing import Dict, List, Optional, Tuple
import torch

__all__ = ["PagedKVCache"]


class PagedKVCache:
    """Manages non-contiguous blocks of Key/Value tensors per sequence (vLLM-style)."""

    def __init__(
        self,
        num_blocks: int,
        block_size: int,
        num_heads: int,
        head_dim: int,
        dtype: torch.dtype = torch.float32,
        device: str | torch.device = "cpu",
    ) -> None:
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.dtype = dtype
        self.device = torch.device(device)

        # Preallocated memory pool: (num_blocks, block_size, num_heads, head_dim)
        self.key_cache = torch.zeros(
            (num_blocks, block_size, num_heads, head_dim),
            dtype=dtype,
            device=self.device,
        )
        self.value_cache = torch.zeros(
            (num_blocks, block_size, num_heads, head_dim),
            dtype=dtype,
            device=self.device,
        )

        self.free_blocks: list[int] = list(range(num_blocks))
        # Maps sequence_id -> list of block indices
        self.block_tables: dict[int, list[int]] = {}
        # Maps sequence_id -> current token length
        self.seq_lengths: dict[int, int] = {}

    def allocate_sequence(self, seq_id: int) -> None:
        """Register a new sequence with an empty block list."""
        if seq_id in self.block_tables:
            raise ValueError(f"Sequence {seq_id} already exists.")
        self.block_tables[seq_id] = []
        self.seq_lengths[seq_id] = 0

    def free_sequence(self, seq_id: int) -> None:
        """Free blocks associated with a sequence."""
        if seq_id in self.block_tables:
            self.free_blocks.extend(self.block_tables[seq_id])
            del self.block_tables[seq_id]
            del self.seq_lengths[seq_id]

    def append_token(
        self,
        seq_id: int,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> None:
        """Appends a single token key and value to sequence's cache.

        Args:
            seq_id: Sequence identifier.
            key: Tensor of shape (num_heads, head_dim).
            value: Tensor of shape (num_heads, head_dim).
        """
        if seq_id not in self.block_tables:
            self.allocate_sequence(seq_id)

        seq_len = self.seq_lengths[seq_id]
        token_offset = seq_len % self.block_size

        if token_offset == 0:
            if not self.free_blocks:
                raise RuntimeError("Out of KV cache blocks.")
            block_idx = self.free_blocks.pop(0)
            self.block_tables[seq_id].append(block_idx)
        else:
            block_idx = self.block_tables[seq_id][-1]

        self.key_cache[block_idx, token_offset] = key.to(device=self.device, dtype=self.dtype)
        self.value_cache[block_idx, token_offset] = value.to(device=self.device, dtype=self.dtype)
        self.seq_lengths[seq_id] = seq_len + 1

    def get_kv(self, seq_id: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns contiguous (K, V) tensors of shape (seq_len, num_heads, head_dim)."""
        if seq_id not in self.block_tables:
            raise KeyError(f"Sequence {seq_id} not found.")

        seq_len = self.seq_lengths[seq_id]
        if seq_len == 0:
            empty = torch.empty((0, self.num_heads, self.head_dim), dtype=self.dtype, device=self.device)
            return empty, empty

        blocks = self.block_tables[seq_id]
        k_blocks = [self.key_cache[b] for b in blocks]
        v_blocks = [self.value_cache[b] for b in blocks]

        k_all = torch.cat(k_blocks, dim=0)[:seq_len]
        v_all = torch.cat(v_blocks, dim=0)[:seq_len]
        return k_all, v_all

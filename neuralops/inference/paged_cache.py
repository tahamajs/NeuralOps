"""
Paged Key-Value Cache

Implements paged KV cache structures for high-throughput, memory-efficient LLM decoding,
supporting virtual block mapping and Copy-on-Write (CoW) sequence forking.

Author: Taha Majlesi
"""

from __future__ import annotations

import collections
import logging
from typing import Any, Dict, List, Optional, Tuple, Union
import torch

try:
    from neuralops.models.paged_state import PagedStateTable
except ImportError:
    PagedStateTable = None

logger = logging.getLogger("NeuralOps.PagedKVCache")

__all__ = ["PagedKVCache", "PagedStateTable"]


class PagedKVCache:
    """
    Manages non-contiguous physical blocks of Key/Value tensors per sequence (vLLM-style).
    Supports virtual block mapping, zero-copy sequence forking, and Copy-on-Write (CoW).
    """

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

        # Preallocated physical memory pool: (num_blocks, block_size, num_heads, head_dim)
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
        # Maps sequence_id -> list of physical block indices
        self.block_tables: dict[int, list[int]] = {}
        # Maps sequence_id -> current token length
        self.seq_lengths: dict[int, int] = {}
        # Reference count per physical block ID for zero-copy sharing & CoW
        self.block_ref_counts: dict[int, int] = collections.defaultdict(int)

    def allocate_sequence(self, seq_id: int) -> None:
        """Register a new sequence with an empty block list."""
        if seq_id in self.block_tables:
            raise ValueError(f"Sequence {seq_id} already exists.")
        self.block_tables[seq_id] = []
        self.seq_lengths[seq_id] = 0

    def fork_sequence(self, parent_seq_id: int, child_seq_id: int) -> list[int]:
        """
        Zero-copy sequence forking via virtual block table reference counting.
        Child shares parent's physical blocks until a token append triggers Copy-on-Write.
        """
        if parent_seq_id not in self.block_tables:
            raise KeyError(f"Parent sequence {parent_seq_id} not registered.")
        if child_seq_id in self.block_tables:
            self.free_sequence(child_seq_id)

        parent_blocks = self.block_tables[parent_seq_id]
        for b in parent_blocks:
            self.block_ref_counts[b] += 1

        self.block_tables[child_seq_id] = list(parent_blocks)
        self.seq_lengths[child_seq_id] = self.seq_lengths[parent_seq_id]
        return list(parent_blocks)

    def copy_on_write(self, seq_id: int, block_idx: int) -> int:
        """
        Executes Copy-on-Write for a shared physical block.
        Allocates a free block, copies block data, updates ref counts, and updates block table.
        """
        if self.block_ref_counts.get(block_idx, 0) <= 1:
            return block_idx  # Exclusively owned, no copy needed

        if not self.free_blocks:
            raise RuntimeError(f"PagedKVCache: Out of free physical blocks during Copy-on-Write.")

        new_block_idx = self.free_blocks.pop(0)
        self.key_cache[new_block_idx].copy_(self.key_cache[block_idx])
        self.value_cache[new_block_idx].copy_(self.value_cache[block_idx])

        self.block_ref_counts[block_idx] -= 1
        self.block_ref_counts[new_block_idx] = 1

        # Replace in seq_id's block table
        table = self.block_tables[seq_id]
        for i in range(len(table) - 1, -1, -1):
            if table[i] == block_idx:
                table[i] = new_block_idx
                break

        return new_block_idx

    def free_sequence(self, seq_id: int) -> None:
        """Free physical blocks associated with a sequence according to ref counts."""
        if seq_id in self.block_tables:
            for b in self.block_tables[seq_id]:
                if b in self.block_ref_counts:
                    self.block_ref_counts[b] -= 1
                    if self.block_ref_counts[b] <= 0:
                        del self.block_ref_counts[b]
                        self.free_blocks.append(b)
                else:
                    self.free_blocks.append(b)
            del self.block_tables[seq_id]
            del self.seq_lengths[seq_id]

    def append_token(
        self,
        seq_id: int,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> None:
        """Appends a single token key and value to sequence's cache, honoring Copy-on-Write."""
        if seq_id not in self.block_tables:
            self.allocate_sequence(seq_id)

        seq_len = self.seq_lengths[seq_id]
        token_offset = seq_len % self.block_size

        if token_offset == 0:
            # Need a brand new block
            if not self.free_blocks:
                raise RuntimeError("Out of KV cache blocks.")
            block_idx = self.free_blocks.pop(0)
            self.block_ref_counts[block_idx] = 1
            self.block_tables[seq_id].append(block_idx)
        else:
            block_idx = self.block_tables[seq_id][-1]
            # If the block is shared by another sequence (ref_count > 1), trigger CoW
            if self.block_ref_counts.get(block_idx, 0) > 1:
                block_idx = self.copy_on_write(seq_id, block_idx)

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

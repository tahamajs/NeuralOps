#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Tests for PagedKVCache lifecycle balancing, Copy-on-Write, and page table invariants.
"""

from __future__ import annotations

import pytest
import torch

from neuralops.inference.paged_cache import PagedKVCache


def test_paged_kv_cache_lifecycle_balance_no_leak():
    """Verify allocate -> append -> free cycles balance with zero block leaks."""
    num_blocks = 16
    block_size = 4
    num_heads = 2
    head_dim = 8

    cache = PagedKVCache(
        num_blocks=num_blocks,
        block_size=block_size,
        num_heads=num_heads,
        head_dim=head_dim,
    )

    initial_free = len(cache.free_blocks)
    assert initial_free == num_blocks

    # Perform multiple allocate -> append -> free cycles
    for cycle in range(25):
        seq_id = 100 + cycle
        cache.allocate_sequence(seq_id)
        assert cache.block_table(seq_id) == []
        assert cache.get_block_table(seq_id) == []

        # Append 9 tokens (spans 3 blocks: 4 + 4 + 1)
        for t in range(9):
            k = torch.randn(num_heads, head_dim)
            v = torch.randn(num_heads, head_dim)
            cache.append_token(seq_id, k, v)

        assert len(cache.block_table(seq_id)) == 3
        assert cache.seq_lengths[seq_id] == 9

        # Free sequence and verify zero leak
        cache.free_sequence(seq_id)
        assert len(cache.free_blocks) == num_blocks
        assert seq_id not in cache.block_tables
        assert seq_id not in cache.seq_lengths
        assert len(cache.block_ref_counts) == 0


def test_paged_kv_cache_block_table_returns_page_ids():
    """Verify block_table and get_block_table return exact physical page IDs."""
    cache = PagedKVCache(num_blocks=8, block_size=2, num_heads=1, head_dim=4)
    cache.allocate_sequence(42)

    # Initial empty list
    assert cache.block_table(42) == []
    assert cache.get_block_table(42) == []

    # Unregistered sequence raises KeyError
    with pytest.raises(KeyError, match="Sequence 999 not registered"):
        cache.block_table(999)
    with pytest.raises(KeyError, match="Sequence 999 not registered"):
        cache.get_block_table(999)

    # Append 3 tokens (2 blocks)
    for _ in range(3):
        cache.append_token(42, torch.ones(1, 4), torch.ones(1, 4))

    pages = cache.block_table(42)
    assert len(pages) == 2
    assert pages == cache.get_block_table(42)
    assert pages == cache.block_tables[42]


def test_paged_kv_cache_copy_on_write_fork():
    """Verify sequence forking shares blocks and triggers copy-on-write on append."""
    cache = PagedKVCache(num_blocks=8, block_size=4, num_heads=1, head_dim=2)

    # Sequence 1 appends 6 tokens (2 blocks: block 0 has 4 tokens, block 1 has 2 tokens)
    cache.allocate_sequence(1)
    for i in range(6):
        k = torch.full((1, 2), float(i))
        v = torch.full((1, 2), float(i * 10))
        cache.append_token(1, k, v)

    seq1_initial_pages = list(cache.block_table(1))
    assert len(seq1_initial_pages) == 2

    # Fork sequence 2 from sequence 1
    forked_pages = cache.fork_sequence(1, 2)
    assert forked_pages == seq1_initial_pages
    assert cache.block_table(2) == seq1_initial_pages

    # Both blocks have ref count = 2
    for b in seq1_initial_pages:
        assert cache.block_ref_counts[b] == 2

    # Verify both read identical data
    k1, v1 = cache.get_kv(1)
    k2, v2 = cache.get_kv(2)
    assert torch.equal(k1, k2)
    assert torch.equal(v1, v2)

    # Sequence 2 appends a token (offset is 2 in block 1, which has ref_count 2 -> triggers CoW)
    new_k = torch.full((1, 2), 999.0)
    new_v = torch.full((1, 2), 9990.0)
    cache.append_token(2, new_k, new_v)

    # Sequence 2 now has a new block for the tail
    seq2_pages = cache.block_table(2)
    assert seq2_pages[0] == seq1_initial_pages[0]  # First block still shared
    assert seq2_pages[1] != seq1_initial_pages[1]  # Second block copied

    # Ref count of first block is still 2; second block is unshared
    assert cache.block_ref_counts[seq1_initial_pages[0]] == 2
    assert cache.block_ref_counts[seq1_initial_pages[1]] == 1
    assert cache.block_ref_counts[seq2_pages[1]] == 1

    # Verify Sequence 1 data was NOT modified by Sequence 2's append
    k1_after, v1_after = cache.get_kv(1)
    assert torch.equal(k1, k1_after)
    assert torch.equal(v1, v1_after)

    # Verify Sequence 2 contains the new token
    k2_after, v2_after = cache.get_kv(2)
    assert k2_after.shape[0] == 7
    assert torch.all(k2_after[6] == 999.0)
    assert torch.all(v2_after[6] == 9990.0)

    # Free both sequences and verify zero leak
    cache.free_sequence(1)
    cache.free_sequence(2)
    assert len(cache.free_blocks) == 8
    assert len(cache.block_ref_counts) == 0


def test_paged_kv_cache_defragment_and_fragmentation_stats():
    """Verify fragmentation calculation and defragmentation compaction."""
    cache = PagedKVCache(num_blocks=10, block_size=2, num_heads=1, head_dim=4)

    # Allocate sequences 1, 2, 3
    for s in [1, 2, 3]:
        cache.allocate_sequence(s)
        for _ in range(2):
            cache.append_token(s, torch.ones(1, 4), torch.ones(1, 4))

    # Free middle sequence (creates a hole)
    cache.free_sequence(2)
    assert len(cache.free_blocks) == 8  # 10 - 2

    # Allocate sequence 4
    cache.allocate_sequence(4)
    cache.append_token(4, torch.full((1, 4), 5.0), torch.full((1, 4), 5.0))

    frag_stats = cache.get_fragmentation_stats()
    assert frag_stats["total_blocks"] == 10
    assert frag_stats["allocated_blocks"] == 3

    # Run defragment
    result = cache.defragment()
    assert result["compacted_blocks"] == 3
    post = result["post_stats"]
    assert post["external_fragmentation_ratio"] == 0.0
    assert cache.free_blocks == [3, 4, 5, 6, 7, 8, 9]

    # Verify data after defrag
    k1, _ = cache.get_kv(1)
    k4, _ = cache.get_kv(4)
    assert torch.all(k1 == 1.0)
    assert torch.all(k4 == 5.0)

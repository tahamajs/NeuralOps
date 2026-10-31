#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""Tests for PagedKVCache."""
import torch
import pytest
from neuralops.inference.paged_cache import PagedKVCache


def test_paged_cache_append_and_get():
    cache = PagedKVCache(num_blocks=10, block_size=4, num_heads=2, head_dim=8)
    cache.allocate_sequence(1)

    for i in range(10):
        k = torch.full((2, 8), fill_value=float(i))
        v = torch.full((2, 8), fill_value=float(i * 10))
        cache.append_token(1, k, v)

    assert cache.seq_lengths[1] == 10
    assert len(cache.block_tables[1]) == 3

    k_out, v_out = cache.get_kv(1)
    assert k_out.shape == (10, 2, 8)
    assert v_out.shape == (10, 2, 8)
    assert torch.all(k_out[5] == 5.0)
    assert torch.all(v_out[5] == 50.0)

    cache.free_sequence(1)
    assert len(cache.free_blocks) == 10


@pytest.mark.parametrize(
    "kwargs",
    [
        {"num_blocks": 0, "block_size": 4, "num_heads": 2, "head_dim": 8},
        {"num_blocks": 4, "block_size": -1, "num_heads": 2, "head_dim": 8},
        {"num_blocks": 4, "block_size": 4, "num_heads": True, "head_dim": 8},
    ],
)
def test_paged_cache_rejects_invalid_capacity(kwargs):
    with pytest.raises(ValueError, match="positive integer"):
        PagedKVCache(**kwargs)


def test_paged_cache_rejects_malformed_token_before_allocating():
    cache = PagedKVCache(num_blocks=2, block_size=4, num_heads=2, head_dim=8)

    with pytest.raises(ValueError, match="must both have shape"):
        cache.append_token(7, torch.zeros(2, 7), torch.zeros(2, 8))

    assert 7 not in cache.block_tables
    assert cache.free_blocks == [0, 1]


def test_paged_cache_fork_does_not_silently_discard_existing_child():
    cache = PagedKVCache(num_blocks=4, block_size=4, num_heads=1, head_dim=2)
    cache.allocate_sequence(1)
    cache.allocate_sequence(2)

    with pytest.raises(ValueError, match="already exists"):
        cache.fork_sequence(1, 2)

    assert cache.block_tables == {1: [], 2: []}
    assert cache.free_blocks == [0, 1, 2, 3]


def test_free_sequence_validates_all_refs_before_mutating_cache():
    cache = PagedKVCache(num_blocks=2, block_size=1, num_heads=1, head_dim=2)
    cache.append_token(1, torch.ones(1, 2), torch.ones(1, 2))
    cache.append_token(1, torch.ones(1, 2), torch.ones(1, 2))
    first_block, second_block = cache.block_tables[1]
    first_ref_count = cache.block_ref_counts[first_block]
    del cache.block_ref_counts[second_block]

    with pytest.raises(RuntimeError, match="no live reference counts"):
        cache.free_sequence(1)

    assert cache.block_ref_counts[first_block] == first_ref_count
    assert cache.block_tables[1] == [first_block, second_block]
    assert cache.free_blocks == []


def test_paged_cache_get_kv_zero_copy_view_and_scattered_blocks():
    """Verify get_kv direct zero-copy view for contiguous blocks and index_select for scattered blocks."""
    cache = PagedKVCache(num_blocks=6, block_size=4, num_heads=2, head_dim=8)
    cache.allocate_sequence(1)

    # 1 block appended
    for i in range(4):
        cache.append_token(1, torch.full((2, 8), fill_value=float(i)), torch.full((2, 8), fill_value=float(i)))

    # 1 block: direct view
    k_view, v_view = cache.get_kv(1)
    assert k_view.shape == (4, 2, 8)
    # Check that it views into key_cache without new allocation
    assert k_view.data_ptr() == cache.key_cache[cache.block_tables[1][0]].data_ptr()

    # Append second contiguous block
    for i in range(4, 8):
        cache.append_token(1, torch.full((2, 8), fill_value=float(i)), torch.full((2, 8), fill_value=float(i)))

    k_view2, v_view2 = cache.get_kv(1)
    assert k_view2.shape == (8, 2, 8)
    # Contiguous blocks: still points into the key_cache data
    assert k_view2.data_ptr() == cache.key_cache[cache.block_tables[1][0]].data_ptr()

    # Test get_kv_blocks without sequence materialization
    k_blocks, v_blocks = cache.get_kv_blocks(1)
    assert k_blocks.shape == (2, 4, 2, 8)

    # Now simulate non-contiguous blocks: [3, 0]
    cache.block_tables[2] = [3, 0]
    cache.seq_lengths[2] = 8
    k_scatter, v_scatter = cache.get_kv(2)
    assert k_scatter.shape == (8, 2, 8)


def test_paged_cache_defragmentation_and_telemetry():
    """Verify defragmentation compacts scattered physical blocks and reports honest telemetry."""
    cache = PagedKVCache(num_blocks=10, block_size=2, num_heads=1, head_dim=4)

    # seq 1 takes blocks 0, 1 (4 tokens)
    cache.allocate_sequence(1)
    for i in range(4):
        cache.append_token(1, torch.full((1, 4), float(i + 1)), torch.full((1, 4), float((i + 1) * 10)))

    # seq 2 takes block 2 (2 tokens)
    cache.allocate_sequence(2)
    for i in range(2):
        cache.append_token(2, torch.full((1, 4), float(100 + i)), torch.full((1, 4), float(200 + i)))

    # seq 3 takes block 3 (2 tokens)
    cache.allocate_sequence(3)
    for i in range(2):
        cache.append_token(3, torch.full((1, 4), float(500 + i)), torch.full((1, 4), float(600 + i)))

    # Free sequence 2: block 2 is now free, creating a hole between [0, 1] and [3]
    cache.free_sequence(2)
    assert 2 in cache.free_blocks

    # seq 1 appends another block: gets block 2 (from free_blocks), so seq 1 has blocks [0, 1, 2]
    # To induce non-contiguity, make seq 4 allocate first, taking block 2
    cache.allocate_sequence(4)
    cache.append_token(4, torch.full((1, 4), 77.0), torch.full((1, 4), 88.0))
    # seq 4 gets block 2!
    assert cache.block_tables[4] == [2]

    # seq 1 appends: gets block 4! seq 1 now has blocks [0, 1, 4] -> non-contiguous transition [1 -> 4]!
    cache.append_token(1, torch.full((1, 4), 99.0), torch.full((1, 4), 990.0))
    assert cache.block_tables[1] == [0, 1, 4]

    pre_stats = cache.get_fragmentation_stats()
    assert pre_stats["external_fragmentation_ratio"] > 0.0
    assert pre_stats["allocated_blocks"] == 5  # blocks 0, 1, 4 (seq 1), 2 (seq 4), 3 (seq 3)
    assert pre_stats["total_blocks"] == 10

    # Execute defragmentation
    result = cache.defragment()
    post_stats = result["post_stats"]

    # All active blocks compacted to 0..4, free blocks 5..9
    assert post_stats["allocated_blocks"] == 5
    assert cache.free_blocks == [5, 6, 7, 8, 9]
    assert post_stats["external_fragmentation_ratio"] == 0.0

    # Verify data integrity after defragmentation
    k1, v1 = cache.get_kv(1)
    assert k1.shape == (5, 1, 4)
    assert torch.all(k1[0] == 1.0)
    assert torch.all(k1[4] == 99.0)
    assert torch.all(v1[4] == 990.0)

    k4, v4 = cache.get_kv(4)
    assert torch.all(k4[0] == 77.0)

    k3, v3 = cache.get_kv(3)
    assert torch.all(k3[0] == 500.0)


def test_paged_cache_defragmentation_preserves_cow_sharing():
    """Verify defragmentation preserves Copy-on-Write shared blocks and reference counts."""
    cache = PagedKVCache(num_blocks=8, block_size=2, num_heads=1, head_dim=2)
    cache.allocate_sequence(1)
    for i in range(4):
        cache.append_token(1, torch.full((1, 2), float(i)), torch.full((1, 2), float(i * 10)))

    # Fork sequence 2 from sequence 1 (shares physical blocks)
    cache.fork_sequence(1, 2)
    assert cache.block_tables[1] == cache.block_tables[2]
    shared_blocks = list(cache.block_tables[1])
    for b in shared_blocks:
        assert cache.block_ref_counts[b] == 2

    # Defragment
    result = cache.defragment()
    assert result["compacted_blocks"] == 2

    # Blocks should still be shared in the compacted table
    new_blocks_1 = cache.block_tables[1]
    new_blocks_2 = cache.block_tables[2]
    assert new_blocks_1 == new_blocks_2 == [0, 1]
    assert cache.block_ref_counts[0] == 2
    assert cache.block_ref_counts[1] == 2

    # Verify both can read correct data
    k1, v1 = cache.get_kv(1)
    k2, v2 = cache.get_kv(2)
    assert torch.equal(k1, k2)
    assert torch.equal(v1, v2)

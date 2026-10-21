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

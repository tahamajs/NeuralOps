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

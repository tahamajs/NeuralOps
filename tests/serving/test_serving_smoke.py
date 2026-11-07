#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""Smoke tests for neuralops.serving subsystem."""

import torch
from neuralops.serving.paged_cache_manager import PagedKVCache


def test_serving_paged_kv_cache_lifecycle_smoke():
    """Verify serving PagedKVCache sequence allocation and deallocation on CPU."""
    cache = PagedKVCache(
        num_blocks=16,
        block_size=8,
        num_heads=2,
        head_dim=16,
        dtype=torch.float32,
        device="cpu",
    )
    assert cache.num_blocks == 16
    assert len(cache.free_blocks) == 16

    # Allocate sequence
    seq_id = 42
    cache.allocate_sequence(seq_id)
    assert seq_id in cache.block_tables

    # Append token
    k = torch.randn(2, 16)
    v = torch.randn(2, 16)
    cache.append_token(seq_id, k, v)
    assert cache.seq_lengths[seq_id] == 1

    # Free sequence
    cache.free_sequence(seq_id)
    assert seq_id not in cache.block_tables

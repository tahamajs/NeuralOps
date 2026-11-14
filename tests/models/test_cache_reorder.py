#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""Tests for cache reordering with UnifiedHybridCache."""

from __future__ import annotations

import torch

from neuralops.models.cache import UnifiedHybridCache
from neuralops.models.modeling import NeuralOpsForCausalLM


def test_unified_hybrid_cache_iter_and_reorder() -> None:
    cache = UnifiedHybridCache(num_layers=2)
    # Populate layers with dummy (k, v) states
    k0 = torch.tensor([[[[1.0]], [[2.0]]]])  # shape [1, 2, 1, 1] -> batch 1
    v0 = torch.tensor([[[[3.0]], [[4.0]]]])
    k1 = torch.tensor([[[[5.0]], [[6.0]]]])
    v1 = torch.tensor([[[[7.0]], [[8.0]]]])

    # Batch of size 3 for beam search test
    k_batch = torch.tensor([
        [[[1.0]]],  # beam 0
        [[[2.0]]],  # beam 1
        [[[3.0]]],  # beam 2
    ])  # [3, 1, 1, 1]
    v_batch = torch.tensor([
        [[[10.0]]],
        [[[20.0]]],
        [[[30.0]]],
    ])

    cache[0] = (k_batch, v_batch)
    cache[1] = (k_batch * 2, v_batch * 2)
    cache._seen_tokens = 5

    # Test __iter__
    items = list(cache)
    assert len(items) == 2
    assert items[0] == (k_batch, v_batch)

    # Beam indices to reorder: select beam 2, beam 0, beam 1
    beam_idx = torch.tensor([2, 0, 1], dtype=torch.long)
    reordered = cache.reorder_cache(beam_idx)

    assert isinstance(reordered, UnifiedHybridCache)
    assert reordered._seen_tokens == 5
    assert len(reordered) == 2

    # Check layer 0
    k_reordered, v_reordered = reordered[0]
    expected_k = torch.tensor([[[[3.0]]], [[[1.0]]], [[[2.0]]]])
    expected_v = torch.tensor([[[[30.0]]], [[[10.0]]], [[[20.0]]]])
    assert torch.allclose(k_reordered, expected_k)
    assert torch.allclose(v_reordered, expected_v)


def test_modeling_reorder_cache_delegates_to_unified_hybrid_cache() -> None:
    cache = UnifiedHybridCache(num_layers=1)
    k_batch = torch.tensor([[[[10.0]]], [[[20.0]]]])
    v_batch = torch.tensor([[[[100.0]]], [[[200.0]]]])
    cache[0] = (k_batch, v_batch)
    cache._seen_tokens = 3

    beam_idx = torch.tensor([1, 0], dtype=torch.long)
    reordered = NeuralOpsForCausalLM._reorder_cache(cache, beam_idx)

    assert isinstance(reordered, UnifiedHybridCache)
    assert reordered._seen_tokens == 3
    assert torch.allclose(reordered[0][0], torch.tensor([[[[20.0]]], [[[10.0]]]]))
    assert torch.allclose(reordered[0][1], torch.tensor([[[[200.0]]], [[[100.0]]]]))

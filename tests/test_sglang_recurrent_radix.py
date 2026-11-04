#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Tests for SGLang Recurrent Radix Serving Integration and Plugin Registration.
"""

from unittest.mock import patch
import pytest
import torch

from neuralops.core.errors import IntegrationNotImplementedError
from neuralops.integrations.sglang import (
    RadixRecurrentNode,
    RadixRecurrentTree,
    register_neuralops_sglang_plugin,
    sglang_is_available,
)


def test_radix_recurrent_tree_insert_and_exact_match():
    """Verify RadixRecurrentTree exact prefix matching and recurrent state retrieval."""
    tree = RadixRecurrentTree()
    state_a = torch.randn(2, 4, 16)
    tokens_a = [101, 2054, 2003, 1037]

    tree.insert(tokens_a, recurrent_state=state_a)
    matched_tokens, cached_state = tree.match_prefix(tokens_a)

    assert matched_tokens == tokens_a
    assert cached_state is not None
    assert torch.equal(cached_state, state_a)


def test_radix_recurrent_tree_match_and_step_partial_and_split():
    """Verify match_and_step behavior for partial prefixes and tree branching."""
    tree = RadixRecurrentTree()
    state_1 = torch.randn(1, 8)
    state_2 = torch.randn(1, 8)

    # Insert prefix [10, 20, 30]
    tree.insert([10, 20, 30], state_1)

    # Branch with [10, 20, 40] -> triggers node splitting
    tree.insert([10, 20, 40], state_2)

    # Query with [10, 20, 30, 99, 100]
    matched, cached_state, remaining = tree.match_and_step([10, 20, 30, 99, 100])
    assert matched == [10, 20, 30]
    assert cached_state is not None
    assert torch.equal(cached_state, state_1)
    assert remaining == [99, 100]

    # Query with [10, 20, 40, 50]
    matched, cached_state, remaining = tree.match_and_step([10, 20, 40, 50])
    assert matched == [10, 20, 40]
    assert cached_state is not None
    assert torch.equal(cached_state, state_2)
    assert remaining == [50]

    # Query with only common prefix [10, 20, 99]
    matched, cached_state, remaining = tree.match_and_step([10, 20, 99])
    assert matched == [10, 20]
    # The split internal node [10, 20] was not inserted with a terminal state
    assert cached_state is None
    assert remaining == [99]


def test_radix_recurrent_tree_match_and_step_layer_idx():
    """Verify match_and_step per-layer state slicing."""
    tree = RadixRecurrentTree()
    layer_states = [torch.tensor([1.0, 2.0]), torch.tensor([3.0, 4.0])]
    tokens = [1, 2, 3]
    tree.insert(tokens, layer_states)

    matched, state_l0, remaining = tree.match_and_step(tokens, layer_idx=0)
    assert matched == tokens
    assert remaining == []
    assert torch.equal(state_l0, layer_states[0])

    matched, state_l1, remaining = tree.match_and_step(tokens, layer_idx=1)
    assert torch.equal(state_l1, layer_states[1])


def test_radix_recurrent_tree_pinning_and_lru_eviction():
    """Verify multi-tenant reference counting and LRU eviction."""
    tree = RadixRecurrentTree()
    state_a = torch.randn(1, 4)
    state_b = torch.randn(1, 4)

    tree.insert([1, 2, 3], state_a)
    tree.insert([4, 5, 6], state_b)

    # Pin sequence [1, 2, 3]
    pinned = tree.inc_ref([1, 2, 3])
    assert pinned > 0

    # Evict 1 node: only [4, 5, 6] should be eligible since [1, 2, 3] is pinned
    evicted = tree.evict(1)
    assert evicted == 1

    # [1, 2, 3] must still match
    matched, state = tree.match_prefix([1, 2, 3])
    assert matched == [1, 2, 3]
    assert state is not None

    # [4, 5, 6] was evicted
    matched_b, _ = tree.match_prefix([4, 5, 6])
    assert matched_b == []

    # Unpin [1, 2, 3] and evict
    unpinned = tree.dec_ref([1, 2, 3])
    assert unpinned > 0
    evicted_unpinned = tree.evict(1)
    assert evicted_unpinned == 1
    matched_after, _ = tree.match_prefix([1, 2, 3])
    assert matched_after == []


def test_radix_recurrent_tree_memory_bounds():
    """Verify automatic LRU eviction under memory bounds (max_cache_bytes)."""
    # Each tensor of shape (100, 100) float32 is 40,000 bytes
    t1 = torch.zeros(100, 100, dtype=torch.float32)
    t2 = torch.zeros(100, 100, dtype=torch.float32)

    # Set budget to accommodate only 1 state tensor (~50,000 bytes)
    tree = RadixRecurrentTree(max_cache_bytes=50000)
    tree.insert([10, 20], t1)
    assert tree.total_cache_bytes >= 40000

    # Inserting second large state exceeds budget and triggers eviction of oldest unpinned
    tree.insert([30, 40], t2)
    assert tree.total_cache_bytes <= 50000


def test_register_neuralops_sglang_plugin_contract():
    """Verify register_neuralops_sglang_plugin fail-closed vs registration behavior."""
    if not sglang_is_available():
        with pytest.raises(IntegrationNotImplementedError) as exc_info:
            register_neuralops_sglang_plugin()
        assert "pip install 'sglang>=0.4.0'" in str(exc_info.value)
    else:
        res = register_neuralops_sglang_plugin()
        assert res is True


def test_register_neuralops_sglang_plugin_mock_flow():
    """Mock sglang availability and registration execution flow."""
    with patch("neuralops.integrations.sglang.sglang_is_available", return_value=True):
        with patch("neuralops.integrations.sglang_backend.registry.register", return_value=True) as mock_reg:
            res = register_neuralops_sglang_plugin()
            assert res is True
            mock_reg.assert_called_once()

    with patch("neuralops.integrations.sglang.sglang_is_available", return_value=False):
        with pytest.raises(IntegrationNotImplementedError) as exc_info:
            register_neuralops_sglang_plugin()
        assert "SGLang is not installed" in str(exc_info.value)

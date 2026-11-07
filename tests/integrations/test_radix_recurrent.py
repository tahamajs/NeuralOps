#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
r"""
Tests for production-grade SGLang RadixRecurrentTree and SGLangRadixAdapter.

Verifies:
1. Exact prefix matching and intermediate state recovery.
2. Node splitting when two prompts share a partial prefix.
3. LRU cache eviction under memory constraints.
4. Mathematical parity between scanning from cached state vs scanning full sequence from scratch ($S_t = \gamma S_{t-1} + K_t^T V_t$).
5. Concurrency safety under parallel insertions and lookups.
6. Fail-closed handling when SGLang runtime is required but unavailable.
"""

from __future__ import annotations

import threading
from unittest.mock import patch
import pytest
import torch

from neuralops.core.exceptions import IntegrationNotImplementedError
from neuralops.integrations.sglang import (
    RadixRecurrentNode,
    RadixRecurrentTree,
    SGLangRadixAdapter,
    get_sglang_radix_adapter,
    sglang_is_available,
)
from neuralops.integrations.sglang.radix_recurrent import linear_recurrent_step


def test_1_exact_prefix_matching_and_intermediate_state_recovery():
    """Test 1: Exact prefix matching and intermediate state recovery."""
    tree = RadixRecurrentTree()
    num_heads, d_k, d_v = 4, 16, 16
    state_a = torch.randn(num_heads, d_k, d_v)
    prompt_a = [101, 2054, 2003, 1037, 3231]

    # Insert prompt and check node structure
    leaf = tree.insert(prompt_a, recurrent_state=state_a)
    assert leaf is not None
    assert isinstance(leaf, RadixRecurrentNode)
    assert leaf.state_tensor is not None
    assert torch.equal(leaf.state_tensor, state_a)
    assert leaf.last_accessed_timestamp > 0

    # match_prefix returns token list and state
    matched_tokens, state = tree.match_prefix(prompt_a)
    assert matched_tokens == prompt_a
    assert state is not None
    assert torch.equal(state, state_a)

    # match_prefix_node returns (node, matched_len)
    matched_node, matched_len = tree.match_prefix_node(prompt_a)
    assert matched_node is leaf
    assert matched_len == len(prompt_a)
    assert matched_node.token_id == prompt_a[0]

    # Partial prefix match
    matched_tokens, state, rem = tree.match_and_step(prompt_a + [9999, 8888])
    assert matched_tokens == prompt_a
    assert torch.equal(state, state_a)
    assert rem == [9999, 8888]


def test_2_node_splitting_when_prompts_share_partial_prefix():
    """Test 2: Node splitting when two prompts share a partial prefix."""
    tree = RadixRecurrentTree()
    h, dk, dv = 2, 8, 8

    # Prefix A: [10, 20, 30, 40]
    tokens_a = [10, 20, 30, 40]
    state_a = torch.randn(h, dk, dv)
    node_a = tree.insert(tokens_a, state_a)

    # Prefix B: [10, 20, 50, 60] - shares [10, 20]
    tokens_b = [10, 20, 50, 60]
    state_b = torch.randn(h, dk, dv)
    node_b = tree.insert(tokens_b, state_b)

    # Verify root has split node child [10, 20]
    root = tree.root
    assert 10 in root.children
    split_node = root.children[10]
    assert split_node.token_chunk == [10, 20]
    assert len(split_node.children) == 2
    assert 30 in split_node.children
    assert 50 in split_node.children

    # Check child chunks
    child_30 = split_node.children[30]
    assert child_30.token_chunk == [30, 40]
    assert torch.equal(child_30.recurrent_state, state_a)

    child_50 = split_node.children[50]
    assert child_50.token_chunk == [50, 60]
    assert torch.equal(child_50.recurrent_state, state_b)

    # Prefix match queries
    matched_a, sa = tree.match_prefix([10, 20, 30, 40])
    assert matched_a == tokens_a
    assert torch.equal(sa, state_a)

    matched_b, sb = tree.match_prefix([10, 20, 50, 60])
    assert matched_b == tokens_b
    assert torch.equal(sb, state_b)

    matched_split, split_s = tree.match_prefix([10, 20, 99])
    assert matched_split == [10, 20]
    assert split_s is None


def test_3_lru_cache_eviction_under_memory_constraints():
    """Test 3: LRU cache eviction under memory constraints."""
    # Each tensor (4, 32, 32) float32 is 4 * 32 * 32 * 4 = 16,384 bytes
    state1 = torch.zeros(4, 32, 32, dtype=torch.float32)
    state2 = torch.zeros(4, 32, 32, dtype=torch.float32)
    state3 = torch.zeros(4, 32, 32, dtype=torch.float32)

    # Budget for 2 states (~35,000 bytes)
    tree = RadixRecurrentTree(max_cache_bytes=35000)

    tree.insert([1, 2, 3], state1)
    tree.insert([4, 5, 6], state2)
    assert tree.total_cache_bytes >= 32768
    assert tree.total_cache_bytes <= 35000

    # Pin first sequence [1, 2, 3]
    tree.inc_ref([1, 2, 3])

    # Inserting state 3 exceeds memory budget, forcing eviction
    # [1, 2, 3] is pinned, so [4, 5, 6] must be evicted
    tree.insert([7, 8, 9], state3)
    assert tree.total_cache_bytes <= 35000

    m1, _ = tree.match_prefix([1, 2, 3])
    assert m1 == [1, 2, 3]

    m2, _ = tree.match_prefix([4, 5, 6])
    assert m2 == []  # Evicted

    m3, _ = tree.match_prefix([7, 8, 9])
    assert m3 == [7, 8, 9]

    # Explicit evict_lru call
    tree.dec_ref([1, 2, 3])
    bytes_freed = tree.evict_lru(num_bytes_to_free=16000)
    assert bytes_freed >= 16384


def test_4_mathematical_parity_cached_vs_full_scratch_scan():
    r"""
    Test 4: Mathematical parity between scanning from cached state vs scanning full sequence.
    Invariant: Zero fake artifacts, exact recurrence $S_t = \gamma S_{t-1} + K_t^T V_t$.
    """
    seq_len = 16
    prefix_len = 10
    h, dk, dv = 4, 16, 16
    gamma = 0.95

    # Deterministic keys and values
    torch.manual_seed(42)
    k = torch.randn(seq_len, h, dk)
    v = torch.randn(seq_len, h, dv)
    token_ids = list(range(100, 100 + seq_len))

    # 1. Full scan from scratch S_0 = 0
    state_full = torch.zeros(h, dk, dv)
    for t in range(seq_len):
        state_full = linear_recurrent_step(state_full, k[t], v[t], gamma=gamma)

    # 2. Prefill scan up to prefix_len
    state_prefix = torch.zeros(h, dk, dv)
    for t in range(prefix_len):
        state_prefix = linear_recurrent_step(state_prefix, k[t], v[t], gamma=gamma)

    # Cache intermediate prefix state in tree
    tree = RadixRecurrentTree()
    tree.insert(token_ids[:prefix_len], recurrent_state=state_prefix)

    # 3. Retrieve cached prefix and resume scan for suffix
    matched_tokens, cached_state, rem_tokens = tree.match_and_step(token_ids)
    assert matched_tokens == token_ids[:prefix_len]
    assert rem_tokens == token_ids[prefix_len:]
    assert cached_state is not None

    state_resumed = cached_state.clone()
    for t in range(prefix_len, seq_len):
        state_resumed = linear_recurrent_step(state_resumed, k[t], v[t], gamma=gamma)

    # 4. Strict numerical parity check: |S_resumed - S_full| < 1e-6
    max_diff = (state_resumed - state_full).abs().max().item()
    assert max_diff < 1e-6, f"Recurrent state divergence: {max_diff}"
    assert torch.allclose(state_resumed, state_full, atol=1e-6, rtol=1e-5)


def test_5_concurrency_safety_thread_safe_radix():
    """Test 5: Concurrency safety with multithreaded insertion and prefix lookups."""
    tree = RadixRecurrentTree()
    num_threads = 8
    items_per_thread = 20
    errors = []

    def worker(worker_id: int):
        try:
            for i in range(items_per_thread):
                tokens = [worker_id * 1000 + i, 1, 2, 3]
                state = torch.ones(2, 4, 4) * (worker_id * 100 + i)
                tree.insert(tokens, recurrent_state=state)
                matched, s = tree.match_prefix(tokens)
                assert matched == tokens
                assert s is not None
        except Exception as e:
            errors.append(e)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(num_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, f"Thread concurrency errors encountered: {errors}"


def test_6_sglang_adapter_and_fail_closed_handling():
    """Test 6: SGLangRadixAdapter and fail-closed handling."""
    adapter = get_sglang_radix_adapter()
    assert isinstance(adapter, SGLangRadixAdapter)
    assert adapter.tree is not None

    tokens = [10, 20, 30]
    st = torch.randn(2, 4, 4)
    adapter.insert_prefix_state(tokens, st)
    matched_len, retrieved = adapter.match_prefix_state(tokens)
    assert matched_len == len(tokens)
    assert retrieved is not None
    assert torch.equal(retrieved, st)

    # Fail closed when sglang is explicitly required but unavailable
    with patch("neuralops.integrations.sglang.sglang_is_available", return_value=False):
        with pytest.raises(IntegrationNotImplementedError) as exc_info:
            get_sglang_radix_adapter(require_sglang=True)
        assert "SGLang runtime is not installed" in str(exc_info.value)

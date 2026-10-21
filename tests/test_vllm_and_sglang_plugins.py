#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors

import pytest
import torch
import torch.nn as nn

from neuralops.integrations.vllm import (
    NeuralOpsvLLMMetadata,
    NeuralOpsvLLMImpl,
    NeuralOpsvLLMAttentionBackend,
    NeuralOpsForCausalLMvLLM,
    register_neuralops_vllm_plugin,
)
from neuralops.integrations.serving import (
    VLLMAttentionAdapter,
    TGIAttentionAdapter,
    SGLangRadixLinearCacheAdapter,
)


def test_vllm_backend_and_metadata_interface():
    assert NeuralOpsvLLMAttentionBackend.get_name() == "NEURALOPS_HYBRID"
    assert NeuralOpsvLLMAttentionBackend.get_impl_cls() is NeuralOpsvLLMImpl
    assert NeuralOpsvLLMAttentionBackend.get_metadata_cls() is NeuralOpsvLLMMetadata

    shape = NeuralOpsvLLMAttentionBackend.get_kv_cache_shape(
        num_blocks=32, block_size=16, num_kv_heads=4, head_size=64
    )
    assert shape == (32, 4, 64, 64)


def test_vllm_impl_fails_closed_until_scheduler_and_cache_contract_are_implemented():
    impl = NeuralOpsvLLMImpl(num_heads=2, head_size=16, scale=1.0 / 4.0)
    q = torch.randn(1, 1, 32)
    with pytest.raises(NotImplementedError, match="paged-cache write"):
        impl.forward(q, q, q, kv_cache=None, attn_metadata=None)


def test_vllm_model_wrapper_does_not_initialize_random_checkpoint_weights():
    class DummyConfig:
        hidden_size = 32
        d_model = 32
        vocab_size = 100
        num_attention_heads = 2
        num_hidden_layers = 1

    with pytest.raises(NotImplementedError, match="random weights"):
        NeuralOpsForCausalLMvLLM(DummyConfig())

    assert register_neuralops_vllm_plugin() is False


def test_sglang_radix_linear_cache_adapter():
    cache = SGLangRadixLinearCacheAdapter(max_cached_nodes=4)

    tokens_a = [101, 2054, 2003, 1037]
    state_a = torch.randn(2, 4, 16, 16)
    cache.cache_finished_req(tokens_a, state_a)

    matched_len, matched_state = cache.match_prefix(tokens_a)
    assert matched_len == 4
    assert torch.equal(matched_state, state_a)

    query_tokens = [101, 2054, 2003, 1037, 5000, 6000]
    matched_len, matched_state = cache.match_prefix(query_tokens)
    assert matched_len == 4
    assert torch.equal(matched_state, state_a)

    cache.inc_lock_ref(tokens_a)
    tokens_b = [1, 2, 3]
    cache.cache_finished_req(tokens_b, torch.randn(1))
    evicted = cache.evict(1)
    assert evicted == 1
    assert tuple(tokens_a) in cache.tree
    assert tuple(tokens_b) not in cache.tree

    cache.dec_lock_ref(tokens_a)
    cache.evict(1)
    assert tuple(tokens_a) not in cache.tree


def test_sglang_prefix_cache_does_not_evict_on_update_and_rejects_all_locked_overflow():
    cache = SGLangRadixLinearCacheAdapter(max_cached_nodes=1)
    cache.cache_unfinished_req([1, 2], "state-a")
    cache.cache_finished_req([1, 2], "updated-state")

    assert cache.tree[(1, 2)]["state"] == "updated-state"
    with pytest.raises(RuntimeError, match="all entries are locked"):
        cache.cache_finished_req([3, 4], "state-b")
    assert list(cache.tree) == [(1, 2)]


def test_sglang_prefix_cache_validates_capacity():
    with pytest.raises(ValueError, match="positive integer"):
        SGLangRadixLinearCacheAdapter(max_cached_nodes=0)

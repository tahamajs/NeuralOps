#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Tests for NeuralOpsHybridHFCache and RadixStateCache state extraction.
Ensures _extract_state_dict handles key_cache as List[Tensor] without unpacking errors.
"""

import torch

from neuralops.surgery.hf_cache_bridge import NeuralOpsHybridHFCache, SGLangRadixStateAdapter


def test_extract_state_dict_from_hybrid_cache():
    """Verify _extract_state_dict does not crash when state.key_cache is a list of tensors."""
    cache = NeuralOpsHybridHFCache()
    # Simulate a populated key_cache
    k_tensor = torch.randn(1, 4, 8, 32)
    cache.key_cache = [k_tensor]

    state_dict = SGLangRadixStateAdapter._extract_state_dict(cache)
    assert 0 in state_dict
    assert torch.equal(state_dict[0], k_tensor)


def test_radix_cache_insert_with_hybrid_cache():
    """Verify SGLangRadixStateAdapter.insert works with NeuralOpsHybridHFCache."""
    cache = NeuralOpsHybridHFCache()
    k_tensor = torch.randn(1, 4, 8, 32)
    cache.key_cache = [k_tensor]

    radix_tree = SGLangRadixStateAdapter(max_cached_nodes=10)
    node = radix_tree.insert(token_ids=[1, 2, 3], state=cache)
    assert node is not None
    assert 0 in node.prefix_state

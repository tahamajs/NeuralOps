#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""Tests for UnifiedHybridCache._seen_tokens updating."""

from __future__ import annotations

import torch

from neuralops.models.cache import UnifiedHybridCache


def test_cache_seen_tokens_tuple_state() -> None:
    cache = UnifiedHybridCache(num_layers=2)
    assert cache._seen_tokens == 0

    # Layer 0 update with seq_len = 4
    # 4D tensor: [batch, heads, seq_len, head_dim]
    k0 = torch.zeros(1, 4, 4, 32)
    v0 = torch.zeros(1, 4, 4, 32)
    cache.update_layer(layer_idx=0, state=(k0, v0), layer_type="full")
    assert cache._seen_tokens == 4

    # Layer 1 update for same step should not increment _seen_tokens
    k1 = torch.zeros(1, 4, 4, 32)
    v1 = torch.zeros(1, 4, 4, 32)
    cache.update_layer(layer_idx=1, state=(k1, v1), layer_type="full")
    assert cache._seen_tokens == 4

    # Next decode step at layer 0 with seq_len = 1
    k0_next = torch.zeros(1, 4, 1, 32)
    v0_next = torch.zeros(1, 4, 1, 32)
    cache.update_layer(layer_idx=0, state=(k0_next, v0_next), layer_type="full")
    assert cache._seen_tokens == 5


def test_cache_seen_tokens_tensor_and_fallback_state() -> None:
    cache = UnifiedHybridCache(num_layers=1)

    # Tensor state with seq_len = 3
    t_state = torch.zeros(1, 2, 3, 16)
    cache.update_layer(layer_idx=0, state=t_state, layer_type="recurrent")
    assert cache._seen_tokens == 3

    # Non-tensor or scalar state falls back to 1 added token
    cache.update_layer(layer_idx=0, state="arbitrary_state", layer_type="custom")
    assert cache._seen_tokens == 4

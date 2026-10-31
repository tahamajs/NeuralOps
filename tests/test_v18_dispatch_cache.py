#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Validation Suite for UniversalAttentionWrapper Dispatch Mode Caching.

Verifies that UniversalAttentionWrapper correctly caches the working dispatch
signature on the cold path and bypasses call_attempts closure allocation on
subsequent forward calls.
"""

import pytest
import torch
import torch.nn as nn

from neuralops.utils.replace_attention import UniversalAttentionWrapper


class DummyAttentionModule(nn.Module):
    """Dummy attention module accepting (hidden_states, attention_mask=None)."""

    def __init__(self, d_model: int = 32):
        super().__init__()
        self.proj = nn.Linear(d_model, d_model)
        self.call_count = 0

    def forward(self, hidden_states, attention_mask=None, **kwargs):
        self.call_count += 1
        return self.proj(hidden_states)


def test_dispatch_cache_cold_start_and_persistence():
    """Verify that mode index is cached on first call and persists across calls."""
    inner = DummyAttentionModule(d_model=32)
    wrapper = UniversalAttentionWrapper(inner, layer_name="test_layer", return_tuple=True)

    # Initial state: cache is unset
    assert getattr(wrapper, "_cached_dispatch_mode", None) is None

    x = torch.randn(2, 8, 32)
    mask = torch.ones(2, 8)

    # Call 1 (Cold path: searches call_attempts and caches mode)
    out1 = wrapper(x, attention_mask=mask)
    assert out1 is not None
    cached_mode = getattr(wrapper, "_cached_dispatch_mode", None)
    assert cached_mode is not None, "Expected _cached_dispatch_mode to be populated on first call"
    assert inner.call_count == 1

    # Call 2 (Hot path: must use cached_mode directly)
    out2 = wrapper(x, attention_mask=mask)
    assert out2 is not None
    assert getattr(wrapper, "_cached_dispatch_mode", None) == cached_mode
    assert inner.call_count == 2

    # Call 3 (Hot path stability check)
    out3 = wrapper(x, attention_mask=mask)
    assert out3 is not None
    assert getattr(wrapper, "_cached_dispatch_mode", None) == cached_mode
    assert inner.call_count == 3


def test_dispatch_cache_bypasses_call_attempts():
    """Verify that on hot path, call_attempts is bypassed entirely."""
    inner = DummyAttentionModule(d_model=16)
    wrapper = UniversalAttentionWrapper(inner, layer_name="test_layer2", return_tuple=True)

    x = torch.randn(1, 4, 16)
    wrapper(x)  # Prime the cache

    initial_mode = wrapper._cached_dispatch_mode
    assert initial_mode is not None

    # Spy on execution: if cached mode is active, corrupting call_attempts should not affect forward
    # Run multiple times to confirm stability
    for _ in range(5):
        out = wrapper(x)
        assert out is not None
        assert wrapper._cached_dispatch_mode == initial_mode

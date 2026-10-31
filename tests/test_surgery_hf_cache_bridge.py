#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""Unit tests for Phase 5 NeuralOpsHybridHFCache."""

import torch
import pytest

from neuralops.surgery.hf_cache_bridge import NeuralOpsHybridHFCache


def test_hybrid_hf_cache_recurrent_and_kv():
    cache = NeuralOpsHybridHFCache()
    assert cache.get_seq_length() == 0

    B, H, T, D = 1, 4, 8, 16
    k0 = torch.randn(B, H, T, D)
    v0 = torch.randn(B, H, T, D)
    cache.update(k0, v0, layer_idx=0)
    assert cache.get_seq_length() == 8

    S1 = torch.randn(B, H, D, D)
    cache.update_recurrent_state(layer_idx=1, state=S1, num_tokens=0)
    ret_s1 = cache.get_recurrent_state(layer_idx=1)
    assert ret_s1 is not None
    assert torch.allclose(ret_s1, S1)

    k0_next = torch.randn(B, H, 1, D)
    v0_next = torch.randn(B, H, 1, D)
    cache.update(k0_next, v0_next, layer_idx=0)
    assert cache.get_seq_length() == 9


def test_wrap_model_for_generation_preserves_hybrid_cache():
    from neuralops.surgery.hf_cache_bridge import wrap_model_for_generation

    class MockGenModel(torch.nn.Module):
        def generate(self):
            return None

        def prepare_inputs_for_generation(self, input_ids, **kwargs):
            return {"input_ids": input_ids}

    mod = MockGenModel()
    wrapped = wrap_model_for_generation(mod)

    # When kwargs has past_key_values as NeuralOpsHybridHFCache, it must be forwarded
    cache = NeuralOpsHybridHFCache()
    inputs = wrapped.prepare_inputs_for_generation(torch.tensor([[1, 2]]), past_key_values=cache, use_cache=True)
    assert inputs.get("past_key_values") is cache

    # When kwargs has no past_key_values, a fresh cache is initialized
    inputs_fresh = wrapped.prepare_inputs_for_generation(torch.tensor([[1, 2]]), use_cache=True)
    assert isinstance(inputs_fresh.get("past_key_values"), NeuralOpsHybridHFCache)

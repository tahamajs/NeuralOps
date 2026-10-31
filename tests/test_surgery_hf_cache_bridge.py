#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""Unit tests for Phase 5 NeuralOpsHybridHFCache and HFCacheBridge."""

import pytest
import torch

from neuralops.surgery.hf_cache_bridge import HFCacheBridge, NeuralOpsHybridHFCache


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


def test_hf_cache_bridge_handles_4d_kv_and_4d_recurrent_state_matrices():
    """Verify HFCacheBridge handles both 4D KV (B, H, L, D) and 4D recurrent state (B, H, Dk, Dv)."""
    cache = HFCacheBridge()
    B, H, Dk, Dv = 2, 4, 32, 32

    # Layer 0: Linear Recurrent Layer with 4D state matrix (B, H, Dk, Dv)
    state_step1 = torch.randn(B, H, Dk, Dv)
    cache.update(state_step1, state_step1, layer_idx=0, cache_kwargs={"is_recurrent": True, "num_tokens": 1})

    assert cache.is_recurrent_layer(0)
    assert cache.get_seq_length(0) == 1  # Must NOT be 32 (Dk)!
    rec_s = cache.get_recurrent_state(0)
    assert rec_s.shape == (B, H, Dk, Dv)

    # Step 2 on Layer 0: Next recurrent step must replace, NOT cat along dim=-2
    state_step2 = torch.randn(B, H, Dk, Dv)
    k_out, v_out = cache.update(state_step2, None, layer_idx=0)
    assert k_out.shape == (B, H, Dk, Dv)
    assert cache.get_seq_length(0) == 2
    assert cache.key_cache[0].shape == (B, H, Dk, Dv)  # If it concatenated, it would be (B, H, 64, 32)!

    # Layer 1: Standard 4D KV attention (B, H, L, D)
    k1 = torch.randn(B, H, 2, Dv)
    v1 = torch.randn(B, H, 2, Dv)
    cache.update(k1, v1, layer_idx=1)
    assert not cache.is_recurrent_layer(1)
    assert cache.get_seq_length(1) == 2
    assert cache.key_cache[1].shape == (B, H, 2, Dv)

    # Decode step on Layer 1: appends 1 token along dim=-2
    k1_next = torch.randn(B, H, 1, Dv)
    v1_next = torch.randn(B, H, 1, Dv)
    cache.update(k1_next, v1_next, layer_idx=1)
    assert cache.key_cache[1].shape == (B, H, 3, Dv)
    assert cache.get_seq_length(1) == 3


def test_hf_cache_bridge_crop_preserves_recurrent_state_matrices():
    """Verify crop() does not slice or corrupt 4D recurrent state matrices."""
    cache = HFCacheBridge()
    B, H, Dk, Dv = 1, 2, 16, 16

    # Setup Layer 0 recurrent state
    S = torch.randn(B, H, Dk, Dv)
    cache.update_recurrent_state(recurrent_states=S, layer_idx=0, num_tokens=5)

    # Setup Layer 1 standard KV cache with 5 tokens
    k = torch.randn(B, H, 5, Dv)
    v = torch.randn(B, H, 5, Dv)
    cache.update(k, v, layer_idx=1)

    # Crop to 3 tokens (e.g. during speculative decoding verification)
    cache.crop(3)

    # Recurrent state must retain its full (B, H, Dk, Dv) matrix dimensions
    assert cache.key_cache[0].shape == (B, H, Dk, Dv)
    assert torch.allclose(cache.get_recurrent_state(0), S)

    # Standard KV cache must be cropped to 3 tokens
    assert cache.key_cache[1].shape == (B, H, 3, Dv)
    assert cache.get_seq_length(0) == 3
    assert cache.get_seq_length(1) == 3


def test_hf_cache_bridge_batch_operations():
    """Verify beam search reordering and batch indexing on mixed KV and recurrent states."""
    cache = HFCacheBridge()
    B, H, L, D = 3, 2, 4, 8

    # Layer 0 recurrent state (B, H, D, D)
    S = torch.randn(B, H, D, D)
    cache.update_recurrent_state(0, S, num_tokens=L)

    # Layer 1 KV cache (B, H, L, D)
    k = torch.randn(B, H, L, D)
    v = torch.randn(B, H, L, D)
    cache.update(k, v, layer_idx=1)

    # Beam reorder: select batch indices [2, 0]
    beam_idx = torch.tensor([2, 0], dtype=torch.long)
    cache.reorder_cache(beam_idx)

    assert cache.key_cache[0].shape == (2, H, D, D)
    assert cache.key_cache[1].shape == (2, H, L, D)
    assert torch.allclose(cache.key_cache[0][0], S[2])
    assert torch.allclose(cache.key_cache[0][1], S[0])


def test_hf_cache_bridge_legacy_conversion():
    """Verify from_legacy_cache and to_legacy_cache with recurrent states."""
    B, H, D = 1, 2, 8
    S0 = torch.randn(B, H, D, D)
    k1 = torch.randn(B, H, 3, D)
    v1 = torch.randn(B, H, 3, D)

    legacy_tuple = ((S0, S0), (k1, v1))
    cache = HFCacheBridge.from_legacy_cache(legacy_tuple)

    assert len(cache) == 2
    exported = cache.to_legacy_cache()
    assert len(exported) == 2
    assert exported[0][0].shape == (B, H, D, D)
    assert exported[1][0].shape == (B, H, 3, D)


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


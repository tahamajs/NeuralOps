"""Unit tests for Phase 5 NeuralOpsHybridHFCache."""

import torch
import pytest

from neuralops.surgery.hf_cache_bridge import NeuralOpsHybridHFCache


def test_hybrid_hf_cache_recurrent_and_kv():
    cache = NeuralOpsHybridHFCache()
    assert cache.get_seq_length() == 0

    # Layer 0: standard KV update
    B, H, T, D = 1, 4, 8, 16
    k0 = torch.randn(B, H, T, D)
    v0 = torch.randn(B, H, T, D)
    cache.update(k0, v0, layer_idx=0)
    assert cache.get_seq_length() == 8

    # Layer 1: constant O(1) recurrent state update
    S1 = torch.randn(B, H, D, D)
    cache.update_recurrent_state(layer_idx=1, state=S1, num_tokens=0)
    ret_s1 = cache.get_recurrent_state(layer_idx=1)
    assert ret_s1 is not None
    assert torch.allclose(ret_s1, S1)

    # Next generation step (T_step = 1)
    k0_next = torch.randn(B, H, 1, D)
    v0_next = torch.randn(B, H, 1, D)
    cache.update(k0_next, v0_next, layer_idx=0)
    assert cache.get_seq_length() == 9

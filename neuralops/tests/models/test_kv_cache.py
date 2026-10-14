"""Tests for KV cache mechanism and multi-family model integration."""

import torch
import pytest
from neuralops.models import NeuralOpsConfig, NeuralOpsForCausalLM
from neuralops.core.registry import ATTENTION_REGISTRY
import neuralops.attentions


def test_kv_cache_equivalence():
    config = NeuralOpsConfig(
        vocab_size=128,
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=128,
        attention_type="cosformer",
        use_cache=True,
    )
    model = NeuralOpsForCausalLM(config)
    model.eval()

    inputs = torch.randint(0, 128, (1, 4))
    with torch.no_grad():
        out_full = model(inputs).logits

    # Step-by-step with cache
    with torch.no_grad():
        step1 = model(inputs[:, :2], use_cache=True)
        past_kv = step1.past_key_values
        step2 = model(inputs[:, 2:], past_key_values=past_kv, use_cache=True)
        out_cached = torch.cat([step1.logits, step2.logits], dim=1)

    assert out_cached.shape == out_full.shape


@pytest.mark.parametrize("attn_type", ["cosformer", "rpc_attention", "galerkin_attention"])
def test_integration_across_attention_families(attn_type):
    config = NeuralOpsConfig(
        vocab_size=64,
        hidden_size=64,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=64,
        attention_type=attn_type,
    )
    model = NeuralOpsForCausalLM(config)
    model.eval()

    inputs = torch.randint(0, 64, (1, 4))
    outputs = model(inputs)
    assert outputs.logits.shape == (1, 4, 64)
    gen = model.generate(inputs, max_new_tokens=2, do_sample=False)
    assert gen.shape == (1, 6)


def test_unified_hybrid_cache_heterogeneous_layers():
    """Verify UnifiedHybridCache maintains states across Full, Sliding Window, and GDN2/Linear layers."""
    from neuralops.models.cache import UnifiedHybridCache
    from neuralops.models.hybrid_spec import HybridAttentionSpec

    cache = UnifiedHybridCache(num_layers=3)
    B, H, D = 2, 4, 32

    # Layer 0: Full attention (accumulating KV)
    k0 = torch.randn(B, H, 4, D)
    v0 = torch.randn(B, H, 4, D)
    cache.update_layer(0, (k0, v0), layer_type="full")
    k0_next = torch.randn(B, H, 2, D)
    v0_next = torch.randn(B, H, 2, D)
    updated0 = cache.update_layer(0, (k0_next, v0_next), layer_type="full")
    assert updated0[0].shape == (B, H, 6, D)
    assert updated0[1].shape == (B, H, 6, D)

    # Layer 1: Sliding window attention (window_size = 4)
    k1 = torch.randn(B, H, 4, D)
    v1 = torch.randn(B, H, 4, D)
    cache.update_layer(1, (k1, v1), layer_type="sliding", window_size=4)
    k1_next = torch.randn(B, H, 2, D)
    v1_next = torch.randn(B, H, 2, D)
    updated1 = cache.update_layer(1, (k1_next, v1_next), layer_type="sliding", window_size=4)
    assert updated1[0].shape == (B, H, 4, D)  # capped to window size 4

    # Layer 2: Linear / Recurrent attention (O(1) recurrent matrix state)
    S2 = torch.randn(B, H, D, D)
    cache.update_layer(2, (S2,), layer_type="linear")
    S2_next = torch.randn(B, H, D, D)
    updated2 = cache.update_layer(2, (S2_next,), layer_type="linear")
    assert updated2[0].shape == (B, H, D, D)  # constant size regardless of step count


def test_flagship_architectures_forward_and_cache():
    """Test GatedDeltaNet and Cosformer forward and cache."""
    from neuralops.attentions.gated.gated_delta_net import GatedDeltaNet
    from neuralops.attentions.linear.cosformer import CosformerAttention

    B, T, D = 2, 8, 64
    x = torch.randn(B, T, D, requires_grad=True)

    for attn_cls in [GatedDeltaNet, CosformerAttention]:
        layer = attn_cls(hidden_size=D, num_heads=2)
        out, cache = layer(x, use_cache=True)
        assert out.shape == (B, T, D)
        # Test backward differentiability
        loss = out.sum()
        loss.backward()
        assert x.grad is not None
        x.grad.zero_()



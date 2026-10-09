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

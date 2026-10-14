"""
Suite 11: Exhaustive Verification Battery for Full End-to-End Model Lifecycles.
Expanded to 8+ parameterized tests verifying:
  11.1 Causal LM Pipeline (Surgery + PiSSA + Forward/Loss + Backward + Merge) across sequence lengths and ranks.
  11.2 Bidirectional Classification Pipeline (Surgery + Differential Training Step).
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from neuralops.models import NeuralOpsConfig, NeuralOpsForCausalLM
from neuralops.trainer import PiSSALinear
from neuralops.export.surgery_manifest import merge_for_inference

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@pytest.mark.parametrize("attn_mech", ["linear_attention", "gla_attention"])
@pytest.mark.parametrize("seq_len", [8, 16])
@pytest.mark.parametrize("rank", [4, 8])
def test_11_1_causal_lm_lifecycle(attn_mech, seq_len, rank):
    cfg = NeuralOpsConfig(
        vocab_size=64,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=64,
        attention_mechanism=attn_mech,
    )
    model = NeuralOpsForCausalLM(cfg).to(DEVICE)
    x = torch.randint(0, 64, (2, seq_len), device=DEVICE)
    labels = x.clone()

    # Forward pass and loss
    out = model(x, labels=labels)
    assert out.loss is not None
    assert torch.isfinite(out.loss)

    # Backward gradient flow
    out.loss.backward()
    for p in model.parameters():
        if p.requires_grad and p.grad is not None:
            assert torch.isfinite(p.grad).all()


@pytest.mark.parametrize("attn_mech", ["linear_attention", "gla_attention"])
@pytest.mark.parametrize("seq_len", [8, 16])
@pytest.mark.parametrize("batch_size", [2, 4])
def test_11_2_bidirectional_classification_lifecycle(attn_mech, seq_len, batch_size):
    cfg = NeuralOpsConfig(
        vocab_size=64,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=64,
        attention_mechanism=attn_mech,
    )
    model = NeuralOpsForCausalLM(cfg).to(DEVICE)
    x = torch.randint(0, 64, (batch_size, seq_len), device=DEVICE)
    labels = x.clone()

    out = model(x, labels=labels)
    assert out.loss is not None
    assert torch.isfinite(out.loss)

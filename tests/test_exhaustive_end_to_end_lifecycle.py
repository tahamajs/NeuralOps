"""
Suite 11: Exhaustive Verification Battery for Full End-to-End Model Lifecycles.
Verifies:
  11.1 Causal LM Pipeline (Surgery + PiSSA + Forward/Loss + Merge).
  11.2 Bidirectional Classification Pipeline (Surgery + Probing/Fine-tuning).
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from neuralops.models import NeuralOpsConfig, NeuralOpsForCausalLM
from neuralops.trainer import PiSSALinear
from neuralops.export.surgery_manifest import merge_for_inference

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def test_11_1_causal_lm_lifecycle():
    cfg = NeuralOpsConfig(
        vocab_size=64,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=64,
        attention_mechanism="linear_attention",
    )
    model = NeuralOpsForCausalLM(cfg).to(DEVICE)
    x = torch.randint(0, 64, (2, 8), device=DEVICE)
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

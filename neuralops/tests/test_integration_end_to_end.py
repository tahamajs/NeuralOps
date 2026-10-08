"""End-to-end integration tests for neuralops modules and pipelines."""
from __future__ import annotations

import types
import pytest
import torch
import torch.nn as nn
from neuralops.core.registry import ATTENTION_REGISTRY, COMPRESSOR_REGISTRY
import neuralops.attentions
import neuralops.compression_engine
from neuralops.utils.replace_attention import replace_attention


class DummyTransformerModel(nn.Module):
    def __init__(self, hidden_dim: int = 64):
        super().__init__()
        self.embedding = nn.Embedding(100, hidden_dim)
        self.linear_in = nn.Linear(hidden_dim, hidden_dim)
        self.linear_out = nn.Linear(hidden_dim, 2)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.embedding(input_ids)
        x = self.linear_in(x)
        return self.linear_out(x[:, 0, :])


@pytest.mark.integration
def test_attention_registry_instantiation_e2e():
    """Verify that multiple representative attention modules can be instantiated and forward passed."""
    keys = ["svd_linear_attention", "cosformer"]
    B, T, D = 2, 16, 64
    x = torch.randn(B, T, D)

    for k in keys:
        cls = ATTENTION_REGISTRY.get(k)
        assert cls is not None
        layer = cls(hidden_size=D, num_heads=4)
        out = layer(x)
        assert out is not None
        if isinstance(out, tuple):
            out = out[0]
        assert out.shape[:2] == (B, T)

    # Test full_attention with config object
    cfg = types.SimpleNamespace(hidden_size=D, num_attention_heads=4, num_heads=4, attention_dropout=0.0)
    full_cls = ATTENTION_REGISTRY.get("full_attention")
    full_layer = full_cls(cfg)
    out_full = full_layer(x)
    assert out_full is not None


@pytest.mark.integration
def test_compressor_registry_instantiation_e2e():
    """Verify that compressor classes can be retrieved from registry without error."""
    keys = ["direct_svd", "slicegpt", "optimal_subspace_projection"]
    model = DummyTransformerModel()

    for k in keys:
        cls = COMPRESSOR_REGISTRY.get(k)
        assert cls is not None


@pytest.mark.integration
def test_replace_attention_distilbert_e2e():
    """Verify replace_attention surgery on DistilBERT."""
    from transformers import AutoModelForSequenceClassification

    model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased")
    model = replace_attention(model, "svd_linear_attention", svd_rank=64)
    x = torch.randint(0, 100, (1, 32))
    out = model(x)
    assert out.logits.shape == (1, 2)

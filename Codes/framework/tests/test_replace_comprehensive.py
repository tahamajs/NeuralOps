from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
from framework.attentions.replace import (_replace_bert_attention,
                                          _replace_distilbert_attention,
                                          _replace_gpt_attention,
                                          build_attention,
                                          replace_attention_in_model)
from framework.core.registry import ATTENTION_REGISTRY


class DummyAttention(nn.Module):
    def __init__(self, hidden_size=64, num_heads=4, **kwargs):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.proj = nn.Linear(hidden_size, hidden_size)

    def forward(self, x, *args, **kwargs):
        return self.proj(x), None


# Register dummy for testing
ATTENTION_REGISTRY.register("dummy_test_replace_attn")(DummyAttention)


class DummyBertModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(model_type="bert", hidden_size=64, num_attention_heads=4)
        layer = nn.Module()
        layer.attention = nn.Module()
        layer.attention.self = nn.Module()
        layer.attention.self.query = nn.Linear(64, 64)
        layer.attention.self.num_attention_heads = 4
        self.encoder = nn.Module()
        self.encoder.layer = nn.ModuleList([layer])


class DummyGPT2Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(model_type="gpt2", hidden_size=64, num_attention_heads=4)
        block = nn.Module()
        block.attn = nn.Module()
        block.attn.embed_dim = 64
        block.attn.num_heads = 4
        self.transformer = nn.Module()
        self.transformer.h = nn.ModuleList([block])


class DummyDistilBertModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(model_type="distilbert", hidden_size=64, num_attention_heads=4)
        layer = nn.Module()
        layer.attention = nn.Module()
        self.distilbert = nn.Module()
        self.distilbert.transformer = nn.Module()
        self.distilbert.transformer.layer = nn.ModuleList([layer])


def test_build_attention():
    attn = build_attention("dummy_test_replace_attn", 64, 4)
    assert isinstance(attn, DummyAttention)
    assert attn.hidden_size == 64


def test_replace_bert_attention():
    model = DummyBertModel()
    replace_attention_in_model(model, "dummy_test_replace_attn")
    assert isinstance(model.encoder.layer[0].attention.self, DummyAttention)


def test_replace_gpt_attention():
    model = DummyGPT2Model()
    replace_attention_in_model(model, "dummy_test_replace_attn")
    assert isinstance(model.transformer.h[0].attn, DummyAttention)


def test_replace_unsupported_model_type():
    model = nn.Module()
    model.config = SimpleNamespace(model_type="unknown_xyz")
    with pytest.raises(ValueError, match="Unsupported model type"):
        replace_attention_in_model(model, "dummy_test_replace_attn")

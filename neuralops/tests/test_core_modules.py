"""Unit tests for neuralops.core modules, base classes, registries, and adapters."""
from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from neuralops.core.base import BaseAttention, BaseCompressor
from neuralops.core.registry import ATTENTION_REGISTRY, COMPRESSOR_REGISTRY
from neuralops.core.adapters import get_adapter, BertAdapter, DistilBertAdapter, LlamaAdapter
from neuralops.core.gqa_utils import repeat_kv
from neuralops.core.attention_utils import create_attention_mask
from neuralops.core.model_utils import count_parameters


class DummyAttention(BaseAttention):
    def forward(self, hidden_states, **kwargs):
        return hidden_states, None


class DummyCompressor(BaseCompressor):
    def compress(self, model=None, **kwargs):
        return model


def test_base_attention_instantiation():
    """Verify BaseAttention subclass instantiation and forward pass."""
    attn = DummyAttention(hidden_size=64, num_heads=4)
    x = torch.randn(2, 8, 64)
    out, weights = attn(x)
    assert out.shape == x.shape
    assert weights is None


def test_base_compressor_instantiation():
    """Verify BaseCompressor subclass instantiation and execution."""
    comp = DummyCompressor()
    model = nn.Linear(10, 10)
    compressed = comp.compress(model)
    assert compressed is model


def test_registry_get_set():
    """Verify attention and compressor registry lookup."""
    assert len(ATTENTION_REGISTRY) == 159
    assert len(COMPRESSOR_REGISTRY) == 73
    assert "svd_linear_attention" in ATTENTION_REGISTRY
    assert "direct_svd" in COMPRESSOR_REGISTRY


def test_registry_aliases():
    """Verify registry alias resolution."""
    mod = ATTENTION_REGISTRY.get("svd_linear_attention")
    assert mod is not None


def test_adapters_for_each_architecture():
    """Verify architecture adapter detection and lookup."""
    bert_adp = get_adapter("bert")
    assert bert_adp is not None

    distil_adp = get_adapter("distilbert")
    assert distil_adp is not None

    llama_adp = get_adapter("llama")
    assert llama_adp is not None


def test_gqa_utils_repeat_kv():
    """Verify repeat_kv expands key/value heads correctly."""
    # (B, num_kv_heads, S, D)
    x = torch.randn(2, 2, 8, 16)
    out = repeat_kv(x, n_rep=4)
    assert out.shape == (2, 8, 8, 16)


def test_attention_mask_creation():
    """Verify create_attention_mask handles causal and padding masking."""
    mask = create_attention_mask(seq_len=8, is_causal=True)
    assert mask is not None
    assert mask.shape[-1] == 8
    assert mask.shape[-2] == 8


def test_model_utils_count_parameters():
    """Verify count_parameters utility."""
    model = nn.Linear(10, 20, bias=True)
    params = count_parameters(model)
    assert params == 10 * 20 + 20

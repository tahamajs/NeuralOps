"""Smoke tests for 10 low-coverage or core modules across attention and compression families.

Tests import, basic forward pass, and edge-cases (shapes, masks, configuration).
"""
import pytest
import torch
import torch.nn as nn
from neuralops.core.registry import ATTENTION_REGISTRY, COMPRESSOR_REGISTRY
import neuralops.attentions
import neuralops.compression_engine

SMOKE_ATTENTIONS = [
    "cosformer",
    "decay_linear_attention",
    "gated_delta_net",
    "galerkin_attention",
    "rpc_attention",
]

SMOKE_COMPRESSORS = [
    "direct_svd",
    "admm_compressor",
    "cur_compressor",
    "adasvd_compressor",
    "aa_svd_compressor",
]

class DummyConfig:
    def __init__(self, hidden_size=64, num_heads=4):
        self.hidden_size = hidden_size
        self.d_model = hidden_size
        self.dim = hidden_size
        self.num_attention_heads = num_heads
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.max_position_embeddings = 512
        self.vocab_size = 1000

@pytest.mark.parametrize("attn_name", SMOKE_ATTENTIONS)
def test_attention_smoke_import_and_forward(attn_name):
    assert attn_name in ATTENTION_REGISTRY
    cls = ATTENTION_REGISTRY[attn_name]
    cfg = DummyConfig()
    try:
        mod = cls(cfg)
    except Exception:
        mod = cls()

    x = torch.randn(2, 8, 64)
    out = mod(x)
    if isinstance(out, tuple):
        out = out[0]
    assert out is not None
    assert out.shape == x.shape

@pytest.mark.parametrize("attn_name", SMOKE_ATTENTIONS)
def test_attention_smoke_edge_cases(attn_name):
    cls = ATTENTION_REGISTRY[attn_name]
    cfg = DummyConfig()
    try:
        mod = cls(cfg)
    except Exception:
        mod = cls()

    # Edge case 1: Single token (batch=1, seq_len=1)
    x1 = torch.randn(1, 1, 64)
    out1 = mod(x1)
    if isinstance(out1, tuple):
        out1 = out1[0]
    assert out1.shape == (1, 1, 64)

    # Edge case 2: Variable length sequence
    x2 = torch.randn(1, 16, 64)
    out2 = mod(x2)
    if isinstance(out2, tuple):
        out2 = out2[0]
    assert out2.shape == (1, 16, 64)

@pytest.mark.parametrize("comp_name", SMOKE_COMPRESSORS)
def test_compressor_smoke_import(comp_name):
    assert comp_name in COMPRESSOR_REGISTRY
    cls = COMPRESSOR_REGISTRY[comp_name]
    assert cls is not None

@pytest.mark.parametrize("comp_name", SMOKE_COMPRESSORS)
def test_compressor_smoke_basic_instantiation(comp_name):
    cls = COMPRESSOR_REGISTRY[comp_name]
    dummy_model = nn.Sequential(
        nn.Linear(32, 32),
        nn.ReLU(),
        nn.Linear(32, 10)
    )
    compressor = cls(dummy_model)
    assert compressor is not None

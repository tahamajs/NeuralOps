import math
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
from neuralops.attentions.adaptive_span_attention import AdaptiveSpanAttention
from neuralops.attentions.adaptive_sparse_attention import \
    AdaptiveSparseAttention
from neuralops.attentions.adaptive_svd_attention import AdaptiveSVDAttention
from neuralops.attentions.adasplash_attention import AdaSplashAttention


@pytest.fixture
def t5_config():
    return SimpleNamespace(
        hidden_size=512,
        d_model=512,
        num_attention_heads=8,
        dropout=0.1,
        max_span=64,
        svd_rank=64,
        relative_attention_num_buckets=32,
        relative_attention_max_distance=128,
        is_decoder=False,
    )

MODULE_CLASSES = [
    ("adaptive_span", AdaptiveSpanAttention),
    ("adaptive_sparse", AdaptiveSparseAttention),
    ("adaptive_svd", AdaptiveSVDAttention),
    ("adasplash", AdaSplashAttention),
]

@pytest.mark.parametrize("name,cls", MODULE_CLASSES)
def test_self_attention_finite_and_shape(name, cls, t5_config):
    mod = cls(t5_config).half()
    B, N, D = 2, 64, 512
    x = torch.randn(B, N, D, dtype=torch.float16)
    mask = torch.ones(B, N, dtype=torch.long)
    mask[1, 10:] = 0

    out, extra = mod(x, attention_mask=mask)
    assert out.shape == (B, N, D), f"{name} self-attn shape mismatch"
    assert torch.isfinite(out).all(), f"{name} self-attn contains NaN/Inf"

@pytest.mark.parametrize("name,cls", MODULE_CLASSES)
def test_cross_attention_shapes_and_values(name, cls, t5_config):
    mod = cls(t5_config).half()
    B, N_q, N_k, D = 2, 16, 64, 512
    q = torch.randn(B, N_q, D, dtype=torch.float16)
    encoder_hidden_states = torch.randn(B, N_k, D, dtype=torch.float16)
    mask = torch.ones(B, N_k, dtype=torch.long)

    out, extra = mod(q, key_value_states=encoder_hidden_states, attention_mask=mask)
    assert out.shape == (B, N_q, D), f"{name} cross-attn shape mismatch"
    assert torch.isfinite(out).all(), f"{name} cross-attn contains NaN/Inf"

@pytest.mark.parametrize("name,cls", MODULE_CLASSES)
def test_relative_position_bias_application(name, cls, t5_config):
    mod = cls(t5_config).half()
    # Add a relative attention bias embedding
    mod.relative_attention_bias = nn.Embedding(32, 8).half()
    mod.has_relative_attention_bias = True

    B, N, D = 2, 32, 512
    x = torch.randn(B, N, D, dtype=torch.float16)
    out, pos_bias = mod(x)
    assert pos_bias is not None, f"{name} did not return position_bias"
    assert torch.isfinite(out).all(), f"{name} output contains NaN/Inf with position bias"
    assert torch.isfinite(pos_bias).all(), f"{name} position_bias contains NaN/Inf"

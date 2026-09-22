"""
Hardening tests for AdaSplashAttention (adasplash_attention.py).

Covers: fail-loud copy_weights, odd-dim RoPE, inject safety, forward shapes.
"""

import pytest
import torch
import torch.nn as nn
from typing import Any


# ---------------------------------------------------------------------------
# Dummy model stubs
# ---------------------------------------------------------------------------

class MockConfig:
    hidden_size = 64
    num_attention_heads = 4
    num_heads = 4
    num_key_value_heads = 4
    head_dim = 16
    is_decoder = False
    model_type = "bert"
    rank = 8
    alpha = 1.5
    lora_rank = 4


class DummyDistilBertAttn(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_lin = nn.Linear(64, 64)
        self.k_lin = nn.Linear(64, 64)
        self.v_lin = nn.Linear(64, 64)
        self.out_lin = nn.Linear(64, 64)


class DummyRobertaAttn(nn.Module):
    """Simulates RoBERTa BertAttention with nested .self / .output.dense."""
    def __init__(self):
        super().__init__()
        self_obj = type("BertSelfAttn", (nn.Module,), {})()
        self_obj.query = nn.Linear(64, 64)
        self_obj.key = nn.Linear(64, 64)
        self_obj.value = nn.Linear(64, 64)
        self.self = self_obj
        out_obj = type("BertOutput", (nn.Module,), {})()
        out_obj.dense = nn.Linear(64, 64)
        self.output = out_obj


class DummyGPT2Attn(nn.Module):
    def __init__(self):
        super().__init__()
        self.c_attn = nn.Linear(64, 192)   # fused q+k+v
        self.c_proj = nn.Linear(64, 64)


class DummyQwenAttn(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(64, 64)
        self.k_proj = nn.Linear(64, 64)
        self.v_proj = nn.Linear(64, 64)
        self.o_proj = nn.Linear(64, 64)


class DummyEmptyAttn(nn.Module):
    """Attention with no extractable projections — should trigger RuntimeError."""
    def forward(self, x):
        return x


class DummyLayer(nn.Module):
    def __init__(self, attn):
        super().__init__()
        self.attention = attn


# ---------------------------------------------------------------------------
# Import
# ---------------------------------------------------------------------------

from framework.attentions.adasplash_attention import AdaSplashAttention, inject_adasplash


# ---------------------------------------------------------------------------
# Tests: copy_weights fail-loud
# ---------------------------------------------------------------------------

class TestCopyWeightsFailLoud:
    def _make_attn(self):
        return AdaSplashAttention(config=MockConfig())

    def test_copy_distilbert_succeeds(self):
        attn = self._make_attn()
        src = DummyDistilBertAttn()
        attn.copy_weights(src)  # must not raise

    def test_copy_roberta_succeeds(self):
        attn = self._make_attn()
        src = DummyRobertaAttn()
        attn.copy_weights(src)  # must not raise

    def test_copy_gpt2_succeeds(self):
        attn = self._make_attn()
        src = DummyGPT2Attn()
        attn.copy_weights(src)  # must not raise

    def test_copy_qwen_succeeds(self):
        attn = self._make_attn()
        src = DummyQwenAttn()
        attn.copy_weights(src)  # must not raise

    def test_copy_empty_raises_runtimeerror(self):
        attn = self._make_attn()
        with pytest.raises(RuntimeError, match="mandatory projections|Refusing"):
            attn.copy_weights(DummyEmptyAttn())

    def test_copy_empty_not_warns_silently(self):
        """copy_weights must NOT silently return — it must raise."""
        attn = self._make_attn()
        raised = False
        try:
            attn.copy_weights(DummyEmptyAttn())
        except RuntimeError:
            raised = True
        assert raised, "copy_weights should raise RuntimeError, not silently return"


# ---------------------------------------------------------------------------
# Tests: odd-dim RoPE guard in forward
# ---------------------------------------------------------------------------

class TestOddDimRoPE:
    def _make_attn(self):
        return AdaSplashAttention(config=MockConfig())

    def test_forward_even_dim_no_rope(self):
        attn = self._make_attn()
        x = torch.randn(2, 8, 64)
        out, _ = attn(x)
        assert out.shape == (2, 8, 64)

    def test_forward_with_rope_even(self):
        attn = self._make_attn()
        x = torch.randn(2, 8, 64)
        cos = torch.ones(2, 8, 16)
        sin = torch.zeros(2, 8, 16)
        out, _ = attn(x, position_embeddings=(cos, sin))
        assert out.shape == (2, 8, 64)

    def test_forward_with_rope_odd_head_dim(self):
        """Odd head_dim must not crash (e.g. head_dim=7)."""
        cfg = type("C", (), {
            "hidden_size": 56, "num_attention_heads": 8, "num_heads": 8,
            "num_key_value_heads": 8, "head_dim": 7, "is_decoder": False,
            "model_type": "bert", "rank": 4, "alpha": 1.5, "lora_rank": 2,
        })()
        attn = AdaSplashAttention(config=cfg)
        x = torch.randn(1, 4, 56)
        cos = torch.ones(1, 4, 7)
        sin = torch.zeros(1, 4, 7)
        out, _ = attn(x, position_embeddings=(cos, sin))
        assert out.shape == (1, 4, 56)


# ---------------------------------------------------------------------------
# Tests: inject safety
# ---------------------------------------------------------------------------

class TestInjectSafety:
    def test_inject_none_returns_none(self):
        assert inject_adasplash(layer=None, model=None, model_config=MockConfig()) is None

    def test_inject_distilbert(self):
        layer = DummyLayer(DummyDistilBertAttn())
        res = inject_adasplash(layer=layer, model_config=MockConfig())
        assert res is not None

    def test_inject_qwen(self):
        class L(nn.Module):
            def __init__(self):
                super().__init__()
                self.self_attn = DummyQwenAttn()
        res = inject_adasplash(layer=L(), model_config=MockConfig())
        assert res is not None

    def test_inject_empty_attn_returns_none(self):
        """Inject with an unrecognised attn (no projections) must return None, not raise."""
        layer = DummyLayer(DummyEmptyAttn())
        res = inject_adasplash(layer=layer, model_config=MockConfig())
        assert res is None

    def test_inject_returns_adasplash_instance(self):
        layer = DummyLayer(DummyDistilBertAttn())
        res = inject_adasplash(layer=layer, model_config=MockConfig())
        assert isinstance(res, AdaSplashAttention)


# ---------------------------------------------------------------------------
# Tests: forward shapes
# ---------------------------------------------------------------------------

class TestForwardShapes:
    def _make_attn(self):
        return AdaSplashAttention(config=MockConfig())

    def test_encoder_output_shape(self):
        attn = self._make_attn()
        x = torch.randn(2, 10, 64)
        out, weights = attn(x)
        assert out.shape == (2, 10, 64)

    def test_output_attentions(self):
        attn = self._make_attn()
        x = torch.randn(1, 6, 64)
        out, weights = attn(x, output_attentions=True)
        assert out.shape == (1, 6, 64)

    def test_with_attention_mask(self):
        attn = self._make_attn()
        x = torch.randn(2, 5, 64)
        mask = torch.zeros(2, 1, 1, 5)
        out, _ = attn(x, attention_mask=mask)
        assert out.shape == (2, 5, 64)

    def test_pretrained_attn_in_constructor(self):
        src = DummyDistilBertAttn()
        attn = AdaSplashAttention(config=MockConfig(), pretrained_attn=src)
        x = torch.randn(1, 4, 64)
        out, _ = attn(x)
        assert out.shape == (1, 4, 64)

"""
Tests for the attention injection utilities.
"""

from __future__ import annotations

import os
import sys

import pytest
import torch
from transformers import DistilBertConfig, DistilBertModel

_FRAMEWORK_PARENT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _FRAMEWORK_PARENT not in sys.path:
    sys.path.insert(0, _FRAMEWORK_PARENT)

import framework.attentions
from framework.core.attention_utils import (available_attention_types,
                                            inject_attention)


class TestAttentionInjection:
    """Verify that inject_attention works correctly on foundation transformer models."""

    @pytest.mark.integration
    def test_inject_attention_on_distilbert(self):
        """Inject known attention types into DistilBERT and verify forward pass execution."""
        config = DistilBertConfig(
            vocab_size=100,
            dim=768,
            n_layers=2,
            n_heads=8,
            hidden_dim=3072,
        )
        model = DistilBertModel(config)
        inject_attention(model, "full", copy_weights=False)

        # Verify forward pass still works
        dummy = (
            model.distilbert.transformer.layer[0].attention
            if hasattr(model, "distilbert")
            else model.transformer.layer[0].attention
        )

        x = torch.randn(2, 8, 768)
        out = dummy(x)
        if isinstance(out, tuple):
            out = out[0]
        assert out.shape == (2, 8, 768)

    @pytest.mark.integration
    def test_inject_attention_with_copy_weights(self):
        """Verify injection with copy_weights enabled."""
        config = DistilBertConfig(
            vocab_size=100,
            dim=768,
            n_layers=2,
            n_heads=8,
            hidden_dim=3072,
        )
        model = DistilBertModel(config)
        inject_attention(model, "full", copy_weights=True)

        x = torch.randint(0, 100, (2, 16))
        outputs = model(input_ids=x)
        assert outputs.last_hidden_state.shape == (2, 16, 768)

    def test_available_attention_types(self):
        """Verify registry queries return non-empty list of attention variants."""
        attn_types = available_attention_types()
        assert len(attn_types) > 0
        assert "full" in attn_types


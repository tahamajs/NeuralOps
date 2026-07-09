"""
Tests for the attention injection utilities.
"""

import pytest


class TestAttentionInjection:
    """Verify that inject_attention works correctly on a real model."""

    @pytest.mark.integration
    def test_inject_attention_on_distilbert(self):
        """Inject a known attention type into DistilBERT and verify no crash."""
        try:
            import framework.attentions
            from framework.core.attention_utils import inject_attention
            from transformers import DistilBertConfig, DistilBertModel
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
            dummy = model.distilbert.transformer.layer[0].attention if hasattr(model, "distilbert") else model.transformer.layer[0].attention
            import torch

            x = torch.randn(2, 8, 768)
            out = dummy(x)
            if isinstance(out, tuple):
                out = out[0]
            assert out.shape == (2, 8, 768)
        except ImportError:
            pytest.skip("framework.core not available")
        except Exception as e:
            pytest.fail(f"inject_attention failed: {e}")

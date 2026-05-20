"""
Tests for the attention injection utilities.
"""

import pytest
from conftest import ATTENTION_REGISTRY


class TestAttentionInjection:
    """Verify that inject_attention works correctly on a real model."""

    @pytest.mark.integration
    def test_inject_attention_on_distilbert(self):
        """Inject a known attention type into DistilBERT and verify no crash."""
        try:
            from framework.core.attention_utils import inject_attention
            from transformers import AutoModel
            model = AutoModel.from_pretrained(
                "distilbert-base-uncased", local_files_only=True)
            inject_attention(model, "full", copy_weights=False)
            # Verify forward pass still works
            dummy = model.distilbert.transformer.layer[0].attention
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
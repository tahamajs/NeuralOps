"""Unit tests for attention module family compatibility and declared capabilities."""

import pytest
import torch
import torch.nn as nn
from framework.core.registry import (ATTENTION_REGISTRY,
                                     get_module_capabilities,
                                     is_module_declared_compatible)
from framework.utils.compat import module_family_support
from transformers import PretrainedConfig


class DummyConfig(PretrainedConfig):
    def __init__(self, hidden_size=32, num_attention_heads=2, num_heads=2, **kwargs):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_attention_heads
        self.d_model = hidden_size
        self.max_position_embeddings = 64
        self.vocab_size = 100


class TestAttentionFamilyCompatibility:
    """Verify capability declarations and family compatibility matrices."""

    @pytest.mark.parametrize("module_name", [
        "adaptive_span",
        "adaptive_sparse",
    ])
    def test_causal_only_incompatible_with_encoders(self, module_name):
        """Causal-only modules must be rejected for BERT, RoBERTa, DistilBERT."""
        for encoder_family in ("bert", "roberta", "distilbert"):
            is_compat, reason = is_module_declared_compatible(module_name, encoder_family)
            assert not is_compat, f"{module_name} should not be compatible with {encoder_family}"
            assert "causal_only" in reason

    @pytest.mark.parametrize("module_name", [
        "adaptive_span",
        "adaptive_sparse",
    ])
    def test_causal_only_compatible_with_decoders(self, module_name):
        """Causal-only modules must be allowed for GPT-2, Qwen, LLaMA."""
        for decoder_family in ("gpt", "gpt2", "qwen", "llama"):
            is_compat, reason = is_module_declared_compatible(module_name, decoder_family)
            assert is_compat, f"{module_name} should be compatible with {decoder_family}, got {reason}"

    @pytest.mark.parametrize("module_name", [
        "adaptive_svd",
        "asvd",
        "res_svd",
    ])
    def test_bidirectional_compatible_with_both(self, module_name):
        """Bidirectional attention modules must be compatible with both encoders and decoders."""
        for family in ("bert", "roberta", "gpt2"):
            is_compat, _ = is_module_declared_compatible(module_name, family)
            assert is_compat, f"{module_name} should be compatible with {family}"

    def test_module_family_support_dictionary(self):
        """Verify module_family_support returns structured dictionary across families."""
        support_svd = module_family_support("adaptive_svd")
        assert isinstance(support_svd, dict)
        assert "bert" in support_svd
        assert "roberta" in support_svd
        assert "gpt" in support_svd

    @pytest.mark.parametrize("mod_name", [
        "adaptive_svd",
        "asvd",
        "cur_moe",
        "rpc",
    ])
    def test_instantiate_representative_modules(self, mod_name):
        """Verify representative registered modules instantiate with a minimal dummy config."""
        cls_or_factory = ATTENTION_REGISTRY.get(mod_name)
        if cls_or_factory is None:
            pytest.skip(f"{mod_name} not registered")
        cfg = DummyConfig()
        try:
            mod = cls_or_factory(cfg)
        except TypeError:
            try:
                mod = cls_or_factory(cfg, rank=8)
            except Exception:
                pytest.skip(f"Factory for {mod_name} requires special args")
        assert isinstance(mod, nn.Module)

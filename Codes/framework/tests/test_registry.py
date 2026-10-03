"""Unit tests for central registry, module registration, and helper module filtering."""

import pytest
import torch.nn as nn
from framework.core.registry import (ATTENTION_REGISTRY, COMPRESSOR_REGISTRY,
                                     Registry, get_evaluatable_modules,
                                     is_helper_module)
from transformers import PretrainedConfig


class DummyConfig(PretrainedConfig):
    def __init__(self, hidden_size=32, num_attention_heads=2, **kwargs):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.num_heads = num_attention_heads
        self.head_dim = hidden_size // num_attention_heads
        self.d_model = hidden_size


class TestRegistryCore:
    """Verify Registry mechanics, duplicate registration handling, and helper exclusion."""

    def test_registry_registration_and_retrieval(self):
        reg = Registry("test_reg")

        @reg.register("my_layer", supports_causal=True, supports_bidirectional=False)
        class MyLayer(nn.Module):
            pass

        assert "my_layer" in reg
        assert reg.get("my_layer") is MyLayer
        metadata = reg.get_metadata("my_layer")
        assert metadata["supports_causal"] is True
        assert metadata["supports_bidirectional"] is False

    def test_duplicate_registration_aliases(self):
        """Registering same class under multiple alias keys must work cleanly without error."""
        reg = Registry("alias_reg")

        @reg.register("alias_a")
        @reg.register("alias_b")
        class SharedClass(nn.Module):
            pass

        assert reg.get("alias_a") is SharedClass
        assert reg.get("alias_b") is SharedClass

    def test_is_helper_module_filtering(self):
        """Helper suffix patterns must be filtered out of evaluatable components."""
        assert is_helper_module("adaptive_svd_factory")
        assert is_helper_module("adaptive_svd_inject")
        assert is_helper_module("my_compressor_fn")
        assert not is_helper_module("adaptive_svd")
        assert not is_helper_module("asvd")

    def test_get_evaluatable_modules_non_empty(self):
        modules = get_evaluatable_modules(include_compressors=True)
        assert len(modules) > 20
        # No helper modules present
        for m in modules:
            assert not is_helper_module(m)

    def test_core_registries_populated(self):
        assert len(ATTENTION_REGISTRY.keys()) > 10
        assert len(COMPRESSOR_REGISTRY.keys()) > 5
        assert "adaptive_svd" in ATTENTION_REGISTRY
        assert "aasvd" in COMPRESSOR_REGISTRY

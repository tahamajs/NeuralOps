import os
import sys
import unittest

import torch
import torch.nn as nn

_FRAMEWORK_PARENT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _FRAMEWORK_PARENT not in sys.path:
    sys.path.insert(0, _FRAMEWORK_PARENT)

import neuralops.attentions
from neuralops.core.registry import ATTENTION_REGISTRY


class DummyConfig:
    def __init__(self, hidden_size=64, num_heads=4, rank=16):
        self.hidden_size = hidden_size
        self.num_attention_heads = num_heads
        self.num_heads = num_heads
        self.head_dim = hidden_size // max(1, num_heads)
        self.rank = rank
        self.dropout = 0.0
        self.attention_dropout = 0.0
        self.window_size = 32
        self.svd_rank = 16


class TestAttentionMechanisms(unittest.TestCase):
    """Test suite verifying attention module registry, forward propagation, and output shapes."""

    def setUp(self):
        self.cfg = DummyConfig(hidden_size=64, num_heads=4, rank=16)
        self.batch_size = 2
        self.seq_len = 16
        self.hidden_size = 64
        self.dummy_x = torch.randn(self.batch_size, self.seq_len, self.hidden_size)

    def _instantiate_module(self, cls):
        try:
            return cls(config=self.cfg)
        except TypeError:
            try:
                return cls(self.cfg)
            except TypeError:
                return cls(hidden_size=self.hidden_size, num_heads=self.cfg.num_heads)

    def test_registry_size(self):
        self.assertGreaterEqual(len(ATTENTION_REGISTRY), 50)

    def test_full_and_gqa_attention(self):
        for name in ["full", "gqa", "mla"]:
            if name in ATTENTION_REGISTRY:
                cls = ATTENTION_REGISTRY.get(name)
                mod = self._instantiate_module(cls)
                out = mod(self.dummy_x)
                if isinstance(out, tuple):
                    out = out[0]
                self.assertEqual(out.shape, (self.batch_size, self.seq_len, self.hidden_size))

    def test_svd_linear_attention(self):
        for name in ["svd_linear", "svd_linear_attn"]:
            if name in ATTENTION_REGISTRY:
                cls = ATTENTION_REGISTRY.get(name)
                mod = self._instantiate_module(cls)
                out = mod(self.dummy_x)
                if isinstance(out, tuple):
                    out = out[0]
                self.assertEqual(out.shape, (self.batch_size, self.seq_len, self.hidden_size))

    def test_next_gen_hybrids(self):
        hybrids = ["mamba_svd_flash", "moe_deltakv_paged", "aegis_spectron_fp8", "aegis_hybrid"]
        for name in hybrids:
            if name in ATTENTION_REGISTRY:
                cls = ATTENTION_REGISTRY.get(name)
                mod = self._instantiate_module(cls)
                out = mod(self.dummy_x)
                if isinstance(out, tuple):
                    out = out[0]
                self.assertEqual(out.shape, (self.batch_size, self.seq_len, self.hidden_size))


if __name__ == "__main__":
    unittest.main()

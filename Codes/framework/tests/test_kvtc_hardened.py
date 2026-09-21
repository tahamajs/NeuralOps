"""
Comprehensive unit test suite for hardened KVTC (KV Cache Transform Coding / PCA-Quant) compressor.

Tests:
  1. test_weight_quantization_not_kv_cache:
     Verifies KVTCCompressor operates on model weight parameters/buffers, not runtime KV cache tensors.
  2. test_integer_quantization:
     Verifies int8 weight buffer storage, scale/zero_point buffers, and element_size() == 1.
  3. test_forward_check_success:
     Verifies forward sanity check runs and produces forward_check_success=True on a valid model.
  4. test_gqa_asymmetric_skipped:
     Verifies GQA models have asymmetric QKV projections skipped when skip_attention_qkv=True.
  5. test_percentile_clip:
     Verifies _percentile_clip clips extreme outliers.
  6. test_real_bytes_measurement:
     Verifies _estimate_actual_bytes returns realistic byte counts reflecting int8 storage.
  7. test_registry_aliases:
     Verifies both original aliases ('kvtc', 'kvtc_quantizer') and honest aliases ('pca_quant') exist.
  8. test_end_to_end_compression:
     Runs full compress() on a model and verifies output shapes, forward pass, and stats integrity.
"""

from __future__ import annotations

import os
import sys
import unittest
from typing import Any, Dict

import torch
import torch.nn as nn

# Ensure project root is in path
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from framework.compression_engine.kvtc_compressor import (
    KVTCCompressor,
    KVTCQuantizedLoRALinear,
    KVTCStore,
    SimpleEntropyCoder,
    kvtc_factory,
)
from framework.core.registry import COMPRESSOR_REGISTRY


class ToyTransformerBlock(nn.Module):
    """Multi-layer toy transformer block for testing compression engines."""

    def __init__(self, d_model: int = 64, d_ff: int = 128):
        super().__init__()
        self.d_model = d_model
        self.q_lin = nn.Linear(d_model, d_model)
        self.k_lin = nn.Linear(d_model, d_model)
        self.v_lin = nn.Linear(d_model, d_model)
        self.out_lin = nn.Linear(d_model, d_model)
        self.fc1 = nn.Linear(d_model, d_ff)
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(d_ff, d_model)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q = self.q_lin(x)
        k = self.k_lin(x)
        v = self.v_lin(x)
        attn = torch.matmul(q, k.transpose(-2, -1)) / (self.d_model ** 0.5)
        attn_weights = self.softmax(attn)
        attn_out = self.out_lin(torch.matmul(attn_weights, v))
        x = x + attn_out
        mlp_out = self.fc2(self.act(self.fc1(x)))
        return x + mlp_out


class MockGQAModel(nn.Module):
    """Mock model with Grouped-Query Attention (num_attention_heads > num_key_value_heads)."""

    def __init__(self, hidden_size: int = 64):
        super().__init__()
        self.config = type("Config", (), {
            "num_attention_heads": 8,
            "num_key_value_heads": 2,
            "hidden_size": hidden_size,
        })()
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size // 4)
        self.v_proj = nn.Linear(hidden_size, hidden_size // 4)
        self.o_proj = nn.Linear(hidden_size, hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.o_proj(x)


class TestKVTCHardened(unittest.TestCase):
    """Comprehensive test suite for hardened KVTC compressor."""

    def setUp(self):
        torch.manual_seed(42)
        self.d_model = 64
        self.batch_size = 2
        self.seq_len = 8
        self.dummy_x = torch.randn(self.batch_size, self.seq_len, self.d_model)

    def test_weight_quantization_not_kv_cache(self):
        """Verify KVTCCompressor operates on model weight layers, not runtime KV cache tensors."""
        model = ToyTransformerBlock(d_model=self.d_model)
        compressor = KVTCCompressor(model, bits=8, pca_energy=0.95, max_error=0.50)
        compressed_model, stats = compressor.compress()

        # The compressed layers are named linear modules within the model
        self.assertGreater(stats["num_compressed"], 0)
        self.assertIn("q_lin", stats["compressed_layers"])
        # The module type has been replaced with KVTCQuantizedLoRALinear
        self.assertIsInstance(compressed_model.q_lin, KVTCQuantizedLoRALinear)
        # Verify stats method confirms weight PCA quantization
        self.assertEqual(stats["method"], "kvtc_pca_quant")

    def test_integer_quantization(self):
        """Verify int8 weight buffer storage, scale/zero_point buffers, and element_size() == 1."""
        in_dim, out_dim = 64, 64
        raw_w = torch.randn(out_dim, in_dim)
        comp = KVTCCompressor(nn.Linear(in_dim, out_dim), bits=8)
        int_w, scale, zp = comp._quantize_to_int(raw_w, bits=8)

        # Must be integer tensor
        self.assertEqual(int_w.dtype, torch.int8)
        self.assertEqual(int_w.element_size(), 1)

        # Test in KVTCQuantizedLoRALinear
        layer = KVTCQuantizedLoRALinear(in_dim, out_dim, int_w, scale, zp)
        self.assertIn("int_weight", dict(layer.named_buffers()))
        self.assertIn("scale", dict(layer.named_buffers()))
        self.assertIn("zero_point", dict(layer.named_buffers()))

        # Forward output verification
        x = torch.randn(2, in_dim)
        out = layer(x)
        self.assertEqual(out.shape, (2, out_dim))

    def test_forward_check_success(self):
        """Verify forward sanity check runs and sets forward_check_success=True."""
        model = ToyTransformerBlock(d_model=self.d_model)
        compressor = KVTCCompressor(model, bits=8, pca_energy=0.95, max_error=0.50)
        _, stats = compressor.compress()
        self.assertTrue(stats["forward_check_success"])
        self.assertIsNone(stats.get("forward_check_error"))

    def test_gqa_asymmetric_skipped(self):
        """Verify GQA models have asymmetric QKV projections skipped when skip_attention_qkv=True."""
        model = MockGQAModel(hidden_size=self.d_model)
        compressor = KVTCCompressor(model, bits=8, skip_attention_qkv=True)

        # q_proj, k_proj, v_proj should be identified as GQA asymmetric
        self.assertTrue(compressor._is_gqa_asymmetric("q_proj", model.q_proj))
        self.assertTrue(compressor._is_gqa_asymmetric("k_proj", model.k_proj))
        self.assertTrue(compressor._is_gqa_asymmetric("v_proj", model.v_proj))

        # Target module check should exclude them
        self.assertFalse(compressor._is_target_module("q_proj", model.q_proj))
        self.assertFalse(compressor._is_target_module("k_proj", model.k_proj))
        self.assertFalse(compressor._is_target_module("v_proj", model.v_proj))

        # But o_proj is symmetric and can be compressed
        self.assertTrue(compressor._is_target_module("o_proj", model.o_proj))

    def test_percentile_clip(self):
        """Verify _percentile_clip clips extreme outliers."""
        w = torch.tensor([-100.0, 0.1, 0.2, 0.3, -0.2, 100.0])
        w_clipped = KVTCCompressor._percentile_clip(w, low=0.1, high=0.9)
        self.assertGreater(w_clipped.min().item(), -100.0)
        self.assertLess(w_clipped.max().item(), 100.0)

    def test_real_bytes_measurement(self):
        """Verify _estimate_actual_bytes returns realistic byte counts reflecting int8 storage."""
        model = ToyTransformerBlock(d_model=self.d_model)
        comp = KVTCCompressor(model, bits=8, pca_energy=0.95, max_error=0.50)
        initial_bytes = comp._estimate_actual_bytes(model)

        comp_model, stats = comp.compress()
        final_bytes = stats["final_actual_bytes"]

        self.assertGreater(initial_bytes, 0)
        self.assertGreater(final_bytes, 0)
        # Compressed model with int8 buffers should have substantially smaller physical footprint
        self.assertLess(final_bytes, initial_bytes)
        self.assertGreater(stats["actual_bytes_reduction_pct"], 30.0)

    def test_registry_aliases(self):
        """Verify both original aliases and honest PCA-quant aliases exist in registry."""
        for alias in ["kvtc", "kvtc_quantizer", "kvtc_factory", "pca_quant", "pca_quant_compressor", "pca_quant_factory"]:
            self.assertIn(alias, COMPRESSOR_REGISTRY, f"Missing alias: {alias}")

    def test_end_to_end_compression(self):
        """Runs full compress() and factory on a model, verifying output shapes and stats."""
        model = ToyTransformerBlock(d_model=self.d_model)
        comp_model, stats = kvtc_factory(model, bits=8, pca_energy=0.95, max_error=0.50, return_stats=True)

        self.assertIsInstance(comp_model, nn.Module)
        self.assertIn("compressed_layers", stats)
        self.assertIn("avg_entropy_bits", stats)
        self.assertIn("entropy_lower_bound_bits", stats)
        self.assertIn("entropy_coding_note", stats)
        self.assertTrue(stats["forward_check_success"])

        # Forward pass on compressed model
        out = comp_model(self.dummy_x)
        self.assertEqual(out.shape, (self.batch_size, self.seq_len, self.d_model))
        self.assertFalse(torch.isnan(out).any())


if __name__ == "__main__":
    unittest.main()

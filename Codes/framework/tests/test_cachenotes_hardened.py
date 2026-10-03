"""
Unit tests for hardened CacheNotes compressor.
Verifies:
1. LoRA wrapper and Conv1D detection and transposition.
2. If layer is already LoRALinear, base_layer is compressed without creating redundant wrappers, and LoRA adapters remain trainable.
3. CacheNotesLoRALinear forward matches x @ weight.T + bias + lora_out.
4. target_attention_only=True skips FFN layers and preserves classifier head.
5. High reconstruction error triggers fallback and freezes fallback parameters (requires_grad=False).
6. CacheNotesKVCache prunes sequence length while strictly preserving sink tokens.
7. Factory works with comp_config=None and returns compression_stats.
"""
from __future__ import annotations

import unittest

import torch
import torch.nn as nn
import torch.nn.functional as F
from framework.compression_engine.cachenotes_compressor import (
    CacheNotesCompressor, CacheNotesKVCache, CacheNotesLoRALinear,
    CompressionPlanningTokens, cachenotes_factory)
from transformers.pytorch_utils import Conv1D


class DummyLoRALinear(nn.Module):
    """Mock pre-existing LoRA linear layer wrapping base_layer."""

    def __init__(self, base_layer: nn.Module, rank: int = 4):
        super().__init__()
        self.base_layer = base_layer
        in_f = base_layer.in_features if hasattr(base_layer, "in_features") else base_layer.weight.shape[0]
        out_f = base_layer.out_features if hasattr(base_layer, "out_features") else base_layer.weight.shape[1]
        self.lora_A = nn.Parameter(torch.randn(rank, in_f))
        self.lora_B = nn.Parameter(torch.randn(out_f, rank))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base_layer(x)
        lora_out = (x @ self.lora_A.t()) @ self.lora_B.t()
        return base_out + lora_out


class DummyTransformerModel(nn.Module):
    """Mock transformer block with attention and FFN layers for testing."""

    def __init__(self):
        super().__init__()
        # Attention projections
        self.attn_q = nn.Linear(32, 32)
        self.attn_k = nn.Linear(32, 32)
        self.attn_v = Conv1D(32, 32)  # Conv1D in attention
        self.attn_out = DummyLoRALinear(nn.Linear(32, 32))  # Pre-existing LoRA in attention

        # FFN projections
        self.ffn_lin1 = nn.Linear(32, 64)
        self.ffn_lin2 = nn.Linear(64, 32)

        # Classification head
        self.classifier = nn.Linear(32, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q = self.attn_q(x)
        k = self.attn_k(x)
        v = self.attn_v(x)
        attn = self.attn_out(q + k + v)
        h = torch.relu(self.ffn_lin1(attn))
        h = self.ffn_lin2(h)
        return self.classifier(h)


class TestCacheNotesHardened(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)

    def test_01_cachenotes_lora_linear_forward_exactness(self):
        """Verify CacheNotesLoRALinear forward matches x @ weight.T + bias + lora_out."""
        in_f = 16
        out_f = 8
        r = 4
        x = torch.randn(2, 5, in_f)
        w = torch.randn(out_f, in_f)
        b = torch.randn(out_f)

        cn_layer = CacheNotesLoRALinear(in_f, out_f, weight=w, bias=b, lora_r=r, lora_alpha=1.0)
        out = cn_layer(x)

        # Reference forward
        x_flat = x.reshape(-1, in_f)
        ref_base = F.linear(x_flat, w, b)
        ref_lora = F.linear(F.linear(x_flat, cn_layer.lora_A), cn_layer.lora_B) * (1.0 / r)
        ref_out = (ref_base + ref_lora).reshape(2, 5, out_f)

        diff = (out - ref_out).abs().max().item()
        self.assertLess(diff, 1e-5)

    def test_02_lora_and_conv1d_support(self):
        """Verify existing LoRALinear base_layer is compressed without dropping adapters, and Conv1D works."""
        model = DummyTransformerModel()
        compressor = CacheNotesCompressor(
            model,
            keep_ratio=0.5,
            lora_r=16,
            max_error=0.45,
            target_attention_only=True,
        )
        compressed_model, stats = compressor.compress()

        # Check attn_out is still the DummyLoRALinear wrapper
        self.assertIsInstance(model.attn_out, DummyLoRALinear)
        # Check LoRA parameters are preserved and trainable
        self.assertTrue(model.attn_out.lora_A.requires_grad)
        self.assertTrue(model.attn_out.lora_B.requires_grad)
        # Base layer weight was frozen
        self.assertFalse(model.attn_out.base_layer.weight.requires_grad)

        # Check Conv1D layer was compressed
        self.assertIn("attn_v", stats["compressed_layers"])

    def test_03_target_attention_only_skips_ffn(self):
        """Verify target_attention_only=True skips FFN and classifier."""
        model = DummyTransformerModel()
        compressor = CacheNotesCompressor(
            model,
            keep_ratio=0.5,
            target_attention_only=True,
            max_error=0.45,
        )
        compressed_model, stats = compressor.compress()

        compressed = stats["compressed_layers"]
        self.assertIn("attn_q", compressed)
        self.assertIn("attn_k", compressed)
        self.assertIn("attn_v", compressed)
        self.assertIn("attn_out", compressed)

        self.assertNotIn("ffn_lin1", compressed)
        self.assertNotIn("ffn_lin2", compressed)
        self.assertNotIn("classifier", compressed)

    def test_04_fallback_and_freezing(self):
        """Verify layer exceeding max_error falls back and has its parameters frozen."""
        model = DummyTransformerModel()
        compressor = CacheNotesCompressor(
            model,
            keep_ratio=0.1,
            max_error=1e-5,  # Impossible threshold forces fallback
            fallback_to_original=True,
            target_attention_only=True,
        )
        compressed_model, stats = compressor.compress()

        self.assertEqual(len(stats["compressed_layers"]), 0)
        self.assertGreater(len(stats["fallback_layers"]), 0)

        for name in stats["fallback_layers"]:
            mod = dict(model.named_modules())[name]
            for p in mod.parameters():
                self.assertFalse(p.requires_grad, f"Parameter in fallback layer {name} was not frozen!")

    def test_05_kv_cache_pruning_and_sink_preservation(self):
        """Verify CacheNotesKVCache prunes tokens while strictly preserving initial sink tokens."""
        kv_cache = CacheNotesKVCache(max_cache_len=8, task_type="qa", keep_ratio=0.5, num_sinks=2)

        B, H, D = 1, 2, 16
        # Generate 12 tokens (exceeding max_cache_len 8)
        new_k = torch.randn(B, H, 12, D)
        new_v = torch.randn(B, H, 12, D)

        # Mark sink tokens with distinct values
        new_k[:, :, :2, :] = 999.0

        pruned_k, pruned_v = kv_cache.update_and_compress(new_k, new_v)

        # Pruned size must match max_cache_len
        self.assertEqual(pruned_k.shape[2], 8)
        self.assertEqual(pruned_v.shape[2], 8)

        # The first 2 tokens must remain the sink tokens (999.0)
        self.assertTrue(torch.allclose(pruned_k[:, :, :2, :], torch.tensor(999.0)))

    def test_06_factory_with_none_config_and_metrics(self):
        """Verify cachenotes_factory works with comp_config=None and records correct metrics."""
        model = DummyTransformerModel()
        compressed_model = cachenotes_factory(
            model,
            comp_config=None,
            keep_ratio=0.5,
            max_error=0.45,
            target_attention_only=True,
            lora_r=16,
        )
        stats = getattr(compressed_model, "compression_stats", {})
        self.assertIn("actual_ranks", stats)
        self.assertIn("avg_rank", stats)
        self.assertIn("kv_cache_budget_saving", stats)
        self.assertIn("avg_reconstruction_error", stats)
        self.assertGreater(len(stats["compressed_layers"]), 0)

    def test_07_cpt_generator_multiple_dimensions_and_training(self):
        """Verify learnable CPTGenerator produces valid vectors across arbitrary dimensions and trains properly."""
        model = DummyTransformerModel()
        compressor = CacheNotesCompressor(
            model,
            task_type="qa",
            task_description="question answering: answer questions from passage",
            use_learned_cpt=True,
            keep_ratio=0.5,
            target_attention_only=True,
        )
        # Test generation across multiple dimensions
        for dim in [32, 64, 128, 768, 1024, 2048]:
            vec = compressor.get_importance(dim, device=torch.device("cpu"))
            self.assertEqual(vec.shape, (dim,))
            self.assertTrue(torch.isfinite(vec).all())
            self.assertTrue((vec >= 0.1).all() and (vec <= 2.0).all())

        # Test training procedure
        loss = compressor.train_cpt_generator(epochs=2, lr=1e-3)
        self.assertTrue(compressor.cpt_generator_trained)
        self.assertGreater(loss, 0.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)

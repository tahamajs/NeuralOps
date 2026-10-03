"""
Unit tests for hardened AstroQuantizer with Fisher-Adaptive Bit Allocation.
Verifies:
1. allocate_fisher_bits solves the integer knapsack budget and averages to target_avg_bits.
2. quantize_dequantize_tensor with heterogeneous per-channel bits produces valid finite values and shapes.
3. LoRA wrapper and Conv1D detection and preservation.
4. target_attention_only=True skips FFN/MLP layers and retains attention projections.
5. High-error fallback triggers, restores weights, and freezes fallback parameters (requires_grad=False).
6. Factory works with comp_config=None and records effective_compression_rate and theoretical_vram_saving.
7. STE autograd backpropagation across heterogeneous bits.
"""
from __future__ import annotations

import unittest

import torch
import torch.nn as nn
from neuralops.compression_engine.astro_quantizer import (
    AstroLinear, AstroQuantizer, allocate_fisher_bits, astro_factory,
    quantize_dequantize_tensor)
from transformers.pytorch_utils import Conv1D


class DummyLoRALinear(nn.Module):
    """Mock LoRA linear layer wrapping base_layer."""

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
        self.attn_out = DummyLoRALinear(nn.Linear(32, 32))  # LoRA in attention

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


class TestAstroHardened(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)

    def test_01_allocate_fisher_bits_budget(self):
        """Verify allocate_fisher_bits matches target_avg_bits within 1/N tolerance."""
        N = 100
        importance = torch.rand(N) * 10.0
        
        # Test target 6.0
        bits_6 = allocate_fisher_bits(importance, target_avg_bits=6.0, min_bits=2, max_bits=8)
        self.assertEqual(bits_6.shape, (N,))
        self.assertTrue((bits_6 >= 2).all() and (bits_6 <= 8).all())
        avg_6 = bits_6.float().mean().item()
        self.assertAlmostEqual(avg_6, 6.0, delta=1.0 / N)

        # Check that highest importance channels got max_bits
        sorted_idx = torch.argsort(importance, descending=True)
        self.assertEqual(bits_6[sorted_idx[0]].item(), 8)

        # Test target 4.5
        bits_45 = allocate_fisher_bits(importance, target_avg_bits=4.5, min_bits=2, max_bits=8)
        avg_45 = bits_45.float().mean().item()
        self.assertAlmostEqual(avg_45, 4.5, delta=1.0 / N)

    def test_02_per_channel_quantization_shapes_and_values(self):
        """Verify heterogeneous per-channel quantization produces correct shapes and finite numbers."""
        W = torch.randn(8, 16)
        channel_bits = torch.tensor([2, 3, 4, 5, 6, 7, 8, 4])
        
        # Symmetric
        W_q_sym = quantize_dequantize_tensor(
            W, per_channel=True, per_channel_bits=channel_bits, symmetric=True
        )
        self.assertEqual(W_q_sym.shape, W.shape)
        self.assertTrue(torch.isfinite(W_q_sym).all())

        # Asymmetric
        W_q_asym = quantize_dequantize_tensor(
            W, per_channel=True, per_channel_bits=channel_bits, symmetric=False
        )
        self.assertEqual(W_q_asym.shape, W.shape)
        self.assertTrue(torch.isfinite(W_q_asym).all())

    def test_03_ste_autograd_flow(self):
        """Verify STE backpropagation flows gradients across heterogeneous per-channel bit allocations."""
        linear = nn.Linear(16, 8)
        channel_bits = torch.tensor([2, 4, 6, 8, 4, 6, 2, 8])
        astro_layer = AstroLinear.from_linear(
            linear, per_channel=True, per_channel_bits=channel_bits
        )
        astro_layer.train()

        x = torch.randn(4, 16, requires_grad=True)
        out = astro_layer(x)
        loss = out.sum()
        loss.backward()

        self.assertIsNotNone(astro_layer.weight.grad)
        self.assertEqual(astro_layer.weight.grad.shape, (8, 16))
        self.assertTrue(torch.isfinite(astro_layer.weight.grad).all())
        self.assertTrue(torch.count_nonzero(astro_layer.weight.grad) > 0)

    def test_04_target_attention_only_skips_ffn(self):
        """Verify target_attention_only=True only compresses attention layers and skips FFN and classifier."""
        model = DummyTransformerModel()
        quantizer = AstroQuantizer(
            model,
            bits=8,
            target_attention_only=True,
            use_fisher_adaptive_bits=False,
            max_error=0.50,
        )
        compressed_model, stats = quantizer.compress()

        compressed = stats["compressed_layers"]
        # Attention layers must be compressed
        self.assertIn("attn_q", compressed)
        self.assertIn("attn_k", compressed)
        self.assertIn("attn_v", compressed)
        self.assertIn("attn_out", compressed)

        # FFN and classifier must NOT be compressed
        self.assertNotIn("ffn_lin1", compressed)
        self.assertNotIn("ffn_lin2", compressed)
        self.assertNotIn("classifier", compressed)

    def test_05_lora_conv1d_support_and_trainability(self):
        """Verify LoRALinear base_layer is quantized while LoRA adapters remain trainable."""
        model = DummyTransformerModel()
        quantizer = AstroQuantizer(
            model,
            bits=8,
            use_ste=True,
            target_attention_only=True,
            max_error=0.50,
        )
        compressed_model, stats = quantizer.compress()

        # Check attn_out is still a DummyLoRALinear
        self.assertIsInstance(model.attn_out, DummyLoRALinear)
        # Check base_layer is now AstroLinear
        self.assertIsInstance(model.attn_out.base_layer, AstroLinear)
        # Check lora_A and lora_B are trainable
        self.assertTrue(model.attn_out.lora_A.requires_grad)
        self.assertTrue(model.attn_out.lora_B.requires_grad)

    def test_06_fallback_and_freezing(self):
        """Verify layer exceeding max_error falls back to original and has its parameters frozen."""
        model = DummyTransformerModel()
        # Set impossible max_error=0.00001 to force all layers to fallback
        quantizer = AstroQuantizer(
            model,
            bits=4,
            max_error=1e-5,
            fallback_to_original=True,
            target_attention_only=True,
        )
        compressed_model, stats = quantizer.compress()

        self.assertEqual(len(stats["compressed_layers"]), 0)
        self.assertGreater(len(stats["fallback_layers"]), 0)

        # Verify fallback parameters are frozen
        for name in stats["fallback_layers"]:
            mod = dict(model.named_modules())[name]
            for p in mod.parameters():
                self.assertFalse(p.requires_grad, f"Parameter in fallback layer {name} was not frozen!")

    def test_07_factory_with_none_config_and_metrics(self):
        """Verify astro_factory works with comp_config=None and returns valid metrics."""
        model = DummyTransformerModel()
        compressed_model = astro_factory(
            model,
            comp_config=None,
            bits=8,
            target_attention_only=True,
            use_fisher_adaptive_bits=True,
            target_avg_bits=6.0,
            max_error=0.50,
        )
        stats = getattr(compressed_model, "compression_stats", {})
        self.assertIn("actual_average_bits", stats)
        self.assertIn("effective_compression_rate", stats)
        self.assertIn("theoretical_vram_saving", stats)
        self.assertAlmostEqual(stats["actual_average_bits"], 6.0, delta=0.1)
        self.assertAlmostEqual(stats["effective_compression_rate"], 16.0 / 6.0, delta=0.1)


if __name__ == "__main__":
    unittest.main(verbosity=2)

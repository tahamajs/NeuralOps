#!/usr/bin/env python3
"""
Unit tests for production-hardened ARMOR Pruner.

Verifies:
1. Support for LoRALinear and Conv1D (transposition, preserving LoRA wrappers, trainable adapters).
2. Factory robustness when comp_config is None, dict, or object.
3. Reconstruction error validation and fallback mechanism when error > max_error with parameter freezing.
4. target_attention_only=True prunes attention layers while strictly skipping FFN/MLP and classifier.
5. Metrics include actual sparsity and effective parameter ratio.
6. ARMORLinear forward matches x @ (sparse_weight * mask)ᵀ + scaling * (x @ lora_Aᵀ) @ lora_Bᵀ + bias.
"""

import math
import sys
import unittest
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

# Ensure framework can be imported
HERE = Path(__file__).resolve().parent
CODES_DIR = HERE.parent.parent
if str(CODES_DIR) not in sys.path:
    sys.path.insert(0, str(CODES_DIR))

from framework.compression_engine.armor_pruner import (ARMORLinear,
                                                       ARMORPruner,
                                                       armor_pruner_factory)
from framework.utils.compat import get_recovery_params


class MockConv1D(nn.Module):
    """Mock of HuggingFace Conv1D (GPT-2 style) with weight shape [in_features, out_features]."""

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.nf = out_features
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.randn(in_features, out_features) * 0.02)
        self.bias = nn.Parameter(torch.zeros(out_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        size_out = x.size()[:-1] + (self.nf,)
        x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        x = x.view(*size_out)
        return x


class MockLoRALinear(nn.Module):
    """Mock of LoRALinear wrapper around a base layer."""

    def __init__(self, base_layer: nn.Module, r: int = 16):
        super().__init__()
        self.base_layer = base_layer
        self.r = r
        in_dim = getattr(base_layer, "in_features", None)
        out_dim = getattr(base_layer, "out_features", getattr(base_layer, "nf", None))
        if in_dim is None and hasattr(base_layer, "weight"):
            in_dim = base_layer.weight.shape[0] if hasattr(base_layer, "nf") else base_layer.weight.shape[1]
        if out_dim is None and hasattr(base_layer, "weight"):
            out_dim = base_layer.weight.shape[1] if hasattr(base_layer, "nf") else base_layer.weight.shape[0]

        self.lora_A = nn.Parameter(torch.randn(r, in_dim) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(out_dim, r))
        self.scaling = 1.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base_layer(x)
        lora_out = (x @ self.lora_A.t()) @ self.lora_B.t() * self.scaling
        return base_out + lora_out


class ToyAttention(nn.Module):
    def __init__(self, hidden_dim: int = 64):
        super().__init__()
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.c_attn = MockConv1D(hidden_dim, hidden_dim)
        self.o_proj = MockLoRALinear(nn.Linear(hidden_dim, hidden_dim), r=8)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.o_proj(self.q_proj(x))


class ToyTransformer(nn.Module):
    def __init__(self, hidden_dim: int = 64, num_classes: int = 2):
        super().__init__()
        self.attn = ToyAttention(hidden_dim)
        self.mlp = nn.ModuleDict({
            "gate_proj": nn.Linear(hidden_dim, hidden_dim * 2),
            "up_proj": nn.Linear(hidden_dim, hidden_dim * 2),
            "down_proj": nn.Linear(hidden_dim * 2, hidden_dim),
        })
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.attn(x)
        h = self.mlp["down_proj"](torch.relu(self.mlp["gate_proj"](h)))
        return self.classifier(h)


class TestARMORHardened(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)

    def test_lora_linear_and_conv1d_detection(self):
        """1. Verify LoRALinear and Conv1D detection, shape preservation, and LoRA preservation."""
        model = ToyTransformer()
        orig_conv1d_shape = model.attn.c_attn.weight.shape  # [64, 64]

        compressor = ARMORPruner(
            model,
            rank=8,
            structured_24=True,
            max_error=0.60,
            target_attention_only=True,
            decouple=True,
        )
        compressed_model, stats = compressor.compress()

        # Check Conv1D was replaced by ARMORLinear (or preserved shape)
        c_attn_mod = compressed_model.attn.c_attn
        self.assertTrue(isinstance(c_attn_mod, (ARMORLinear, MockConv1D)))
        self.assertEqual(c_attn_mod.out_features if hasattr(c_attn_mod, "out_features") else c_attn_mod.nf, 64)

        # Check LoRALinear wrapper preserved
        self.assertIsInstance(compressed_model.attn.o_proj, MockLoRALinear)
        self.assertTrue(compressed_model.attn.o_proj.lora_A.requires_grad)
        self.assertTrue(compressed_model.attn.o_proj.lora_B.requires_grad)

        # Base layer inside LoRALinear should be ARMORLinear
        self.assertIsInstance(compressed_model.attn.o_proj.base_layer, ARMORLinear)
        self.assertFalse(compressed_model.attn.o_proj.base_layer.sparse_weight.requires_grad)
        self.assertTrue(compressed_model.attn.o_proj.base_layer.lora_A.requires_grad)
        self.assertTrue(compressed_model.attn.o_proj.base_layer.lora_B.requires_grad)

    def test_factory_with_none_config(self):
        """2. Verify armor_pruner_factory works safely with None, dict, and config object."""
        model1 = ToyTransformer()
        comp1 = armor_pruner_factory(model1, comp_config=None, target_attention_only=True)
        self.assertIsNotNone(comp1)

        model2 = ToyTransformer()
        comp2 = armor_pruner_factory(model2, comp_config={"armor_rank": 8, "armor_max_error": 0.45}, target_attention_only=True)
        self.assertIsNotNone(comp2)

        class DummyConfig:
            armor_rank = 16
            armor_max_error = 0.45
            armor_sparsity = 0.5

        model3 = ToyTransformer()
        comp3 = armor_pruner_factory(model3, comp_config=DummyConfig(), target_attention_only=True)
        self.assertIsNotNone(comp3)

    def test_reconstruction_error_and_fallback(self):
        """3. Verify error thresholding triggers fallback to original weights and freezes parameters."""
        model = ToyTransformer()
        orig_q_w = model.attn.q_proj.weight.clone()

        compressor = ARMORPruner(
            model,
            max_error=0.001,  # triggers 100% fallback
            fallback_to_original=True,
            target_attention_only=True,
        )
        compressed_model, stats = compressor.compress()

        self.assertGreater(len(compressor.fallback_layers), 0)
        self.assertIn("attn.q_proj", compressor.fallback_layers)
        self.assertTrue(torch.equal(model.attn.q_proj.weight, orig_q_w))

        # Fallback parameters must be frozen
        for name in compressor.fallback_layers:
            mod = dict(model.named_modules())[name]
            for p in mod.parameters():
                self.assertFalse(p.requires_grad, f"Fallback layer {name} parameter was not frozen!")

    def test_target_attention_only_skips_ffn(self):
        """4. Verify target_attention_only=True strictly skips MLP/FFN and classifier."""
        model = ToyTransformer()
        orig_mlp_down = model.mlp["down_proj"].weight.clone()
        orig_classifier = model.classifier.weight.clone()

        compressor = ARMORPruner(
            model,
            max_error=0.60,
            target_attention_only=True,
        )
        _, stats = compressor.compress()

        for name in compressor.compressed_layers:
            self.assertTrue(name.startswith("attn."))
            self.assertFalse("mlp" in name)
            self.assertFalse("classifier" in name)

        self.assertTrue(torch.equal(model.mlp["down_proj"].weight, orig_mlp_down))
        self.assertTrue(torch.equal(model.classifier.weight, orig_classifier))

    def test_metrics_include_sparsity_and_effective_ratio(self):
        """5. Verify stats include actual sparsity and effective parameter ratio."""
        model = ToyTransformer()
        compressor = ARMORPruner(
            model,
            rank=8,
            structured_24=True,
            max_error=0.60,
            target_attention_only=True,
            decouple=True,
        )
        _, stats = compressor.compress()

        self.assertIn("actual_sparsity", stats)
        self.assertAlmostEqual(stats["actual_sparsity"], 0.5, places=1)
        self.assertIn("effective_parameter_ratio", stats)
        self.assertIn("base_sparse_params", stats)
        self.assertIn("correction_params", stats)
        self.assertGreater(stats["correction_params"], 0)
        self.assertGreater(stats["base_sparse_params"], 0)

    def test_armor_linear_forward_exactness(self):
        """6. Verify ARMORLinear forward matches x @ (sparse_weight * mask)ᵀ + scaling * (x @ lora_Aᵀ) @ lora_Bᵀ + bias."""
        in_dim, out_dim, r = 32, 48, 8
        linear = nn.Linear(in_dim, out_dim, bias=True)
        mask = torch.randint(0, 2, (out_dim, in_dim)).float()
        lora_A = torch.randn(r, in_dim) * 0.05
        lora_B = torch.randn(out_dim, r) * 0.05
        lora_alpha = 2.0
        scaling = lora_alpha / r

        armor_layer = ARMORLinear.from_linear(
            linear,
            mask=mask,
            lora_A=lora_A,
            lora_B=lora_B,
            lora_alpha=lora_alpha,
        )

        x = torch.randn(4, 16, in_dim)

        # Expected calculation
        x_flat = x.reshape(-1, in_dim)
        w_eff = (linear.weight.data * mask)
        expected_sparse = F.linear(x_flat, w_eff)
        expected_lr = scaling * ((x_flat @ lora_A.t()) @ lora_B.t())
        expected_out = (expected_sparse + expected_lr + linear.bias.data).view(4, 16, out_dim)

        actual_out = armor_layer(x)
        self.assertTrue(torch.allclose(actual_out, expected_out, atol=1e-5))


if __name__ == "__main__":
    unittest.main(verbosity=2)

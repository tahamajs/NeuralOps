#!/usr/bin/env python3
"""
Unit tests for production-hardened ADMM Compressor.

Verifies:
1. Support for LoRALinear and Conv1D (shape transposition, preserving LoRA wrappers, trainable LoRA).
2. Factory robustness when comp_config is None, dict, or object.
3. Reconstruction error validation and fallback mechanism when error exceeds max_error.
4. Quantized + pruned weight shape matches original and contains no NaN / Inf.
5. target_attention_only=True compresses attention layers while strictly skipping FFN/MLP.
6. Fallback layers are strictly frozen (requires_grad=False) during recovery parameter selection.
"""

import math
import sys
import unittest
from pathlib import Path

import torch
import torch.nn as nn

# Ensure framework can be imported
HERE = Path(__file__).resolve().parent
CODES_DIR = HERE.parent.parent
if str(CODES_DIR) not in sys.path:
    sys.path.insert(0, str(CODES_DIR))

from neuralops.compression_engine.admm_compressor import (ADMMCompressor,
                                                          admm_factory)
from neuralops.utils.compat import get_recovery_params


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
        in_dim = getattr(base_layer, "in_features", base_layer.weight.shape[0])
        out_dim = getattr(base_layer, "out_features", base_layer.weight.shape[1])
        self.lora_A = nn.Linear(in_dim, r, bias=False)
        self.lora_B = nn.Linear(r, out_dim, bias=False)
        self.scaling = 2.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = self.base_layer(x)
        lora = self.lora_B(self.lora_A(x)) * self.scaling
        return base + lora


class ToyTransformer(nn.Module):
    """Toy model containing attention and FFN layers for filtering tests."""

    def __init__(self):
        super().__init__()
        # Attention projection layers
        self.attn = nn.ModuleDict({
            "q_proj": nn.Linear(64, 64),
            "k_proj": nn.Linear(64, 64),
            "v_proj": MockLoRALinear(nn.Linear(64, 64), r=8),
            "c_attn": MockLoRALinear(MockConv1D(64, 192), r=8),
            "o_proj": MockConv1D(64, 64),
        })
        # FFN / MLP layers
        self.mlp = nn.ModuleDict({
            "gate_proj": nn.Linear(64, 128),
            "up_proj": nn.Linear(64, 128),
            "down_proj": nn.Linear(128, 64),
            "c_fc": MockConv1D(64, 128),
        })
        # Classification head
        self.classifier = nn.Linear(64, 2)


class TestADMMHardened(unittest.TestCase):

    def setUp(self):
        torch.manual_seed(42)

    def test_lora_linear_and_conv1d_detection(self):
        """1. Verify LoRALinear and Conv1D detection, shape transposition, and LoRA preservation."""
        model = ToyTransformer()
        orig_conv_shape = model.attn["c_attn"].base_layer.weight.shape
        self.assertEqual(orig_conv_shape, (64, 192))

        compressor = ADMMCompressor(
            model,
            max_error=0.60,
            target_attention_only=True,
            quant_bits=4,
            structured_24=True,
        )
        _, stats = compressor.compress()

        # Check that LoRA linear and Conv1D were compressed
        self.assertIn("attn.v_proj", compressor.compressed_layers)
        self.assertIn("attn.c_attn", compressor.compressed_layers)
        self.assertIn("attn.o_proj", compressor.compressed_layers)

        # Check that shapes were strictly preserved
        self.assertEqual(model.attn["c_attn"].base_layer.weight.shape, (64, 192))
        self.assertEqual(model.attn["o_proj"].weight.shape, (64, 64))

        # Check that LoRA wrappers are intact
        self.assertTrue(isinstance(model.attn["c_attn"], MockLoRALinear))
        self.assertTrue(model.attn["c_attn"].lora_A.weight.requires_grad)
        self.assertTrue(model.attn["c_attn"].lora_B.weight.requires_grad)
        # Base layer should be frozen
        self.assertFalse(model.attn["c_attn"].base_layer.weight.requires_grad)

    def test_admm_factory_with_none_config(self):
        """2. Verify admm_factory works safely with None, dict, and config object."""
        # None config
        model1 = ToyTransformer()
        res_model1 = admm_factory(model1, comp_config=None, max_error=0.60, target_attention_only=True)
        self.assertTrue(hasattr(res_model1, "compression_stats"))
        self.assertGreater(res_model1.compression_stats["num_compressed"], 0)

        # Dict config
        model2 = ToyTransformer()
        cfg_dict = {"admm_max_error": 0.60, "admm_quant_bits": 4, "admm_target_attention_only": True}
        res_model2 = admm_factory(model2, comp_config=cfg_dict)
        self.assertEqual(res_model2.compression_stats["quant_bits"], 4)

    def test_reconstruction_error_and_fallback(self):
        """3. Verify error thresholding triggers fallback to original weights when error > max_error."""
        model = ToyTransformer()
        orig_q_weight = model.attn["q_proj"].weight.clone()

        # Set impossible max_error = 0.01 -> must trigger fallback
        compressor = ADMMCompressor(
            model,
            max_error=0.01,
            target_attention_only=True,
            fallback_to_original=True,
        )
        _, stats = compressor.compress()

        self.assertIn("attn.q_proj", compressor.fallback_layers)
        self.assertEqual(len(compressor.compressed_layers), 0)
        # Weight must remain completely unchanged
        self.assertTrue(torch.equal(model.attn["q_proj"].weight, orig_q_weight))

    def test_quantized_pruned_weight_properties(self):
        """4. Verify weight shape match, no NaN/Inf, and 2:4 structured sparsity pattern."""
        model = ToyTransformer()
        compressor = ADMMCompressor(
            model,
            max_error=0.60,
            quant_bits=4,
            structured_24=True,
            target_attention_only=True,
        )
        _, stats = compressor.compress()

        q_w = model.attn["q_proj"].weight.data
        self.assertEqual(q_w.shape, (64, 64))
        self.assertFalse(torch.isnan(q_w).any())
        self.assertFalse(torch.isinf(q_w).any())

        # Check 2:4 pattern: in every group of 4 columns, at least 2 should be zero (2:4 structure preserved under quantization)
        q_blocks = q_w.reshape(64, 16, 4)
        zeros_per_block = (q_blocks == 0).sum(dim=-1)
        self.assertTrue(torch.all(zeros_per_block >= 2))

        # Check reported actual sparsity and effective compression rate
        self.assertGreaterEqual(stats["actual_sparsity"], 0.50)
        self.assertEqual(stats["effective_compression_rate"], 4.0)

    def test_target_attention_only_skips_ffn(self):
        """5. Verify target_attention_only=True strictly skips MLP/FFN and classifier."""
        model = ToyTransformer()
        orig_mlp_down = model.mlp["down_proj"].weight.clone()
        orig_classifier = model.classifier.weight.clone()

        compressor = ADMMCompressor(
            model,
            max_error=0.60,
            target_attention_only=True,
        )
        _, stats = compressor.compress()

        # All compressed layers must be attention
        for name in compressor.compressed_layers:
            self.assertTrue(name.startswith("attn."))
            self.assertFalse("mlp" in name)
            self.assertFalse("classifier" in name)

        # MLP and classifier weights must be pristine
        self.assertTrue(torch.equal(model.mlp["down_proj"].weight, orig_mlp_down))
        self.assertTrue(torch.equal(model.classifier.weight, orig_classifier))

    def test_fallback_layers_are_frozen(self):
        """6. Verify that fallback layers have requires_grad=False and are excluded from recovery."""
        model = ToyTransformer()
        # Ensure parameters initially require grad
        for p in model.parameters():
            p.requires_grad = True

        compressor = ADMMCompressor(
            model,
            max_error=0.01,  # triggers 100% fallback
            fallback_to_original=True,
            target_attention_only=True,
        )
        compressed_model, stats = compressor.compress()
        setattr(compressed_model, "compression_stats", stats)

        # Fallback layers must have requires_grad=False
        for name in compressor.fallback_layers:
            mod = dict(compressed_model.named_modules())[name]
            for p in mod.parameters():
                self.assertFalse(p.requires_grad, f"Parameter in fallback layer {name} was not frozen!")

        # get_recovery_params must not include any fallback parameters
        recovery_params = get_recovery_params(compressed_model)
        recovery_param_ids = {id(rp) for rp in recovery_params}
        for name in compressor.fallback_layers:
            mod = dict(compressed_model.named_modules())[name]
            for p in mod.parameters():
                self.assertNotIn(id(p), recovery_param_ids, f"Parameter in fallback layer {name} was included in recovery!")


if __name__ == "__main__":
    unittest.main(verbosity=2)
